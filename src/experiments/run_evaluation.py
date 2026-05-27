"""
run_evaluation.py
-----------------
Unified evaluation that replaces the scattered eval scripts. Produces
publication-grade outputs in a single pass:

  1.  Train word-TFIDF + char-TFIDF detectors on the fixed train split.
  2.  For each detector × {P0_test, P1_test_std, P2_test_std,
                            P1_test_sim, P2_test_sim}:
        - point metrics (acc, P, R, F1, AUROC, MCC, Brier, ECE)
        - independent bootstrap 95% CIs on each metric
  3.  Paired-bootstrap CIs on metric *differences* between adjacent
      paraphrase stages (P0 → P1, P1 → P2, P0 → P2) per track per detector.
      This is the correct way to report degradation Δ with uncertainty.
  4.  Area-Under-Robustness-Curve (AURC) and linear degradation slope
      per detector per track per metric — scalar summary of total drop.
  5.  Hardness-stratified evaluation under THREE hardness definitions
      (margin, abs_margin, entropy) so the bucket finding does not depend
      on a single circular definition.
  6.  Per-bucket metrics with bootstrap CIs, computed independently
      for each detector and each hardness definition.

Outputs (results/eval/)
  metrics_point_and_ci.csv       — main table: detector, split, metric,
                                   point, ci_lo, ci_hi
  metrics_paired_diff.csv        — degradation Δ table with paired CI
  aurc_summary.csv               — AURC + slope per detector × track × metric
  hardness_buckets_multi.csv     — per-bucket metrics across 3 hardness defs
  bucket_overlap_kendall.json    — how similar are the 3 hardness rankings?
  full_predictions.parquet       — raw predictions for downstream analyses
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.detectors import build_char_tfidf_lr, build_word_tfidf_lr
from lib.hardness import (
    all_text_based_hardness,
    hardness_cross_detector,
    hardness_from_probs,
)
from lib.io import load_jsonl, load_test_ids, to_xy
from lib.metrics import (
    METRIC_FUNCS,
    area_under_robustness_curve,
    bootstrap_metric_ci,
    paired_bootstrap_diff_ci,
    point_metrics,
    relative_degradation_slope,
)


# ── config ────────────────────────────────────────────────────────────────────
N_BOOT = 2000
SEED = 42
METRICS_FOR_CI = ["acc", "f1", "auroc", "mcc", "brier", "ece"]
METRICS_FOR_DIFF = ["acc", "f1", "auroc", "mcc", "brier", "ece"]
# Hardness definitions to test.
#   "margin_self_word"  : margin from word detector (CIRCULAR if evaluating word)
#   "margin_self_char"  : margin from char detector (CIRCULAR if evaluating char)
#   "margin_cross"      : margin from the OTHER detector (non-circular)
#   "readability_fk"    : Flesch-Kincaid grade level (TEXT-ONLY, detector-independent)
#   "length"            : word count (text-only)
#   "ttr"               : type-token ratio (text-only)
HARDNESS_METHODS = ["margin_self_word", "margin_self_char", "margin_cross",
                    "readability_fk", "length", "ttr"]
TRACKS = {
    "standard":   paths.STANDARD_TRACK,
    "simplified": paths.SIMPLIFIED_TRACK,
}

OUT_DIR = paths.RESULTS / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────


def load_training_data() -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    p0_all = load_jsonl(paths.P0_PATH)
    train_ids = load_test_ids(paths.TRAIN_IDS)
    val_ids = load_test_ids(paths.VAL_IDS)
    train_rows = [r for r in p0_all if r["id"] in train_ids]
    val_rows = [r for r in p0_all if r["id"] in val_ids]
    X_tr, y_tr = to_xy(train_rows)
    X_va, y_va = to_xy(val_rows)
    return X_tr, y_tr, X_va, y_va


def load_eval_splits() -> dict[str, list[dict]]:
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all = load_jsonl(paths.P0_PATH)
    p0_test = [r for r in p0_all if r["id"] in test_ids]

    return {
        "P0_test":            p0_test,
        "P1_test_standard":   load_jsonl(paths.PARAPHRASE_PATHS["P1_test_standard"]),
        "P2_test_standard":   load_jsonl(paths.PARAPHRASE_PATHS["P2_test_standard"]),
        "P1_test_simplified": load_jsonl(paths.PARAPHRASE_PATHS["P1_test_simplified"]),
        "P2_test_simplified": load_jsonl(paths.PARAPHRASE_PATHS["P2_test_simplified"]),
    }


def evaluate_splits(detector, splits: dict[str, list[dict]]) -> dict:
    """Returns per-split dict of (y_true, y_pred, y_prob)."""
    out = {}
    for name, rows in splits.items():
        X, y = to_xy(rows)
        prob = detector.predict_proba(X)
        pred = (prob >= 0.5).astype(int)
        out[name] = dict(y_true=y, y_pred=pred, y_prob=prob, ids=[r["id"] for r in rows])
    return out


# ── Section 1 + 2: point metrics + bootstrap CIs ──────────────────────────────


def compute_point_and_ci(detector_name: str, eval_data: dict) -> pd.DataFrame:
    rows = []
    for split_name, d in eval_data.items():
        pts = point_metrics(d["y_true"], d["y_pred"], d["y_prob"])
        for m in METRICS_FOR_CI:
            ci = bootstrap_metric_ci(
                d["y_true"], d["y_pred"], d["y_prob"],
                metric_fn=METRIC_FUNCS[m],
                n_boot=N_BOOT, seed=SEED,
            )
            rows.append({
                "detector":    detector_name,
                "split":       split_name,
                "n":           int(len(d["y_true"])),
                "metric":      m,
                "point":       round(ci.point, 4),
                "ci_lo":       round(ci.lo, 4),
                "ci_hi":       round(ci.hi, 4),
                "ci_width":    round(ci.hi - ci.lo, 4),
            })
    return pd.DataFrame(rows)


# ── Section 3: paired-bootstrap on differences ────────────────────────────────


def compute_paired_diffs(detector_name: str, eval_data: dict) -> pd.DataFrame:
    """Δ = metric(B) − metric(A), paired bootstrap CI."""
    pairs: list[tuple[str, str, str]] = []
    for track_name, stages in TRACKS.items():
        for i in range(len(stages)):
            for j in range(i + 1, len(stages)):
                a, b = stages[i], stages[j]
                pairs.append((a, b, track_name))

    rows = []
    for a, b, track in pairs:
        da, db = eval_data[a], eval_data[b]
        for m in METRICS_FOR_DIFF:
            ci = paired_bootstrap_diff_ci(
                y_true=da["y_true"],
                y_pred_a=da["y_pred"], y_prob_a=da["y_prob"],
                y_pred_b=db["y_pred"], y_prob_b=db["y_prob"],
                metric_fn=METRIC_FUNCS[m],
                n_boot=N_BOOT, seed=SEED,
            )
            sig = (ci.lo > 0) or (ci.hi < 0)
            rows.append({
                "detector":     detector_name,
                "track":        track,
                "comparison":   f"{a} -> {b}",
                "split_a":      a,
                "split_b":      b,
                "metric":       m,
                "delta":        round(ci.point, 4),
                "ci_lo":        round(ci.lo, 4),
                "ci_hi":        round(ci.hi, 4),
                "significant":  bool(sig),
            })
    return pd.DataFrame(rows)


# ── Section 4: AURC and degradation slope ─────────────────────────────────────


def compute_aurc(detector_name: str, eval_data: dict) -> pd.DataFrame:
    rows = []
    for track_name, stages in TRACKS.items():
        for m in METRICS_FOR_CI:
            vals = []
            for s in stages:
                d = eval_data[s]
                vals.append(METRIC_FUNCS[m](d["y_true"], d["y_pred"], d["y_prob"]))
            aurc = area_under_robustness_curve(vals)
            slope = relative_degradation_slope(vals)
            rows.append({
                "detector":  detector_name,
                "track":     track_name,
                "metric":    m,
                "stage_0":   round(vals[0], 4) if vals else None,
                "stage_1":   round(vals[1], 4) if len(vals) > 1 else None,
                "stage_2":   round(vals[2], 4) if len(vals) > 2 else None,
                "aurc":      round(aurc, 4),
                "slope":     round(slope, 4),
                "drop_p0_p2": round(vals[0] - vals[-1], 4),
            })
    return pd.DataFrame(rows)


# ── Section 5 + 6: hardness buckets under multiple definitions ────────────────


def build_hardness_assignments(
    detector_name: str,
    eval_data: dict,
    all_detector_data: dict,
) -> dict[str, dict[str, str]]:
    """
    Build id -> bucket maps for EVERY hardness definition.
    `all_detector_data` = dict mapping detector_name -> evaluation data for
    that detector (used to compute cross-detector hardness).
    """
    p0 = eval_data["P0_test"]
    p0_ids = p0["ids"]
    p0_texts = []
    # We need P0 texts for text-based hardness; not stored in eval_data — pull
    # them from p0_test rows via the IDs.
    # The eval_data only has y_true, y_pred, y_prob, ids — text isn't there.
    # We need to read it from the raw P0 file. To avoid re-reading, accept
    # texts as a parallel param... but simpler: read here.
    p0_raw = load_jsonl(paths.P0_PATH)
    p0_by_id = {r["id"]: r for r in p0_raw}
    p0_texts = [p0_by_id[i]["text"] for i in p0_ids]

    assignments: dict[str, dict[str, str]] = {}

    # Self-margin from word detector (circular if evaluating word)
    word_p0_probs = all_detector_data["word_tfidf_lr"]["P0_test"]["y_prob"]
    h_word = hardness_from_probs(word_p0_probs, method="margin")
    assignments["margin_self_word"] = dict(zip(p0_ids, h_word.buckets))

    # Self-margin from char detector
    char_p0_probs = all_detector_data["char_tfidf_lr"]["P0_test"]["y_prob"]
    h_char = hardness_from_probs(char_p0_probs, method="margin")
    assignments["margin_self_char"] = dict(zip(p0_ids, h_char.buckets))

    # Cross-detector margin: when evaluating one detector, use the OTHER
    # detector's margin to define hardness. This is the key non-circular
    # variant for each evaluated detector.
    if detector_name == "word_tfidf_lr":
        h_cross = hardness_cross_detector(char_p0_probs, method="margin")
    else:
        h_cross = hardness_cross_detector(word_p0_probs, method="margin")
    assignments["margin_cross"] = dict(zip(p0_ids, h_cross.buckets))

    # Text-only hardness (detector-independent)
    text_hardness = all_text_based_hardness(p0_texts)
    for name, h in text_hardness.items():
        assignments[name] = dict(zip(p0_ids, h.buckets))

    return assignments


def compute_hardness_kendall_concordance(
    detector_name: str,
    eval_data: dict,
    all_detector_data: dict,
) -> dict:
    """Kendall's tau between SCORE vectors (not buckets) under all definitions
    measured on the P0 reference split."""
    p0_ids = eval_data["P0_test"]["ids"]
    p0_raw = load_jsonl(paths.P0_PATH)
    p0_by_id = {r["id"]: r for r in p0_raw}
    p0_texts = [p0_by_id[i]["text"] for i in p0_ids]

    word_p0_probs = all_detector_data["word_tfidf_lr"]["P0_test"]["y_prob"]
    char_p0_probs = all_detector_data["char_tfidf_lr"]["P0_test"]["y_prob"]

    scores: dict[str, np.ndarray] = {}
    scores["margin_self_word"] = hardness_from_probs(word_p0_probs, "margin").scores
    scores["margin_self_char"] = hardness_from_probs(char_p0_probs, "margin").scores
    cross_probs = char_p0_probs if detector_name == "word_tfidf_lr" else word_p0_probs
    scores["margin_cross"] = hardness_cross_detector(cross_probs, "margin").scores
    text_h = all_text_based_hardness(p0_texts)
    for name, h in text_h.items():
        scores[name] = h.scores

    keys = list(scores.keys())
    taus = {}
    for i, a in enumerate(keys):
        for b in keys[i+1:]:
            tau, p = kendalltau(scores[a], scores[b])
            taus[f"{a}__vs__{b}"] = {
                "tau": round(float(tau), 4),
                "pvalue": float(p),
            }
    return taus


def compute_hardness_buckets(
    detector_name: str,
    eval_data: dict,
    all_detector_data: dict,
) -> tuple[pd.DataFrame, dict]:
    """Per-bucket evaluation under EVERY hardness definition, with bootstrap CIs."""
    from lib.metrics import METRIC_FUNCS

    assignments = build_hardness_assignments(detector_name, eval_data, all_detector_data)
    kendall_taus = compute_hardness_kendall_concordance(detector_name, eval_data, all_detector_data)

    rows = []
    for method, id_to_bucket in assignments.items():
        for split_name, d in eval_data.items():
            ids = d["ids"]
            buckets = np.array([id_to_bucket.get(i, "Unknown") for i in ids])
            for b in ("Easy", "Medium", "Hard"):
                mask = buckets == b
                if mask.sum() == 0:
                    continue
                yt, yp, ypr = d["y_true"][mask], d["y_pred"][mask], d["y_prob"][mask]
                pts = point_metrics(yt, yp, ypr)
                ci_f1 = bootstrap_metric_ci(
                    yt, yp, ypr, metric_fn=METRIC_FUNCS["f1"],
                    n_boot=N_BOOT, seed=SEED,
                )
                ci_acc = bootstrap_metric_ci(
                    yt, yp, ypr, metric_fn=METRIC_FUNCS["acc"],
                    n_boot=N_BOOT, seed=SEED,
                )
                rows.append({
                    "detector":   detector_name,
                    "hardness":   method,
                    "split":      split_name,
                    "bucket":     b,
                    "n":          int(mask.sum()),
                    "n_human":    int((yt == 0).sum()),
                    "n_llm":      int((yt == 1).sum()),
                    "acc":        round(pts["acc"], 4),
                    "acc_ci_lo":  round(ci_acc.lo, 4),
                    "acc_ci_hi":  round(ci_acc.hi, 4),
                    "f1":         round(pts["f1"], 4),
                    "f1_ci_lo":   round(ci_f1.lo, 4),
                    "f1_ci_hi":   round(ci_f1.hi, 4),
                    "auroc":      round(pts["auroc"], 4),
                    "ece":        round(pts["ece"], 4),
                })
    return pd.DataFrame(rows), kendall_taus


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    print("Loading data...")
    X_tr, y_tr, X_va, y_va = load_training_data()
    eval_splits_raw = load_eval_splits()

    # Print class balance for the record (this is a known issue)
    for name, rows in eval_splits_raw.items():
        n_h = sum(1 for r in rows if r["label"] == "human")
        n_l = sum(1 for r in rows if r["label"] == "llm")
        print(f"  {name}: n={len(rows)} (human={n_h}, llm={n_l})")

    print(f"\nTrain: {len(X_tr)} | Val: {len(X_va)} | Test: {len(eval_splits_raw['P0_test'])}")

    print("\nTraining detectors...")
    word_det = build_word_tfidf_lr(X_tr, y_tr, seed=SEED)
    char_det = build_char_tfidf_lr(X_tr, y_tr, seed=SEED)
    print(f"  word vocab size: {len(word_det.vectorizer.vocabulary_):,}")
    print(f"  char vocab size: {len(char_det.vectorizer.vocabulary_):,}")

    # Persist artifacts for downstream scripts
    joblib.dump(word_det.vectorizer, paths.RESULTS / "vectorizer.joblib")
    joblib.dump(word_det.classifier, paths.RESULTS / "model.joblib")
    joblib.dump(char_det.vectorizer, paths.RESULTS / "vectorizer_char.joblib")
    joblib.dump(char_det.classifier, paths.RESULTS / "model_char.joblib")

    # ── evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating word detector on all splits...")
    word_data = evaluate_splits(word_det, eval_splits_raw)
    print("Evaluating char detector on all splits...")
    char_data = evaluate_splits(char_det, eval_splits_raw)

    detectors = {
        "word_tfidf_lr": (word_det, word_data),
        "char_tfidf_lr": (char_det, char_data),
    }

    # ── Section 1+2 ───────────────────────────────────────────────────────────
    print("\n[1/4] Point metrics + bootstrap CIs...")
    dfs = []
    for name, (_, data) in detectors.items():
        dfs.append(compute_point_and_ci(name, data))
    df_point = pd.concat(dfs, ignore_index=True)
    df_point.to_csv(OUT_DIR / "metrics_point_and_ci.csv", index=False)

    # ── Section 3 ─────────────────────────────────────────────────────────────
    print("[2/4] Paired-bootstrap CIs on degradation deltas...")
    dfs = []
    for name, (_, data) in detectors.items():
        dfs.append(compute_paired_diffs(name, data))
    df_diff = pd.concat(dfs, ignore_index=True)
    df_diff.to_csv(OUT_DIR / "metrics_paired_diff.csv", index=False)

    # ── Section 4 ─────────────────────────────────────────────────────────────
    print("[3/4] AURC + slope...")
    dfs = []
    for name, (_, data) in detectors.items():
        dfs.append(compute_aurc(name, data))
    df_aurc = pd.concat(dfs, ignore_index=True)
    df_aurc.to_csv(OUT_DIR / "aurc_summary.csv", index=False)

    # ── Section 5+6 ───────────────────────────────────────────────────────────
    print(f"[4/4] Hardness buckets ({len(HARDNESS_METHODS)} definitions, "
          f"incl. non-circular) with per-bucket CIs...")
    dfs = []
    taus_all = {}
    # Build a dict of detector_name -> eval_data for cross-detector lookups
    all_detector_data = {name: data for name, (_, data) in detectors.items()}
    for name, (_, data) in detectors.items():
        df_b, taus = compute_hardness_buckets(name, data, all_detector_data)
        dfs.append(df_b)
        taus_all[name] = taus
    df_buckets = pd.concat(dfs, ignore_index=True)
    df_buckets.to_csv(OUT_DIR / "hardness_buckets_multi.csv", index=False)
    (OUT_DIR / "bucket_overlap_kendall.json").write_text(
        json.dumps(taus_all, indent=2), encoding="utf-8"
    )

    # ── Persist raw predictions for downstream scripts ────────────────────────
    pred_rows = []
    for det_name, (_, data) in detectors.items():
        for split_name, d in data.items():
            for i, pid in enumerate(d["ids"]):
                pred_rows.append({
                    "detector": det_name,
                    "split": split_name,
                    "id": pid,
                    "y_true": int(d["y_true"][i]),
                    "y_prob": float(d["y_prob"][i]),
                    "y_pred": int(d["y_pred"][i]),
                })
    df_preds = pd.DataFrame(pred_rows)
    df_preds.to_csv(OUT_DIR / "predictions.csv", index=False)

    # ── Headline report ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("HEADLINE NUMBERS")
    print("=" * 80)

    # 1. main metrics with CI for word detector
    print("\n[word_tfidf_lr] Main metrics with 95% CIs:")
    for split in ["P0_test", "P1_test_standard", "P2_test_standard",
                  "P1_test_simplified", "P2_test_simplified"]:
        f1_row = df_point[(df_point["detector"] == "word_tfidf_lr") &
                          (df_point["split"] == split) &
                          (df_point["metric"] == "f1")].iloc[0]
        acc_row = df_point[(df_point["detector"] == "word_tfidf_lr") &
                           (df_point["split"] == split) &
                           (df_point["metric"] == "acc")].iloc[0]
        print(f"  {split:<25}  F1={f1_row['point']:.4f} "
              f"[{f1_row['ci_lo']:.4f}, {f1_row['ci_hi']:.4f}]   "
              f"Acc={acc_row['point']:.4f} "
              f"[{acc_row['ci_lo']:.4f}, {acc_row['ci_hi']:.4f}]")

    # 2. AURC headline
    print("\nAURC (accuracy, higher = more robust):")
    for det in ["word_tfidf_lr", "char_tfidf_lr"]:
        for track in ["standard", "simplified"]:
            r = df_aurc[(df_aurc["detector"] == det) &
                        (df_aurc["track"] == track) &
                        (df_aurc["metric"] == "acc")].iloc[0]
            print(f"  {det:<15} {track:<11}  AURC_acc={r['aurc']:.4f}  "
                  f"slope={r['slope']:+.4f}  P0->P2 drop={r['drop_p0_p2']:.4f}")

    # 3. Hard bucket F1 across hardness defs (word detector)
    print("\n[word_tfidf_lr] Hard-bucket F1 across hardness definitions:")
    print(f"  {'hardness':<12} {'P0':<22} {'P2_std':<22} {'P2_sim':<22}")
    for h in HARDNESS_METHODS:
        row_p0 = df_buckets[(df_buckets["detector"] == "word_tfidf_lr") &
                            (df_buckets["hardness"] == h) &
                            (df_buckets["split"] == "P0_test") &
                            (df_buckets["bucket"] == "Hard")]
        row_std = df_buckets[(df_buckets["detector"] == "word_tfidf_lr") &
                             (df_buckets["hardness"] == h) &
                             (df_buckets["split"] == "P2_test_standard") &
                             (df_buckets["bucket"] == "Hard")]
        row_sim = df_buckets[(df_buckets["detector"] == "word_tfidf_lr") &
                             (df_buckets["hardness"] == h) &
                             (df_buckets["split"] == "P2_test_simplified") &
                             (df_buckets["bucket"] == "Hard")]
        if len(row_p0) and len(row_std) and len(row_sim):
            r_p0, r_std, r_sim = row_p0.iloc[0], row_std.iloc[0], row_sim.iloc[0]
            print(f"  {h:<12} {r_p0['f1']:.3f} [{r_p0['f1_ci_lo']:.2f},{r_p0['f1_ci_hi']:.2f}]   "
                  f"{r_std['f1']:.3f} [{r_std['f1_ci_lo']:.2f},{r_std['f1_ci_hi']:.2f}]   "
                  f"{r_sim['f1']:.3f} [{r_sim['f1_ci_lo']:.2f},{r_sim['f1_ci_hi']:.2f}]")

    # 4. Hardness concordance
    print("\nHardness concordance (Kendall's tau, word_tfidf_lr P0 probs):")
    for pair, info in taus_all["word_tfidf_lr"].items():
        print(f"  {pair:<35}  tau={info['tau']:+.4f}  p={info['pvalue']:.3e}")

    print(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
