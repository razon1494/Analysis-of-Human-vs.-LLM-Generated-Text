"""
run_multiseed.py
----------------
Multi-seed training-variance experiment.

Question
--------
The fixed 80/10/10 split with seed=42 gives one realization of train/val.
How sensitive are the reported numbers to that random choice?

Method
------
The test set has paraphrases generated only for the original 100 rows, so
we cannot vary the test split (it would invalidate the existing P1/P2 files).
We CAN vary train/val by drawing K stratified splits from the
non-test pool. For each seed:

  1. Stratified-resample train/val from the (P0 \ test) pool, preserving
     human:LLM ratios.
  2. Re-train both word-TFIDF and char-TFIDF detectors.
  3. Re-evaluate on the FIXED test set across all paraphrase stages.
  4. Re-compute hardness assignments and bucket metrics.

We then report:
  - mean and across-seed 95% percentile interval for every metric
  - per-bucket metric × stage × seed table
  - "robust degradation" — Δ that holds across seeds

This separates *training variance* (how much your numbers move with a
different random training draw) from *test variance* (captured by the
within-seed bootstrap on the fixed n=100 test). Both should be reported.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.detectors import build_char_tfidf_lr, build_word_tfidf_lr
from lib.hardness import (
    all_text_based_hardness,
    hardness_cross_detector,
    hardness_from_probs,
)
from lib.io import label_to_int, load_jsonl, load_test_ids, to_xy
from lib.metrics import (
    METRIC_FUNCS,
    area_under_robustness_curve,
    expected_calibration_error,
    point_metrics,
    relative_degradation_slope,
)


# ── config ────────────────────────────────────────────────────────────────────
N_SEEDS = 20
TRAIN_VAL_FRAC = 0.9      # 90% non-test pool -> train+val
VAL_FRAC_WITHIN = 0.111   # ~10% of total goes to val (i.e. 10/90 of remainder)
METRICS = ["acc", "f1", "auroc", "mcc", "brier", "ece"]
HARDNESS_METHODS = [
    "margin_self_word", "margin_self_char", "margin_cross",
    "readability_fk", "length", "ttr",
]
OUT_DIR = paths.RESULTS / "eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all() -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    p0_all = load_jsonl(paths.P0_PATH)
    test_ids = load_test_ids(paths.TEST_IDS)
    test_rows = [r for r in p0_all if r["id"] in test_ids]
    non_test_rows = [r for r in p0_all if r["id"] not in test_ids]
    eval_splits = {
        "P0_test":            test_rows,
        "P1_test_standard":   load_jsonl(paths.PARAPHRASE_PATHS["P1_test_standard"]),
        "P2_test_standard":   load_jsonl(paths.PARAPHRASE_PATHS["P2_test_standard"]),
        "P1_test_simplified": load_jsonl(paths.PARAPHRASE_PATHS["P1_test_simplified"]),
        "P2_test_simplified": load_jsonl(paths.PARAPHRASE_PATHS["P2_test_simplified"]),
    }
    return non_test_rows, test_rows, eval_splits


def stratified_split(rows: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """Stratified train/val split of `rows` by label."""
    y = np.array([label_to_int(r["label"]) for r in rows], dtype=int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRAC_WITHIN, random_state=seed)
    tr_idx, va_idx = next(sss.split(np.zeros_like(y), y))
    return [rows[i] for i in tr_idx], [rows[i] for i in va_idx]


def eval_detector_one_seed(
    detector,
    eval_splits: dict[str, list[dict]],
    text_hardness_assignments: dict[str, dict[str, str]],
    other_p0_probs: np.ndarray,
) -> tuple[dict, dict, dict]:
    """
    Returns:
      flat_metrics       : dict (split, metric) -> value
      bucket_metrics     : dict (method, split, bucket, metric) -> value (+ n)
      p0_probs           : ndarray of P0 probabilities (used by cross-detector hardness)
    """
    # Compute predictions on each split
    pred_data = {}
    for sp, rows in eval_splits.items():
        X, y = to_xy(rows)
        prob = detector.predict_proba(X)
        pred = (prob >= 0.5).astype(int)
        pred_data[sp] = {
            "y_true": y, "y_pred": pred, "y_prob": prob,
            "ids": [r["id"] for r in rows],
        }

    # Flat metrics on every split
    flat = {}
    for sp, d in pred_data.items():
        pts = point_metrics(d["y_true"], d["y_pred"], d["y_prob"])
        for m, v in pts.items():
            flat[(sp, m)] = v

    # Build hardness assignments
    p0 = pred_data["P0_test"]
    p0_probs = p0["y_prob"]
    p0_ids = p0["ids"]
    assignments = dict(text_hardness_assignments)  # text-based already passed in
    # Self margin
    h_self = hardness_from_probs(p0_probs, "margin")
    assignments[f"margin_self_{detector.name.split('_')[0]}"] = dict(zip(p0_ids, h_self.buckets))
    # Cross margin: use the OTHER detector's probs
    h_cross = hardness_cross_detector(other_p0_probs, "margin")
    assignments["margin_cross"] = dict(zip(p0_ids, h_cross.buckets))

    bucket_metrics = {}
    for method, id_to_bucket in assignments.items():
        for sp, d in pred_data.items():
            ids = d["ids"]
            buckets = np.array([id_to_bucket.get(i, "Unknown") for i in ids])
            for b in ("Easy", "Medium", "Hard"):
                mask = buckets == b
                if mask.sum() == 0:
                    continue
                yt = d["y_true"][mask]
                yp = d["y_pred"][mask]
                ypr = d["y_prob"][mask]
                pts = point_metrics(yt, yp, ypr)
                for m, v in pts.items():
                    bucket_metrics[(method, sp, b, m)] = v
                bucket_metrics[(method, sp, b, "n")] = int(mask.sum())

    return flat, bucket_metrics, p0_probs


def main():
    print(f"Multi-seed experiment (n_seeds={N_SEEDS})\n")
    non_test_rows, test_rows, eval_splits = load_all()

    # Class composition log
    def lc(rows): return (
        sum(1 for r in rows if r["label"] == "human"),
        sum(1 for r in rows if r["label"] == "llm"),
    )
    print(f"Non-test pool: n={len(non_test_rows)} (human={lc(non_test_rows)[0]}, "
          f"llm={lc(non_test_rows)[1]})")
    print(f"Test (fixed):  n={len(test_rows)} (human={lc(test_rows)[0]}, "
          f"llm={lc(test_rows)[1]})\n")

    # Pre-compute text-based hardness once (it depends only on P0 texts)
    p0_test_texts = [r["text"] for r in test_rows]
    p0_test_ids = [r["id"] for r in test_rows]
    text_hardness = all_text_based_hardness(p0_test_texts)
    text_hardness_assignments = {
        name: dict(zip(p0_test_ids, h.buckets)) for name, h in text_hardness.items()
    }

    # ── per-seed collection ────────────────────────────────────────────────────
    flat_records = []           # one row per (seed, detector, split, metric)
    bucket_records = []         # one row per (seed, detector, method, split, bucket, metric)
    aurc_records = []           # one row per (seed, detector, track, metric)

    for seed_idx, seed in enumerate(range(N_SEEDS)):
        train_rows, val_rows = stratified_split(non_test_rows, seed)
        X_tr, y_tr = to_xy(train_rows)
        X_va, y_va = to_xy(val_rows)

        word_det = build_word_tfidf_lr(X_tr, y_tr, seed=seed)
        char_det = build_char_tfidf_lr(X_tr, y_tr, seed=seed)

        # First pass: get P0 probs for each detector (needed for cross-margin)
        word_p0_probs = word_det.predict_proba(p0_test_texts)
        char_p0_probs = char_det.predict_proba(p0_test_texts)

        # Evaluate word with char as cross-detector
        word_flat, word_buckets, _ = eval_detector_one_seed(
            word_det, eval_splits, text_hardness_assignments, other_p0_probs=char_p0_probs,
        )
        char_flat, char_buckets, _ = eval_detector_one_seed(
            char_det, eval_splits, text_hardness_assignments, other_p0_probs=word_p0_probs,
        )

        # Record flat metrics
        for det_name, flat in [("word_tfidf_lr", word_flat), ("char_tfidf_lr", char_flat)]:
            for (sp, m), v in flat.items():
                flat_records.append({
                    "seed": seed, "detector": det_name, "split": sp,
                    "metric": m, "value": v,
                })

        # Record bucket metrics
        for det_name, bm in [("word_tfidf_lr", word_buckets), ("char_tfidf_lr", char_buckets)]:
            for (method, sp, b, m), v in bm.items():
                bucket_records.append({
                    "seed": seed, "detector": det_name,
                    "hardness": method, "split": sp, "bucket": b,
                    "metric": m, "value": v,
                })

        # AURC per detector × track × metric
        TRACKS = {
            "standard":   ["P0_test", "P1_test_standard", "P2_test_standard"],
            "simplified": ["P0_test", "P1_test_simplified", "P2_test_simplified"],
        }
        for det_name, flat in [("word_tfidf_lr", word_flat), ("char_tfidf_lr", char_flat)]:
            for track, stages in TRACKS.items():
                for m in METRICS:
                    vals = [flat[(s, m)] for s in stages]
                    aurc = area_under_robustness_curve(vals)
                    slope = relative_degradation_slope(vals)
                    aurc_records.append({
                        "seed": seed, "detector": det_name, "track": track,
                        "metric": m, "aurc": aurc, "slope": slope,
                        "stage0": vals[0], "stage1": vals[1], "stage2": vals[2],
                        "drop_p0_p2": vals[0] - vals[-1],
                    })

        print(f"  seed {seed:02d}: train={len(train_rows):3d} val={len(val_rows):3d}  "
              f"word_P0_acc={word_flat[('P0_test', 'acc')]:.3f}  "
              f"char_P0_acc={char_flat[('P0_test', 'acc')]:.3f}")

    # ── Save raw per-seed tables ──────────────────────────────────────────────
    df_flat = pd.DataFrame(flat_records)
    df_buckets = pd.DataFrame(bucket_records)
    df_aurc = pd.DataFrame(aurc_records)
    df_flat.to_csv(OUT_DIR / "multiseed_raw_flat.csv", index=False)
    df_buckets.to_csv(OUT_DIR / "multiseed_raw_buckets.csv", index=False)
    df_aurc.to_csv(OUT_DIR / "multiseed_raw_aurc.csv", index=False)

    # ── Aggregate across seeds ────────────────────────────────────────────────
    def summarize(df, group_cols, val_col="value"):
        out = (
            df.groupby(group_cols)[val_col]
              .agg(["mean", "std",
                    lambda x: np.nanpercentile(x, 2.5),
                    lambda x: np.nanpercentile(x, 97.5),
                    "count"])
              .reset_index()
        )
        out.columns = list(group_cols) + ["mean", "std", "ci_lo", "ci_hi", "n_seeds"]
        return out

    summary_flat = summarize(df_flat, ["detector", "split", "metric"])
    summary_buckets = summarize(
        df_buckets[df_buckets["metric"] != "n"],
        ["detector", "hardness", "split", "bucket", "metric"],
    )
    summary_aurc = summarize(df_aurc, ["detector", "track", "metric"], val_col="aurc")

    summary_flat.to_csv(OUT_DIR / "multiseed_summary_flat.csv", index=False)
    summary_buckets.to_csv(OUT_DIR / "multiseed_summary_buckets.csv", index=False)
    summary_aurc.to_csv(OUT_DIR / "multiseed_summary_aurc.csv", index=False)

    # ── Headline report ───────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"MULTI-SEED SUMMARY (n_seeds={N_SEEDS})")
    print("=" * 80)

    print("\nAcross-seed mean (95% percentile interval) — word_tfidf_lr:")
    for sp in ["P0_test", "P1_test_standard", "P2_test_standard",
               "P1_test_simplified", "P2_test_simplified"]:
        r_acc = summary_flat[(summary_flat["detector"] == "word_tfidf_lr") &
                             (summary_flat["split"] == sp) &
                             (summary_flat["metric"] == "acc")].iloc[0]
        r_f1 = summary_flat[(summary_flat["detector"] == "word_tfidf_lr") &
                            (summary_flat["split"] == sp) &
                            (summary_flat["metric"] == "f1")].iloc[0]
        print(f"  {sp:<25}  Acc={r_acc['mean']:.4f} "
              f"[{r_acc['ci_lo']:.4f}, {r_acc['ci_hi']:.4f}]  "
              f"F1={r_f1['mean']:.4f} [{r_f1['ci_lo']:.4f}, {r_f1['ci_hi']:.4f}]")

    print("\nAcross-seed AURC (acc):")
    for det in ["word_tfidf_lr", "char_tfidf_lr"]:
        for track in ["standard", "simplified"]:
            r = summary_aurc[(summary_aurc["detector"] == det) &
                             (summary_aurc["track"] == track) &
                             (summary_aurc["metric"] == "acc")].iloc[0]
            print(f"  {det:<15} {track:<11}  AURC={r['mean']:.4f} "
                  f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]")

    print("\nAcross-seed Hard-bucket F1 (word_tfidf_lr):")
    print(f"  {'hardness':<20} {'P0':<20} {'P2_sim':<20} {'mean drop':<10}")
    for h in HARDNESS_METHODS:
        r_p0 = summary_buckets[(summary_buckets["detector"] == "word_tfidf_lr") &
                               (summary_buckets["hardness"] == h) &
                               (summary_buckets["split"] == "P0_test") &
                               (summary_buckets["bucket"] == "Hard") &
                               (summary_buckets["metric"] == "f1")]
        r_sim = summary_buckets[(summary_buckets["detector"] == "word_tfidf_lr") &
                                (summary_buckets["hardness"] == h) &
                                (summary_buckets["split"] == "P2_test_simplified") &
                                (summary_buckets["bucket"] == "Hard") &
                                (summary_buckets["metric"] == "f1")]
        if len(r_p0) and len(r_sim):
            p0 = r_p0.iloc[0]
            sim = r_sim.iloc[0]
            print(f"  {h:<20} {p0['mean']:.3f} [{p0['ci_lo']:.2f},{p0['ci_hi']:.2f}]  "
                  f"{sim['mean']:.3f} [{sim['ci_lo']:.2f},{sim['ci_hi']:.2f}]  "
                  f"{p0['mean'] - sim['mean']:+.3f}")

    print(f"\nSaved: {OUT_DIR / 'multiseed_summary_flat.csv'}")
    print(f"Saved: {OUT_DIR / 'multiseed_summary_buckets.csv'}")
    print(f"Saved: {OUT_DIR / 'multiseed_summary_aurc.csv'}")


if __name__ == "__main__":
    main()
