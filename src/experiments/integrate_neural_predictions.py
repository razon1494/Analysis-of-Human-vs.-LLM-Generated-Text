"""
integrate_neural_predictions.py
-------------------------------
Takes the RoBERTa predictions CSV produced by the Colab notebook and folds
them into the existing multi-seed / paired-bootstrap / hardness analysis
framework, producing comparable outputs to the word- and char-TFIDF detectors.

Input
-----
    results/eval/roberta_predictions.csv
        Columns: seed, detector, split, id, y_true, y_prob, y_pred

Outputs
-------
    results/eval/roberta_summary_flat.csv       — flat metrics ± across-seed CI
    results/eval/roberta_summary_buckets.csv    — per-bucket metrics ± CI
    results/eval/roberta_summary_aurc.csv       — AURC ± CI
    results/eval/combined_detector_summary.csv  — all 3 detectors side-by-side
    figures/fig12_three_detector_robustness.png — overlay plot
    figures/fig13_universal_hardness_collapse.png — Hard-bucket collapse on all detectors
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.hardness import all_text_based_hardness, hardness_from_probs
from lib.io import load_jsonl, load_test_ids
from lib.metrics import (
    METRIC_FUNCS,
    area_under_robustness_curve,
    expected_calibration_error,
    point_metrics,
    relative_degradation_slope,
)


mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200, "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})

PREDS_CSV = paths.RESULTS / "eval" / "roberta_predictions.csv"
OUT = paths.RESULTS / "eval"
METRICS = ["acc", "f1", "auroc", "mcc", "brier", "ece"]
SPLITS = ["P0_test", "P1_test_standard", "P2_test_standard",
          "P1_test_simplified", "P2_test_simplified"]
TRACKS = {
    "standard":   ["P0_test", "P1_test_standard", "P2_test_standard"],
    "simplified": ["P0_test", "P1_test_simplified", "P2_test_simplified"],
}
DETECTOR_COLOR = {
    "word_tfidf_lr": "#1f77b4",
    "char_tfidf_lr": "#d62728",
    "roberta_base":  "#2ca02c",
}


def per_seed_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(df["seed"].unique()):
        for sp in SPLITS:
            sub = df[(df["seed"] == seed) & (df["split"] == sp)]
            if len(sub) == 0:
                continue
            yt = sub["y_true"].values.astype(int)
            yp = sub["y_pred"].values.astype(int)
            ypr = sub["y_prob"].values.astype(float)
            pts = point_metrics(yt, yp, ypr)
            for m, v in pts.items():
                rows.append({"seed": seed, "split": sp, "metric": m, "value": v})
    return pd.DataFrame(rows)


def per_seed_aurc(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in sorted(df["seed"].unique()):
        for track, stages in TRACKS.items():
            for m in METRICS:
                vals = []
                for sp in stages:
                    sub = df[(df["seed"] == seed) & (df["split"] == sp)]
                    if len(sub) == 0:
                        vals.append(float("nan"))
                        continue
                    yt = sub["y_true"].values.astype(int)
                    yp = sub["y_pred"].values.astype(int)
                    ypr = sub["y_prob"].values.astype(float)
                    vals.append(METRIC_FUNCS[m](yt, yp, ypr))
                rows.append({
                    "seed": seed, "track": track, "metric": m,
                    "stage0": vals[0], "stage1": vals[1], "stage2": vals[2],
                    "aurc":  area_under_robustness_curve(vals),
                    "slope": relative_degradation_slope(vals),
                    "drop_p0_p2": vals[0] - vals[-1],
                })
    return pd.DataFrame(rows)


def per_seed_buckets(df: pd.DataFrame,
                     other_detector_probs_by_seed: dict) -> pd.DataFrame:
    """
    For each seed, compute hardness buckets using:
      - margin_self_roberta : RoBERTa P0 margin (CIRCULAR for RoBERTa)
      - margin_cross_word   : word TFIDF P0 margin (non-circular)
      - margin_cross_char   : char TFIDF P0 margin (non-circular)
      - readability_fk      : Flesch-Kincaid
      - length              : word count
      - ttr                 : type-token ratio

    Then evaluate per-bucket F1 etc on the RoBERTa predictions.
    """
    p0_all = load_jsonl(paths.P0_PATH)
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_test_rows = [r for r in p0_all if r["id"] in test_ids]
    p0_text_by_id = {r["id"]: r["text"] for r in p0_test_rows}

    rows = []
    for seed in sorted(df["seed"].unique()):
        # RoBERTa P0 probs for this seed
        p0_sub = df[(df["seed"] == seed) & (df["split"] == "P0_test")]
        p0_ids = list(p0_sub["id"].values)
        p0_probs_roberta = p0_sub["y_prob"].values

        # Cross-detector probs from the multiseed run: word/char models for the
        # corresponding seed. If unavailable, fall back to single-seed values
        # from the run_evaluation.py predictions.
        word_probs = other_detector_probs_by_seed.get("word", None)
        char_probs = other_detector_probs_by_seed.get("char", None)

        # Build hardness assignments
        assignments = {}
        h_self = hardness_from_probs(p0_probs_roberta, "margin")
        assignments["margin_self_roberta"] = dict(zip(p0_ids, h_self.buckets))
        if word_probs is not None:
            h_w = hardness_from_probs(word_probs, "margin")
            assignments["margin_cross_word"] = dict(zip(p0_ids, h_w.buckets))
        if char_probs is not None:
            h_c = hardness_from_probs(char_probs, "margin")
            assignments["margin_cross_char"] = dict(zip(p0_ids, h_c.buckets))

        p0_texts = [p0_text_by_id[i] for i in p0_ids]
        text_h = all_text_based_hardness(p0_texts)
        for name, h in text_h.items():
            assignments[name] = dict(zip(p0_ids, h.buckets))

        # Per-bucket metrics
        for method, id_to_bucket in assignments.items():
            for sp in SPLITS:
                sub = df[(df["seed"] == seed) & (df["split"] == sp)]
                if len(sub) == 0:
                    continue
                ids = sub["id"].values
                buckets = np.array([id_to_bucket.get(i, "Unknown") for i in ids])
                for b in ("Easy", "Medium", "Hard"):
                    mask = buckets == b
                    if mask.sum() == 0:
                        continue
                    yt = sub["y_true"].values[mask].astype(int)
                    yp = sub["y_pred"].values[mask].astype(int)
                    ypr = sub["y_prob"].values[mask].astype(float)
                    pts = point_metrics(yt, yp, ypr)
                    for m, v in pts.items():
                        rows.append({
                            "seed": seed, "hardness": method, "split": sp,
                            "bucket": b, "metric": m, "value": v,
                            "n": int(mask.sum()),
                        })
    return pd.DataFrame(rows)


def summarize_across_seeds(df: pd.DataFrame, group_cols: list[str],
                           value_col: str = "value") -> pd.DataFrame:
    out = (
        df.groupby(group_cols)[value_col]
          .agg(["mean", "std",
                lambda x: float(np.nanpercentile(x, 2.5)),
                lambda x: float(np.nanpercentile(x, 97.5)),
                "count"])
          .reset_index()
    )
    out.columns = list(group_cols) + ["mean", "std", "ci_lo", "ci_hi", "n_seeds"]
    return out


def fig_three_detector_comparison(df_flat_roberta: pd.DataFrame):
    """
    Overlay all 3 detectors (word, char, RoBERTa) on the same axes for accuracy
    and F1 across the 5 paraphrase stages. Both standard and simplified tracks.
    """
    word_char_df = pd.read_csv(OUT / "multiseed_summary_flat.csv")
    word_char_df["detector_label"] = word_char_df["detector"]
    roberta_summary = summarize_across_seeds(df_flat_roberta, ["split", "metric"])
    roberta_summary["detector_label"] = "roberta_base"
    # Combine
    combined = pd.concat([
        word_char_df[["detector_label", "split", "metric", "mean", "ci_lo", "ci_hi"]],
        roberta_summary.rename(columns={})[["detector_label", "split", "metric", "mean", "ci_lo", "ci_hi"]],
    ], ignore_index=True)
    combined.to_csv(OUT / "combined_detector_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharey=True)
    metrics = ["acc", "f1"]
    tracks = [("standard",   ["P0_test", "P1_test_standard",   "P2_test_standard"]),
              ("simplified", ["P0_test", "P1_test_simplified", "P2_test_simplified"])]
    for ri, (track, stages) in enumerate(tracks):
        for ci, m in enumerate(metrics):
            ax = axes[ri, ci]
            for det, color in DETECTOR_COLOR.items():
                vals, lo, hi = [], [], []
                for sp in stages:
                    r = combined[(combined["detector_label"] == det) &
                                 (combined["split"] == sp) &
                                 (combined["metric"] == m)]
                    if len(r) == 0:
                        vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    else:
                        vals.append(r.iloc[0]["mean"])
                        lo.append(r.iloc[0]["ci_lo"])
                        hi.append(r.iloc[0]["ci_hi"])
                x = np.arange(len(stages))
                ax.plot(x, vals, marker="o", lw=2, color=color, label=det)
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)
            ax.set_xticks(x); ax.set_xticklabels(["P0", "P1", "P2"])
            ax.set_title(f"{m.upper()} — {track}")
            ax.set_ylim(0.4, 1.02)
            if ci == 0:
                ax.set_ylabel(f"{track}\n{m}")
            if ri == 0 and ci == 1:
                ax.legend(loc="lower left")
    fig.suptitle("Three-detector robustness comparison (multi-seed; 95% CI shown)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig12_three_detector_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_universal_hardness_collapse(df_buckets_roberta: pd.DataFrame):
    """
    Hard-bucket F1 across paraphrase stages for ALL THREE detectors and the
    key NON-CIRCULAR hardness definitions. Tests the universality claim.
    """
    # RoBERTa bucket summary
    rob_sum = summarize_across_seeds(
        df_buckets_roberta[df_buckets_roberta["metric"] == "f1"],
        ["hardness", "split", "bucket"],
    )
    # word/char bucket summary
    wc = pd.read_csv(OUT / "multiseed_summary_buckets.csv")
    wc_f1 = wc[wc["metric"] == "f1"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    stages = ["P0_test", "P1_test_standard", "P2_test_standard",
              "P1_test_simplified", "P2_test_simplified"]
    stage_lbl = ["P0", "P1\nstd", "P2\nstd", "P1\nsim", "P2\nsim"]

    # Use the most defensible non-circular hardness for each detector:
    # readability_fk is universal — same buckets for all detectors. Use it.
    hardness_to_use = "readability_fk"
    for ax_i, det in enumerate(["word_tfidf_lr", "char_tfidf_lr", "roberta_base"]):
        ax = axes[ax_i]
        for bucket, color in zip(["Easy", "Medium", "Hard"],
                                 ["#2ca02c", "#ff7f0e", "#d62728"]):
            vals, lo, hi = [], [], []
            for sp in stages:
                if det == "roberta_base":
                    sub = rob_sum[(rob_sum["hardness"] == hardness_to_use) &
                                  (rob_sum["split"] == sp) &
                                  (rob_sum["bucket"] == bucket)]
                else:
                    sub = wc_f1[(wc_f1["detector"] == det) &
                                (wc_f1["hardness"] == hardness_to_use) &
                                (wc_f1["split"] == sp) &
                                (wc_f1["bucket"] == bucket)]
                if len(sub) == 0:
                    vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                else:
                    vals.append(sub.iloc[0]["mean"])
                    lo.append(sub.iloc[0]["ci_lo"])
                    hi.append(sub.iloc[0]["ci_hi"])
            x = np.arange(len(stages))
            ax.plot(x, vals, marker="o", lw=2, color=color, label=bucket)
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)
        ax.set_xticks(x); ax.set_xticklabels(stage_lbl, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(det)
        if ax_i == 0:
            ax.set_ylabel(f"Hard-bucket F1\n(hardness = {hardness_to_use})")
            ax.legend(loc="lower left")
    fig.suptitle(f"Universal hardness collapse — non-circular hardness ({hardness_to_use})\n"
                 "Same Hard-bucket samples on all three detectors → tests cross-family universality.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig13_universal_hardness_collapse.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    if not PREDS_CSV.exists():
        print(f"ERROR: {PREDS_CSV} not found.")
        print("Run the Colab notebook (colab/roberta_detector_finetune.ipynb)")
        print("first and copy roberta_predictions.csv into results/eval/.")
        sys.exit(1)

    df = pd.read_csv(PREDS_CSV)
    print(f"Loaded {len(df)} predictions across "
          f"{df['seed'].nunique()} seeds and {df['split'].nunique()} splits.\n")

    # ── flat metrics per seed + summary ───────────────────────────────────────
    print("[1/3] Computing flat metrics per seed...")
    df_flat = per_seed_metrics(df)
    df_flat.to_csv(OUT / "roberta_raw_flat.csv", index=False)
    summary_flat = summarize_across_seeds(df_flat, ["split", "metric"])
    summary_flat.to_csv(OUT / "roberta_summary_flat.csv", index=False)

    # ── AURC per seed + summary ───────────────────────────────────────────────
    print("[2/3] Computing AURC per seed...")
    df_aurc = per_seed_aurc(df)
    df_aurc.to_csv(OUT / "roberta_raw_aurc.csv", index=False)
    summary_aurc = summarize_across_seeds(df_aurc, ["track", "metric"], "aurc")
    summary_aurc.to_csv(OUT / "roberta_summary_aurc.csv", index=False)

    # ── per-bucket per seed + summary ─────────────────────────────────────────
    # For cross-detector hardness, we'd need word/char P0 probs for the SAME
    # train/val splits the RoBERTa was trained on. Since the Colab uses a
    # different (fixed) train/val split than our 20-seed runs, we use the
    # single-seed predictions CSV from run_evaluation.py as cross-margin
    # reference.
    print("[3/3] Computing per-bucket metrics under multiple hardness definitions...")
    other_probs_by_seed = {}
    single_preds_path = OUT / "predictions.csv"
    if single_preds_path.exists():
        single_preds = pd.read_csv(single_preds_path)
        word_p0 = single_preds[(single_preds.detector == "word_tfidf_lr") &
                               (single_preds.split == "P0_test")]
        char_p0 = single_preds[(single_preds.detector == "char_tfidf_lr") &
                               (single_preds.split == "P0_test")]
        # Align to the order of RoBERTa P0 IDs
        rob_p0_ids = df[(df["seed"] == df["seed"].iloc[0]) & (df["split"] == "P0_test")]["id"].values
        word_map = dict(zip(word_p0["id"], word_p0["y_prob"]))
        char_map = dict(zip(char_p0["id"], char_p0["y_prob"]))
        other_probs_by_seed["word"] = np.array([word_map.get(i, np.nan) for i in rob_p0_ids])
        other_probs_by_seed["char"] = np.array([char_map.get(i, np.nan) for i in rob_p0_ids])

    df_buckets = per_seed_buckets(df, other_probs_by_seed)
    df_buckets.to_csv(OUT / "roberta_raw_buckets.csv", index=False)
    summary_buckets = summarize_across_seeds(
        df_buckets, ["hardness", "split", "bucket", "metric"],
    )
    summary_buckets.to_csv(OUT / "roberta_summary_buckets.csv", index=False)

    # ── Headline print ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RoBERTa-base across seeds")
    print("=" * 70)
    for sp in SPLITS:
        r_acc = summary_flat[(summary_flat["split"] == sp) & (summary_flat["metric"] == "acc")]
        r_f1 = summary_flat[(summary_flat["split"] == sp) & (summary_flat["metric"] == "f1")]
        if len(r_acc) and len(r_f1):
            ra = r_acc.iloc[0]; rf = r_f1.iloc[0]
            print(f"  {sp:<25}  Acc={ra['mean']:.4f} [{ra['ci_lo']:.4f},{ra['ci_hi']:.4f}]   "
                  f"F1={rf['mean']:.4f} [{rf['ci_lo']:.4f},{rf['ci_hi']:.4f}]")
    print("\nAURC (across seeds):")
    for track in ["standard", "simplified"]:
        for m in ["acc", "f1", "auroc"]:
            r = summary_aurc[(summary_aurc["track"] == track) & (summary_aurc["metric"] == m)]
            if len(r):
                print(f"  {track:<11} {m:<6}: {r.iloc[0]['mean']:.4f} "
                      f"[{r.iloc[0]['ci_lo']:.4f}, {r.iloc[0]['ci_hi']:.4f}]")

    print("\nHard-bucket F1 under non-circular hardness (readability_fk):")
    for sp in SPLITS:
        r = summary_buckets[(summary_buckets["hardness"] == "readability_fk") &
                            (summary_buckets["split"] == sp) &
                            (summary_buckets["bucket"] == "Hard") &
                            (summary_buckets["metric"] == "f1")]
        if len(r):
            r0 = r.iloc[0]
            print(f"  {sp:<25}  F1={r0['mean']:.4f} [{r0['ci_lo']:.4f}, {r0['ci_hi']:.4f}]")

    # ── Generate comparison figures ───────────────────────────────────────────
    print("\nGenerating combined figures...")
    fig_three_detector_comparison(df_flat)
    fig_universal_hardness_collapse(df_buckets)

    print("\nDone. Combined comparison ready at:")
    print(f"  {OUT / 'combined_detector_summary.csv'}")
    print(f"  {paths.FIGURES / 'fig12_three_detector_robustness.png'}")
    print(f"  {paths.FIGURES / 'fig13_universal_hardness_collapse.png'}")


if __name__ == "__main__":
    main()
