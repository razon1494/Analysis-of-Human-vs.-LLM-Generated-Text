"""
make_figures.py
---------------
Generates publication-quality figures from results/eval/*.csv.

Figures produced (all in figures/):

  fig01_aggregate_robustness.png
      Aggregate F1 and accuracy over paraphrase stages, both detectors,
      both tracks, with bootstrap CIs as shaded bands.

  fig02_hardness_trajectory.png
      THE HEADLINE FIGURE: Hard-bucket F1 trajectory across paraphrase
      stages, for each hardness definition. Shows that the collapse
      persists under NON-CIRCULAR hardness (margin_cross, readability_fk).

  fig03_hardness_concordance_heatmap.png
      Kendall's tau matrix between hardness definitions. Demonstrates
      that detector-margin and text-complexity measure different things.

  fig04_aurc_bars.png
      AURC summary per detector × track × metric — single-scalar
      robustness summary.

  fig05_paired_diff_forest.png
      Forest plot of paired-bootstrap Δ CIs for each (track, comparison,
      detector) combination. Visual significance check.

  fig06_calibration_reliability.png
      Reliability diagrams + ECE per detector per stage. Shows
      calibration drift under paraphrasing.

  fig07_class_imbalance_warning.png
      Explicit visualization of the 38/62 test set imbalance to flag in
      the paper. Honest reporting.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl, load_test_ids, to_xy
from lib.metrics import expected_calibration_error


# ── style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi":     120,
    "savefig.dpi":    200,
    "font.size":      10,
    "font.family":    "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":      True,
    "grid.alpha":     0.25,
    "grid.linewidth": 0.5,
})

EVAL_DIR = paths.RESULTS / "eval"
FIG_DIR = paths.FIGURES
FIG_DIR.mkdir(parents=True, exist_ok=True)

STAGE_LABEL = {
    "P0_test": "P0",
    "P1_test_standard": "P1\n(std)",
    "P2_test_standard": "P2\n(std)",
    "P1_test_simplified": "P1\n(sim)",
    "P2_test_simplified": "P2\n(sim)",
}
DETECTOR_COLOR = {
    "word_tfidf_lr": "#1f77b4",
    "char_tfidf_lr": "#d62728",
}
TRACK_LS = {"standard": "-", "simplified": "--"}
BUCKET_COLOR = {"Easy": "#2ca02c", "Medium": "#ff7f0e", "Hard": "#d62728"}


def fig01_aggregate_robustness():
    df = pd.read_csv(EVAL_DIR / "metrics_point_and_ci.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)

    for col, metric in enumerate(["acc", "f1"]):
        for row, track in enumerate(["standard", "simplified"]):
            ax = axes[row, col]
            stages = ["P0_test", f"P1_test_{track}", f"P2_test_{track}"]
            x_labels = ["P0", "P1", "P2"]
            for det, color in DETECTOR_COLOR.items():
                sub = df[(df["detector"] == det) & (df["metric"] == metric)]
                vals = [sub[sub["split"] == s].iloc[0] for s in stages]
                points = [v["point"] for v in vals]
                lo = [v["ci_lo"] for v in vals]
                hi = [v["ci_hi"] for v in vals]
                x = np.arange(len(stages))
                ax.plot(x, points, marker="o", color=color,
                        label=det.replace("_tfidf_lr", " TF-IDF"))
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)
            ax.set_xticks(np.arange(len(stages)))
            ax.set_xticklabels(x_labels)
            ax.set_title(f"{metric.upper()} — {track} track")
            ax.set_ylim(0.0, 1.05)
            if col == 0:
                ax.set_ylabel(metric.upper())
            ax.legend(loc="lower left", frameon=True)
    fig.suptitle("Aggregate robustness with 95% bootstrap CIs",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig01_aggregate_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig02_hardness_trajectory():
    df = pd.read_csv(EVAL_DIR / "hardness_buckets_multi.csv")
    methods = ["margin_self_word", "margin_self_char", "margin_cross",
               "readability_fk", "length", "ttr"]
    method_titles = {
        "margin_self_word": "(circular) margin: word detector",
        "margin_self_char": "margin: char detector",
        "margin_cross":     "margin: cross-detector (NON-CIRCULAR)",
        "readability_fk":   "Flesch-Kincaid grade (NON-CIRCULAR)",
        "length":           "word count (NON-CIRCULAR)",
        "ttr":              "type-token ratio (NON-CIRCULAR)",
    }
    stages = ["P0_test", "P1_test_standard", "P2_test_standard",
              "P1_test_simplified", "P2_test_simplified"]
    stage_x = list(range(len(stages)))

    # 2 rows (one per detector) x 6 columns (one per hardness method)
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), sharey=True)

    for row, det in enumerate(["word_tfidf_lr", "char_tfidf_lr"]):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            for bucket, color in BUCKET_COLOR.items():
                vals_pt = []
                vals_lo = []
                vals_hi = []
                ns = []
                for s in stages:
                    sub = df[(df["detector"] == det) &
                             (df["hardness"] == method) &
                             (df["split"] == s) &
                             (df["bucket"] == bucket)]
                    if len(sub) == 0:
                        vals_pt.append(np.nan)
                        vals_lo.append(np.nan)
                        vals_hi.append(np.nan)
                        ns.append(0)
                    else:
                        vals_pt.append(sub.iloc[0]["f1"])
                        vals_lo.append(sub.iloc[0]["f1_ci_lo"])
                        vals_hi.append(sub.iloc[0]["f1_ci_hi"])
                        ns.append(int(sub.iloc[0]["n"]))
                ax.plot(stage_x, vals_pt, marker="o", lw=2, color=color,
                        label=f"{bucket} (n≈{ns[0] if ns else 0})")
                ax.fill_between(stage_x, vals_lo, vals_hi, alpha=0.15, color=color)
            ax.set_xticks(stage_x)
            ax.set_xticklabels(["P0", "P1\nstd", "P2\nstd", "P1\nsim", "P2\nsim"], fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{method_titles[method]}", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(f"{det.replace('_tfidf_lr', '')}\nHard-bucket F1", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8)
            # mark non-circular columns
            if "NON-CIRCULAR" in method_titles[method]:
                ax.set_facecolor("#fff7e6")

    fig.suptitle("F1 by hardness bucket × paraphrase stage × hardness definition\n"
                 "Yellow-shaded panels use hardness definitions that do NOT depend on the evaluated detector.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig02_hardness_trajectory.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig03_hardness_concordance_heatmap():
    taus_all = json.loads((EVAL_DIR / "bucket_overlap_kendall.json").read_text(encoding="utf-8"))
    # Use word detector view (results are slightly different for char)
    taus = taus_all["word_tfidf_lr"]
    methods = ["margin_self_word", "margin_self_char", "margin_cross",
               "readability_fk", "length", "ttr"]
    M = np.full((len(methods), len(methods)), np.nan)
    for i, a in enumerate(methods):
        for j, b in enumerate(methods):
            if i == j:
                M[i, j] = 1.0
            else:
                key1 = f"{a}__vs__{b}"
                key2 = f"{b}__vs__{a}"
                if key1 in taus:
                    M[i, j] = taus[key1]["tau"]
                elif key2 in taus:
                    M[i, j] = taus[key2]["tau"]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = M[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.5 else "black", fontsize=10)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    ax.grid(False)
    plt.colorbar(im, ax=ax, label="Kendall's tau")
    ax.set_title("Hardness concordance (Kendall's tau)\n"
                 "Detector-margin hardness is uncorrelated with text-complexity hardness.",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig03_hardness_concordance_heatmap.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig04_aurc_bars():
    df = pd.read_csv(EVAL_DIR / "aurc_summary.csv")
    metrics = ["acc", "f1", "auroc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m]
        labels = []
        vals = []
        colors = []
        for det in ["word_tfidf_lr", "char_tfidf_lr"]:
            for track in ["standard", "simplified"]:
                r = sub[(sub["detector"] == det) & (sub["track"] == track)]
                if len(r):
                    labels.append(f"{det.replace('_tfidf_lr', '')}\n{track}")
                    vals.append(r.iloc[0]["aurc"])
                    colors.append(DETECTOR_COLOR[det] if track == "standard"
                                  else mpl.colors.to_rgba(DETECTOR_COLOR[det], 0.5))
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"AURC ({m})")
        ax.set_title(f"AURC — {m}")
        ax.axhline(1.0, color="gray", linestyle=":", lw=0.6, alpha=0.5)
    fig.suptitle("Area Under Robustness Curve (1.0 = perfect, 0.0 = total failure)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig04_aurc_bars.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig05_paired_diff_forest():
    df = pd.read_csv(EVAL_DIR / "metrics_paired_diff.csv")
    # Focus on accuracy + F1
    metrics = ["acc", "f1"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 6.5))

    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m].copy()
        sub = sub.sort_values(["track", "detector", "comparison"]).reset_index(drop=True)
        y = np.arange(len(sub))
        for i, r in sub.iterrows():
            color = DETECTOR_COLOR[r["detector"]]
            ax.errorbar(
                r["delta"], i,
                xerr=[[r["delta"] - r["ci_lo"]], [r["ci_hi"] - r["delta"]]],
                fmt="o", color=color, ecolor=color,
                elinewidth=1.5, capsize=4, markersize=5,
            )
            if r["significant"]:
                ax.scatter(r["delta"], i, marker="*", s=80,
                           color=color, edgecolor="black", linewidth=0.6, zorder=5)
        ax.axvline(0, color="gray", linestyle="--", lw=0.8)
        labels = [f"[{r['detector'].replace('_tfidf_lr','')}/{r['track'][:3]}] {r['comparison']}"
                  for _, r in sub.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlabel(f"Δ {m} (B − A)  with 95% paired bootstrap CI")
        ax.set_title(f"Paired-bootstrap CI on {m} degradation")
        ax.invert_yaxis()
        # significance legend marker
        from matplotlib.lines import Line2D
        sig_marker = Line2D([], [], marker="*", color="black", markersize=10,
                            linestyle="", label="significant @ 95%")
        ax.legend(handles=[sig_marker], loc="lower left", fontsize=8)
    fig.suptitle("Forest plot: Δmetric across paraphrase stages",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig05_paired_diff_forest.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig06_calibration_reliability():
    """Reliability diagrams for word detector across all splits."""
    preds = pd.read_csv(EVAL_DIR / "predictions.csv")
    splits = ["P0_test", "P1_test_standard", "P2_test_standard",
              "P1_test_simplified", "P2_test_simplified"]
    detectors = ["word_tfidf_lr", "char_tfidf_lr"]

    fig, axes = plt.subplots(2, 5, figsize=(17, 7), sharex=True, sharey=True)
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    mid = 0.5 * (bins[:-1] + bins[1:])

    for row, det in enumerate(detectors):
        for col, sp in enumerate(splits):
            ax = axes[row, col]
            sub = preds[(preds["detector"] == det) & (preds["split"] == sp)]
            if len(sub) == 0:
                continue
            y_true = sub["y_true"].values
            y_prob = sub["y_prob"].values
            bin_ids = np.digitize(y_prob, bins[1:-1])
            bin_conf, bin_acc, bin_w = [], [], []
            for b in range(n_bins):
                mask = bin_ids == b
                if mask.sum() == 0:
                    bin_conf.append(np.nan)
                    bin_acc.append(np.nan)
                    bin_w.append(0)
                    continue
                bin_conf.append(y_prob[mask].mean())
                bin_acc.append((y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean())
                bin_w.append(mask.sum())
            ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)

            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6)
            ax.plot(bin_conf, bin_acc, marker="o", color=DETECTOR_COLOR[det], lw=1.5)
            # bin weights shown as bar colors with per-bar alpha encoded in RGBA
            max_w = max(1, max(bin_w))
            rgba = mpl.colors.to_rgba(DETECTOR_COLOR[det])
            for c, w in zip(mid, bin_w):
                a = (w / max_w) * 0.7
                ax.bar([c], [0.05], width=0.08,
                       color=(rgba[0], rgba[1], rgba[2], a),
                       edgecolor="none")
            ax.set_title(f"{sp.replace('_test', '')}\nECE = {ece:.3f}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(f"{det.replace('_tfidf_lr', '')}\nObserved P(correct)", fontsize=9)
            if row == 1:
                ax.set_xlabel("Predicted P(LLM)")
    fig.suptitle("Reliability diagrams (10 bins). ECE measures gap between confidence and accuracy.",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG_DIR / "fig06_calibration_reliability.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig07_class_imbalance_warning():
    """Honest reporting figure: test split is not 50/50."""
    p0_all = load_jsonl(paths.P0_PATH)
    test_ids = load_test_ids(paths.TEST_IDS)
    train_ids = load_test_ids(paths.TRAIN_IDS)
    val_ids = load_test_ids(paths.VAL_IDS)

    def counts(rows):
        return (
            sum(1 for r in rows if r["label"] == "human"),
            sum(1 for r in rows if r["label"] == "llm"),
        )
    tr_h, tr_l = counts([r for r in p0_all if r["id"] in train_ids])
    va_h, va_l = counts([r for r in p0_all if r["id"] in val_ids])
    te_h, te_l = counts([r for r in p0_all if r["id"] in test_ids])

    splits = ["Train", "Val", "Test"]
    h_counts = [tr_h, va_h, te_h]
    l_counts = [tr_l, va_l, te_l]

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, h_counts, width=0.38, color="#1f77b4", label="Human")
    ax.bar(x + 0.2, l_counts, width=0.38, color="#d62728", label="LLM")
    for i, (h, l) in enumerate(zip(h_counts, l_counts)):
        total = h + l
        ax.text(i - 0.2, h + 3, str(h), ha="center", fontsize=9)
        ax.text(i + 0.2, l + 3, str(l), ha="center", fontsize=9)
        ax.text(i, max(h, l) + 25, f"{h}h/{l}l  (LLM={l/max(1,total):.0%})",
                ha="center", fontsize=8, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("# samples")
    ax.set_title("Class balance per split\n"
                 "Random 80/10/10 split (no stratification) produced an imbalanced test set\n"
                 "F1 numbers should be interpreted with this in mind.",
                 fontsize=10, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    out = FIG_DIR / "fig07_class_imbalance_warning.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    print("Generating figures...")
    fig01_aggregate_robustness()
    fig02_hardness_trajectory()
    fig03_hardness_concordance_heatmap()
    fig04_aurc_bars()
    fig05_paired_diff_forest()
    fig06_calibration_reliability()
    fig07_class_imbalance_warning()
    print("\nAll figures saved to:", FIG_DIR)


if __name__ == "__main__":
    main()
