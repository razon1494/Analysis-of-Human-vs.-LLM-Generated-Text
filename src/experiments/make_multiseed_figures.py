"""
make_multiseed_figures.py
-------------------------
Visualizations derived from the multi-seed run (run_multiseed.py).

Figures:
  fig08_multiseed_robustness.png
      Aggregate Acc/F1/AUROC across stages, with across-seed shaded
      intervals — much tighter than the single-seed bootstrap, and
      separates training variance from test variance.

  fig09_multiseed_hardness_grid.png
      Hard-bucket F1 trajectories under EVERY hardness definition
      (3 detector-based + 3 text-only), with cross-seed CIs. The
      central figure for defending against circularity.

  fig10_multiseed_aurc_bars.png
      AURC bars with cross-seed CIs.
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
})

EVAL = paths.RESULTS / "eval"
FIG = paths.FIGURES
FIG.mkdir(parents=True, exist_ok=True)

DETECTOR_COLOR = {
    "word_tfidf_lr": "#1f77b4",
    "char_tfidf_lr": "#d62728",
}
BUCKET_COLOR = {"Easy": "#2ca02c", "Medium": "#ff7f0e", "Hard": "#d62728"}


def fig08_multiseed_robustness():
    df = pd.read_csv(EVAL / "multiseed_summary_flat.csv")
    metrics = ["acc", "f1", "auroc"]
    tracks = [("standard",   ["P0_test", "P1_test_standard",   "P2_test_standard"]),
              ("simplified", ["P0_test", "P1_test_simplified", "P2_test_simplified"])]
    fig, axes = plt.subplots(len(tracks), len(metrics), figsize=(13, 6.5), sharey=True)

    for ri, (track_name, stages) in enumerate(tracks):
        for ci, m in enumerate(metrics):
            ax = axes[ri, ci]
            for det, color in DETECTOR_COLOR.items():
                rows = [
                    df[(df["detector"] == det) & (df["split"] == s) & (df["metric"] == m)].iloc[0]
                    for s in stages
                ]
                means = [r["mean"] for r in rows]
                lo = [r["ci_lo"] for r in rows]
                hi = [r["ci_hi"] for r in rows]
                x = np.arange(len(stages))
                ax.plot(x, means, marker="o", color=color, lw=2,
                        label=det.replace("_tfidf_lr", " TF-IDF"))
                ax.fill_between(x, lo, hi, alpha=0.18, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(["P0", "P1", "P2"])
            ax.set_title(f"{m.upper()} — {track_name}")
            ax.set_ylim(0.4, 1.02)
            if ci == 0:
                ax.set_ylabel(f"{track_name}\nmetric value")
            if ri == 0 and ci == 2:
                ax.legend(loc="lower left")
    fig.suptitle(
        "Multi-seed robustness (20 seeds; across-seed 95% percentile interval shown)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG / "fig08_multiseed_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig09_multiseed_hardness_grid():
    df = pd.read_csv(EVAL / "multiseed_summary_buckets.csv")
    df = df[df["metric"] == "f1"]
    methods = ["margin_self_word", "margin_self_char", "margin_cross",
               "readability_fk", "length", "ttr"]
    method_label = {
        "margin_self_word": "(CIRCULAR)\nmargin: word",
        "margin_self_char": "margin: char",
        "margin_cross":     "NON-CIRCULAR\nmargin: cross",
        "readability_fk":   "NON-CIRCULAR\nFlesch-Kincaid",
        "length":           "NON-CIRCULAR\nword count",
        "ttr":              "NON-CIRCULAR\nTTR",
    }
    stages = ["P0_test", "P1_test_standard", "P2_test_standard",
              "P1_test_simplified", "P2_test_simplified"]
    stage_xticks = ["P0", "P1\nstd", "P2\nstd", "P1\nsim", "P2\nsim"]

    fig, axes = plt.subplots(2, 6, figsize=(20, 7.5), sharey=True)

    for row, det in enumerate(["word_tfidf_lr", "char_tfidf_lr"]):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            for bucket, color in BUCKET_COLOR.items():
                vals = []
                lo, hi = [], []
                for s in stages:
                    sub = df[(df["detector"] == det) & (df["hardness"] == method) &
                             (df["split"] == s) & (df["bucket"] == bucket)]
                    if len(sub) == 0:
                        vals.append(np.nan); lo.append(np.nan); hi.append(np.nan)
                    else:
                        r = sub.iloc[0]
                        vals.append(r["mean"]); lo.append(r["ci_lo"]); hi.append(r["ci_hi"])
                x = np.arange(len(stages))
                ax.plot(x, vals, marker="o", lw=2, color=color, label=bucket)
                ax.fill_between(x, lo, hi, alpha=0.15, color=color)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_xticks, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(method_label[method], fontsize=9)
            if "NON-CIRCULAR" in method_label[method]:
                ax.set_facecolor("#fff7e6")
            if col == 0:
                ax.set_ylabel(f"{det.replace('_tfidf_lr', '')}\nbucket F1", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(
        "Multi-seed Hard-bucket F1 collapse under SIX hardness definitions "
        "(top = word detector, bottom = char detector)\n"
        "Yellow panels: hardness independent of the evaluated detector — collapse persists.",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out = FIG / "fig09_multiseed_hardness_grid.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig10_multiseed_aurc_bars():
    df = pd.read_csv(EVAL / "multiseed_summary_aurc.csv")
    metrics = ["acc", "f1", "auroc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4))
    for ax, m in zip(axes, metrics):
        sub = df[df["metric"] == m]
        labels, vals, los, his, colors = [], [], [], [], []
        for det in ["word_tfidf_lr", "char_tfidf_lr"]:
            for track in ["standard", "simplified"]:
                r = sub[(sub["detector"] == det) & (sub["track"] == track)]
                if len(r):
                    labels.append(f"{det.replace('_tfidf_lr', '')}\n{track}")
                    vals.append(r.iloc[0]["mean"])
                    los.append(r.iloc[0]["mean"] - r.iloc[0]["ci_lo"])
                    his.append(r.iloc[0]["ci_hi"] - r.iloc[0]["mean"])
                    colors.append(DETECTOR_COLOR[det] if track == "standard"
                                  else mpl.colors.to_rgba(DETECTOR_COLOR[det], 0.55))
        bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6,
                      yerr=[los, his], capsize=3,
                      error_kw=dict(ecolor="black", lw=1))
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.025, f"{v:.3f}",
                    ha="center", fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"AURC ({m})")
        ax.set_title(f"{m.upper()}")
        ax.axhline(1.0, color="gray", linestyle=":", lw=0.6, alpha=0.5)
    fig.suptitle("Multi-seed AURC with 95% across-seed CI", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = FIG / "fig10_multiseed_aurc_bars.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    fig08_multiseed_robustness()
    fig09_multiseed_hardness_grid()
    fig10_multiseed_aurc_bars()


if __name__ == "__main__":
    main()
