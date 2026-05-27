"""
evaluate_auroc.py
-----------------
Computes AUROC for every evaluation condition and plots overlaid
ROC curves for both detectors (word n-gram and char n-gram) across
all paraphrase stages and tracks.

Outputs
-------
    results/auroc_summary.csv          -- AUROC per condition per detector
    figures/roc_curves_standard.png    -- ROC curves, standard track
    figures/roc_curves_simplified.png  -- ROC curves, simplified track
    figures/roc_curves_both_f1.png     -- side-by-side AUROC bar chart

Usage
-----
    python src/evaluate_auroc.py

Requirements
------------
    pip install scikit-learn joblib numpy matplotlib pandas
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = "results"
FIGURES_DIR   = "figures"
P0_PATH       = os.path.join("data", "p0", "p0.jsonl")
TEST_IDS_PATH = os.path.join("data", "splits", "test_ids.txt")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_xy(rows):
    X = [r["text"] for r in rows]
    y = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)
    return X, y


def get_probs(rows, vectorizer, clf):
    X, y = to_xy(rows)
    Xv   = vectorizer.transform(X)
    prob = clf.predict_proba(Xv)[:, 1]
    return y, prob


# ── plotting ──────────────────────────────────────────────────────────────────

COLORS = {
    "P0_test":            "#1f77b4",   # blue
    "P1_test_standard":   "#ff7f0e",   # orange
    "P2_test_standard":   "#d62728",   # red
    "P1_test_simplified": "#9467bd",   # purple
    "P2_test_simplified": "#8c564b",   # brown
}

LABELS = {
    "P0_test":            "P0 (original)",
    "P1_test_standard":   "P1 standard",
    "P2_test_standard":   "P2 standard",
    "P1_test_simplified": "P1 simplified",
    "P2_test_simplified": "P2 simplified",
}


def plot_roc_track(split_names, y_probs_word, y_probs_char, y_trues,
                   title: str, outfile: str):
    """
    Plot ROC curves for a given set of splits, showing both word and
    char n-gram detectors on the same axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, (detector_label, probs_dict) in zip(
        axes,
        [("Word n-gram", y_probs_word), ("Char n-gram", y_probs_char)],
    ):
        for name in split_names:
            y_true = y_trues[name]
            y_prob = probs_dict[name]
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            ax.plot(
                fpr, tpr,
                color=COLORS[name],
                lw=1.8,
                label=f"{LABELS[name]}  AUC={auc:.3f}",
            )

        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"{detector_label}", fontsize=12)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.01])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


def plot_auroc_bar(records: list, outfile: str):
    """
    Side-by-side bar chart: AUROC per condition for both detectors.
    """
    df = pd.DataFrame(records)
    splits = df["split"].unique().tolist()

    x     = np.arange(len(splits))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))

    word_vals = df[df["detector"] == "word"]["auroc"].values
    char_vals = df[df["detector"] == "char"]["auroc"].values

    bars1 = ax.bar(x - width / 2, word_vals, width, label="Word n-gram", color="#4878CF")
    bars2 = ax.bar(x + width / 2, char_vals, width, label="Char n-gram", color="#6ACC65")

    # value labels on bars
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=7.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(s, s) for s in splits], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("AUROC Comparison: Word vs Char N-gram Detector", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    vec_word = joblib.load(os.path.join(RESULTS_DIR, "vectorizer.joblib"))
    clf_word = joblib.load(os.path.join(RESULTS_DIR, "model.joblib"))

    # Char model may not exist yet if train_char_ngram_detector.py hasn't run
    char_available = os.path.exists(os.path.join(RESULTS_DIR, "vectorizer_char.joblib"))
    if char_available:
        vec_char = joblib.load(os.path.join(RESULTS_DIR, "vectorizer_char.joblib"))
        clf_char = joblib.load(os.path.join(RESULTS_DIR, "model_char.joblib"))
        print("Char n-gram model loaded.")
    else:
        print(
            "WARNING: char n-gram model not found. "
            "Run train_char_ngram_detector.py first for full comparison.\n"
            "Proceeding with word n-gram only."
        )

    # ── load all test splits ──────────────────────────────────────────────────
    test_ids = set(Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines())
    p0_rows  = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    splits_def = {
        "P0_test":            p0_rows,
        "P1_test_standard":   load_jsonl(os.path.join("data", "p1", "p1_test.jsonl")),
        "P2_test_standard":   load_jsonl(os.path.join("data", "p2", "p2_test.jsonl")),
        "P1_test_simplified": load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        "P2_test_simplified": load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    }

    # ── compute probabilities ─────────────────────────────────────────────────
    y_trues      = {}
    probs_word   = {}
    probs_char   = {}

    print("\nAUROC Summary\n" + "-" * 55)
    print(f"{'Split':<25} {'Word AUROC':>12}  {'Char AUROC':>12}")
    print("-" * 55)

    records = []

    for name, rows in splits_def.items():
        y_true, prob_w = get_probs(rows, vec_word, clf_word)
        y_trues[name]    = y_true
        probs_word[name] = prob_w
        auc_w = roc_auc_score(y_true, prob_w)

        records.append({"split": name, "detector": "word", "auroc": round(auc_w, 4)})

        if char_available:
            _, prob_c = get_probs(rows, vec_char, clf_char)
            probs_char[name] = prob_c
            auc_c = roc_auc_score(y_true, prob_c)
            records.append({"split": name, "detector": "char", "auroc": round(auc_c, 4)})
            print(f"{name:<25} {auc_w:>12.4f}  {auc_c:>12.4f}")
        else:
            probs_char[name] = prob_w   # fallback: duplicate word probs
            print(f"{name:<25} {auc_w:>12.4f}  {'N/A':>12}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "auroc_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ── ROC curve plots ───────────────────────────────────────────────────────
    # Standard track
    plot_roc_track(
        split_names   = ["P0_test", "P1_test_standard", "P2_test_standard"],
        y_probs_word  = probs_word,
        y_probs_char  = probs_char,
        y_trues       = y_trues,
        title         = "ROC Curves — Standard Paraphrase Track",
        outfile       = os.path.join(FIGURES_DIR, "roc_curves_standard.png"),
    )

    # Simplified track
    plot_roc_track(
        split_names   = ["P0_test", "P1_test_simplified", "P2_test_simplified"],
        y_probs_word  = probs_word,
        y_probs_char  = probs_char,
        y_trues       = y_trues,
        title         = "ROC Curves — Simplified Paraphrase Track",
        outfile       = os.path.join(FIGURES_DIR, "roc_curves_simplified.png"),
    )

    # Bar chart comparison (only if char model available)
    if char_available:
        plot_auroc_bar(
            records = records,
            outfile = os.path.join(FIGURES_DIR, "auroc_bar_comparison.png"),
        )


if __name__ == "__main__":
    main()
