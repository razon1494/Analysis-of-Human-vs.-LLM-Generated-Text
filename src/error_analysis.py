"""
error_analysis.py
-----------------
Two analyses in one script:

1. QUALITATIVE ERROR EXAMPLES
   For each paraphrase condition, surfaces the most "interesting"
   misclassified examples: Hard-bucket false positives and false
   negatives with their confidence scores and the original text
   (P0 version) for comparison.

2. CONFUSION MATRIX HEATMAPS
   Plots a 2×3 grid of confusion matrices (one per condition)
   as annotated heatmaps.

3. MCC TREND
   Plots Matthews Correlation Coefficient across all conditions
   for both word and char n-gram detectors.

Outputs
-------
    results/error_examples.json          -- qualitative examples (top-5 per split)
    results/mcc_summary.csv              -- MCC per condition per detector
    figures/confusion_matrices.png       -- 2×3 heatmap grid
    figures/mcc_trend.png                -- MCC trend line plot

Usage
-----
    python src/error_analysis.py

Requirements
------------
    pip install scikit-learn joblib numpy matplotlib seaborn pandas
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)

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


def get_preds(rows, vectorizer, clf):
    X, y = to_xy(rows)
    Xv   = vectorizer.transform(X)
    pred = clf.predict(Xv)
    prob = clf.predict_proba(Xv)[:, 1]
    return y, pred, prob


# ── confusion matrix heatmaps ─────────────────────────────────────────────────

def plot_confusion_matrices(splits_def: dict, vectorizer, clf, outfile: str):
    """
    Plots a grid of confusion matrices for all conditions.
    """
    n_splits = len(splits_def)
    ncols    = 3
    nrows    = (n_splits + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes      = axes.flatten()

    LABELS = ["Human", "LLM"]

    for i, (name, rows) in enumerate(splits_def.items()):
        y, pred, _ = get_preds(rows, vectorizer, clf)
        cm  = confusion_matrix(y, pred)
        acc = np.trace(cm) / np.sum(cm)
        mcc = matthews_corrcoef(y, pred)

        ax = axes[i]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=LABELS,
            yticklabels=LABELS,
            ax=ax,
            cbar=False,
            linewidths=0.5,
        )
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("True",      fontsize=9)
        ax.set_title(
            f"{name}\nacc={acc:.3f}  mcc={mcc:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Confusion Matrices Across Paraphrase Conditions (Word N-gram Detector)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


# ── MCC trend ─────────────────────────────────────────────────────────────────

def plot_mcc_trend(splits_def: dict, vec_word, clf_word,
                   vec_char, clf_char, char_available: bool, outfile: str):
    """
    Line plot of MCC across all conditions for both detectors.
    """
    names, mcc_word, mcc_char = [], [], []

    for name, rows in splits_def.items():
        y, pred_w, _ = get_preds(rows, vec_word, clf_word)
        names.append(name)
        mcc_word.append(matthews_corrcoef(y, pred_w))
        if char_available:
            _, pred_c, _ = get_preds(rows, vec_char, clf_char)
            mcc_char.append(matthews_corrcoef(y, pred_c))

    short_names = [
        n.replace("_test", "").replace("_standard", "\n(std)")
         .replace("_simplified", "\n(sim)")
        for n in names
    ]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(short_names, mcc_word, marker="o", lw=2,
            color="#1f77b4", label="Word n-gram")
    if char_available:
        ax.plot(short_names, mcc_char, marker="s", lw=2,
                linestyle="--", color="#ff7f0e", label="Char n-gram")

    ax.axhline(0, color="gray", linestyle=":", lw=0.8, alpha=0.6)
    ax.set_ylabel("Matthews Correlation Coefficient (MCC)", fontsize=11)
    ax.set_xlabel("Condition", fontsize=11)
    ax.set_title(
        "MCC Under Iterative Paraphrasing\n(higher = better; 0 = random)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-0.1, 1.05)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")

    return names, mcc_word, mcc_char


# ── qualitative error examples ────────────────────────────────────────────────

def extract_error_examples(
    name: str,
    rows: list,
    p0_rows: list,
    vectorizer, clf,
    n_examples: int = 5,
) -> list:
    """
    For a given split, find the most confident misclassifications
    and return them with their P0 counterpart for comparison.
    """
    X, y = to_xy(rows)
    Xv   = vectorizer.transform(X)
    pred = clf.predict(Xv)
    prob = clf.predict_proba(Xv)[:, 1]

    # Build id → P0 text map
    p0_text_map = {r["id"]: r["text"] for r in p0_rows}

    errors = []
    for i, (true_lbl, pred_lbl, p_llm) in enumerate(zip(y, pred, prob)):
        if true_lbl == pred_lbl:
            continue        # correct prediction, skip

        row        = rows[i]
        true_class = "llm" if true_lbl == 1 else "human"
        pred_class = "llm" if pred_lbl == 1 else "human"
        # Confidence: probability assigned to the predicted (wrong) class
        conf = p_llm if pred_lbl == 1 else (1 - p_llm)

        errors.append({
            "split":       name,
            "id":          row.get("id", ""),
            "true_label":  true_class,
            "pred_label":  pred_class,
            "confidence":  round(float(conf), 4),
            "p_llm":       round(float(p_llm), 4),
            "text":        row["text"],
            "text_p0":     p0_text_map.get(row.get("id", ""), "N/A"),
        })

    # Sort by confidence (most confidently wrong first)
    errors.sort(key=lambda x: x["confidence"], reverse=True)
    return errors[:n_examples]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    vec_word = joblib.load(os.path.join(RESULTS_DIR, "vectorizer.joblib"))
    clf_word = joblib.load(os.path.join(RESULTS_DIR, "model.joblib"))

    char_available = os.path.exists(os.path.join(RESULTS_DIR, "vectorizer_char.joblib"))
    if char_available:
        vec_char = joblib.load(os.path.join(RESULTS_DIR, "vectorizer_char.joblib"))
        clf_char = joblib.load(os.path.join(RESULTS_DIR, "model_char.joblib"))
    else:
        vec_char = vec_word
        clf_char = clf_word

    # ── load splits ───────────────────────────────────────────────────────────
    test_ids = set(Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines())
    p0_rows  = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    splits_def = {
        "P0_test":            p0_rows,
        "P1_test_standard":   load_jsonl(os.path.join("data", "p1", "p1_test.jsonl")),
        "P2_test_standard":   load_jsonl(os.path.join("data", "p2", "p2_test.jsonl")),
        "P1_test_simplified": load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        "P2_test_simplified": load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    }

    # ── 1. Confusion matrix heatmaps ──────────────────────────────────────────
    plot_confusion_matrices(
        splits_def, vec_word, clf_word,
        outfile=os.path.join(FIGURES_DIR, "confusion_matrices.png"),
    )

    # ── 2. MCC trend ──────────────────────────────────────────────────────────
    names, mcc_w, mcc_c = plot_mcc_trend(
        splits_def, vec_word, clf_word, vec_char, clf_char, char_available,
        outfile=os.path.join(FIGURES_DIR, "mcc_trend.png"),
    )

    # Save MCC CSV
    mcc_records = []
    for n, mw in zip(names, mcc_w):
        mcc_records.append({"split": n, "detector": "word", "mcc": round(mw, 4)})
    if char_available:
        for n, mc in zip(names, mcc_c):
            mcc_records.append({"split": n, "detector": "char", "mcc": round(mc, 4)})
    pd.DataFrame(mcc_records).to_csv(
        os.path.join(RESULTS_DIR, "mcc_summary.csv"), index=False
    )
    print(f"Saved: {os.path.join(RESULTS_DIR, 'mcc_summary.csv')}")

    # ── 3. Qualitative error examples ─────────────────────────────────────────
    print("\nExtracting qualitative error examples …")
    all_examples = []

    for name, rows in splits_def.items():
        examples = extract_error_examples(
            name, rows, p0_rows, vec_word, clf_word, n_examples=5
        )
        all_examples.extend(examples)
        print(f"  {name}: {len(examples)} error examples found")

    # Save as JSON for readability
    err_json = os.path.join(RESULTS_DIR, "error_examples.json")
    with open(err_json, "w", encoding="utf-8") as f:
        json.dump(all_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved: {err_json}")

    # Pretty-print top examples for P1_test_simplified (most interesting)
    print("\n" + "=" * 70)
    print("TOP ERROR EXAMPLES — P1_test_simplified (most confidently wrong)")
    print("=" * 70)
    for ex in [e for e in all_examples if e["split"] == "P1_test_simplified"][:3]:
        print(f"\n  ID:         {ex['id']}")
        print(f"  True label: {ex['true_label']}")
        print(f"  Predicted:  {ex['pred_label']}  (confidence={ex['confidence']:.3f})")
        print(f"  P0 text:    {ex['text_p0'][:200]} …")
        print(f"  Paraphrased:{ex['text'][:200]} …")
        print("-" * 70)


if __name__ == "__main__":
    main()

