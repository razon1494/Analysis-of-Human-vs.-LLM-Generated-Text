"""
analyze_top_features.py
-----------------------
Extracts and visualises the most discriminative n-gram features
from the trained TF-IDF + Logistic Regression detector.

Produces:
  - A ranked table of top-N features for each class (LLM / Human)
  - A horizontal bar chart showing top features per class
  - Per-feature analysis: does the feature survive paraphrasing?
    (i.e. does its average TF-IDF weight drop across P0 → P1 → P2?)

Outputs
-------
    results/top_features.csv              -- top features table
    results/feature_survival.csv          -- feature weight drift table
    figures/top_features_bar.png          -- bar chart of top features
    figures/feature_weight_drift.png      -- line plot of weight drift

Usage
-----
    python src/analyze_top_features.py

Requirements
------------
    pip install scikit-learn joblib numpy pandas matplotlib
"""

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── config ────────────────────────────────────────────────────────────────────
TOP_N       = 20        # features per class to show in bar chart
SURVIVAL_N  = 15        # features to track across paraphrase stages

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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # ── load model ────────────────────────────────────────────────────────────
    vectorizer = joblib.load(os.path.join(RESULTS_DIR, "vectorizer.joblib"))
    clf        = joblib.load(os.path.join(RESULTS_DIR, "model.joblib"))

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs         = clf.coef_[0]           # shape: (n_features,)

    # Positive coef  → predicts LLM
    # Negative coef  → predicts Human
    top_llm_idx   = np.argsort(coefs)[::-1][:TOP_N]
    top_human_idx = np.argsort(coefs)[:TOP_N]

    top_llm_feats   = [(feature_names[i], float(coefs[i])) for i in top_llm_idx]
    top_human_feats = [(feature_names[i], float(coefs[i])) for i in top_human_idx]

    # ── print summary ─────────────────────────────────────────────────────────
    print(f"\nTop {TOP_N} features predicting LLM-generated text:")
    print(f"{'Feature':<30} {'Coefficient':>12}")
    print("-" * 44)
    for feat, coef in top_llm_feats:
        print(f"  {feat:<28} {coef:>12.4f}")

    print(f"\nTop {TOP_N} features predicting Human-written text:")
    print(f"{'Feature':<30} {'Coefficient':>12}")
    print("-" * 44)
    for feat, coef in top_human_feats:
        print(f"  {feat:<28} {coef:>12.4f}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    records = (
        [{"rank": i+1, "class": "llm",   "feature": f, "coef": c}
         for i, (f, c) in enumerate(top_llm_feats)]
        +
        [{"rank": i+1, "class": "human", "feature": f, "coef": c}
         for i, (f, c) in enumerate(top_human_feats)]
    )
    df_feats = pd.DataFrame(records)
    feat_csv = os.path.join(RESULTS_DIR, "top_features.csv")
    df_feats.to_csv(feat_csv, index=False)
    print(f"\nSaved: {feat_csv}")

    # ── bar chart ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # LLM side
    ax = axes[0]
    feats_l = [f for f, _ in top_llm_feats[::-1]]
    coefs_l = [c for _, c in top_llm_feats[::-1]]
    ax.barh(feats_l, coefs_l, color="#d62728", alpha=0.85)
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=10)
    ax.set_title(f"Top {TOP_N} LLM-indicative features", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

    # Human side
    ax = axes[1]
    feats_h = [f for f, _ in top_human_feats]
    coefs_h = [c for _, c in top_human_feats]
    ax.barh(feats_h, coefs_h, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Logistic Regression Coefficient", fontsize=10)
    ax.set_title(f"Top {TOP_N} Human-indicative features", fontsize=11, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.7)
    ax.grid(axis="x", alpha=0.3)
    ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Most Discriminative TF-IDF N-gram Features\n(Word n-gram detector)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    bar_out = os.path.join(FIGURES_DIR, "top_features_bar.png")
    plt.savefig(bar_out, dpi=200)
    plt.close()
    print(f"Saved: {bar_out}")

    # ── feature weight survival across paraphrase stages ─────────────────────
    # Track average TF-IDF activation of top-LLM features as text is paraphrased.
    # A feature that "survives" paraphrasing stays active; one that erodes drops.

    test_ids = set(Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines())
    p0_rows  = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    # Only track LLM rows (label=llm) for LLM features
    def llm_rows_only(rows):
        return [r for r in rows if r["label"] == "llm"]

    stages = {
        "P0": llm_rows_only(p0_rows),
        "P1_std": llm_rows_only(load_jsonl(os.path.join("data", "p1", "p1_test.jsonl"))),
        "P2_std": llm_rows_only(load_jsonl(os.path.join("data", "p2", "p2_test.jsonl"))),
        "P1_sim": llm_rows_only(load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl"))),
        "P2_sim": llm_rows_only(load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl"))),
    }

    # Top-N LLM features to track
    tracked_features = [f for f, _ in top_llm_feats[:SURVIVAL_N]]
    tracked_indices  = [np.where(feature_names == f)[0][0] for f in tracked_features]

    survival_records = []
    stage_means      = {}

    for stage_name, rows in stages.items():
        if not rows:
            continue
        X  = [r["text"] for r in rows]
        Xv = vectorizer.transform(X)
        # mean TF-IDF weight per tracked feature across LLM test docs
        mean_weights = np.array(Xv[:, tracked_indices].mean(axis=0)).flatten()
        stage_means[stage_name] = mean_weights
        for feat, w in zip(tracked_features, mean_weights):
            survival_records.append({
                "stage": stage_name,
                "feature": feat,
                "mean_tfidf": round(float(w), 6),
            })

    df_surv = pd.DataFrame(survival_records)
    surv_csv = os.path.join(RESULTS_DIR, "feature_survival.csv")
    df_surv.to_csv(surv_csv, index=False)
    print(f"Saved: {surv_csv}")

    # ── feature survival line plot ────────────────────────────────────────────
    stage_order = ["P0", "P1_std", "P2_std", "P1_sim", "P2_sim"]
    stage_labels = {
        "P0": "P0", "P1_std": "P1\n(std)", "P2_std": "P2\n(std)",
        "P1_sim": "P1\n(sim)", "P2_sim": "P2\n(sim)",
    }

    # Plot top-8 LLM features only for readability
    plot_features = tracked_features[:8]
    plot_indices  = [tracked_features.index(f) for f in plot_features]

    fig, ax = plt.subplots(figsize=(10, 5))

    cmap = plt.get_cmap("tab10")
    for j, feat in enumerate(plot_features):
        y_vals = []
        for s in stage_order:
            if s in stage_means:
                y_vals.append(stage_means[s][plot_indices[j]])
            else:
                y_vals.append(np.nan)
        ax.plot(
            [stage_labels[s] for s in stage_order],
            y_vals,
            marker="o",
            lw=1.6,
            color=cmap(j),
            label=f'"{feat}"',
        )

    ax.set_xlabel("Paraphrase Stage", fontsize=11)
    ax.set_ylabel("Mean TF-IDF Weight (LLM docs)", fontsize=11)
    ax.set_title(
        "Feature Weight Survival Under Paraphrasing\n(Top LLM-indicative features)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    surv_plot = os.path.join(FIGURES_DIR, "feature_weight_drift.png")
    plt.savefig(surv_plot, dpi=200)
    plt.close()
    print(f"Saved: {surv_plot}")

    print("\nDone. Key insight to report:")
    print(
        "  Features that drop sharply from P0 → P1 are the ones driving\n"
        "  signature erosion. Features that remain high are more robust signals."
    )


if __name__ == "__main__":
    main()
