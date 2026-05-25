"""
feature_drift_extended.py
--------------------------
Extends feature_drift.py to cover BOTH paraphrase tracks
(standard and simplified) and splits results by label
(human vs LLM) to show which class shifts more.

Also adds two additional features:
  - avg_word_length  : average characters per word (complexity proxy)
  - sentence_length_std : standard deviation of sentence lengths
    (captures structural homogenisation under paraphrasing)

Outputs
-------
    results/feature_drift_extended.csv   -- full table (all splits, by label)
    figures/drift_radar_std.png          -- radar chart, standard track
    figures/drift_ttr_comparison.png     -- TTR comparison across all conditions
    figures/drift_words_per_sent.png     -- words-per-sentence comparison

Usage
-----
    python src/feature_drift_extended.py

Requirements
------------
    pip install pandas matplotlib numpy
"""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


# ── text feature extraction ───────────────────────────────────────────────────
SENT_SPLIT = re.compile(r"[.!?]+")


def features(text: str) -> dict:
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = len(words)

    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    n_sents = max(1, len(sents))

    unique_words = len(set(words)) if words else 0
    ttr = unique_words / n_words if n_words else 0.0

    punct = sum(1 for ch in text if ch in ".,;:!?")
    punct_rate = punct / max(1, len(text))

    avg_word_len = (
        sum(len(w) for w in words) / n_words if n_words else 0.0
    )

    # Sentence length std dev (homogenisation proxy)
    sent_word_counts = [
        len(re.findall(r"\b\w+\b", s.lower())) for s in sents
    ]
    sent_len_std = float(np.std(sent_word_counts)) if len(sent_word_counts) > 1 else 0.0

    return {
        "words":            n_words,
        "sents":            n_sents,
        "words_per_sent":   n_words / n_sents,
        "ttr":              ttr,
        "punct_rate":       punct_rate,
        "uniq_word_ratio":  unique_words / max(1, n_words),
        "avg_word_len":     avg_word_len,
        "sent_len_std":     sent_len_std,
    }


def summarize(name: str, rows: list, label_filter: str = None) -> dict:
    if label_filter:
        rows = [r for r in rows if r.get("label") == label_filter]
    if not rows:
        return {}
    feats = [features(r["text"]) for r in rows]
    df    = pd.DataFrame(feats)
    out   = df.mean(numeric_only=True).to_dict()
    out["split"]  = name
    out["label"]  = label_filter or "all"
    out["n"]      = len(rows)
    return out


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Load all test splits
    test_ids = set(
        Path(os.path.join("data", "splits", "test_ids.txt"))
        .read_text(encoding="utf-8")
        .splitlines()
    )
    p0_rows  = [
        r for r in load_jsonl(os.path.join("data", "p0", "p0.jsonl"))
        if r["id"] in test_ids
    ]

    splits = {
        "P0_test":            p0_rows,
        "P1_test_standard":   load_jsonl(os.path.join("data", "p1", "p1_test.jsonl")),
        "P2_test_standard":   load_jsonl(os.path.join("data", "p2", "p2_test.jsonl")),
        "P1_test_simplified": load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        "P2_test_simplified": load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    }

    # ── compute features per split × label ────────────────────────────────────
    records = []
    for split_name, rows in splits.items():
        for lbl in ("all", "human", "llm"):
            row = summarize(split_name, rows, None if lbl == "all" else lbl)
            if row:
                records.append(row)

    df = pd.DataFrame(records)
    out_csv = os.path.join("results", "feature_drift_extended.csv")
    df.to_csv(out_csv, index=False)
    print(df[df["label"] == "all"].to_string(index=False))
    print(f"\nSaved: {out_csv}")

    # ── Plot 1: TTR across all conditions (all labels) ────────────────────────
    df_all = df[df["label"] == "all"].copy()

    std_splits = ["P0_test", "P1_test_standard",   "P2_test_standard"]
    sim_splits = ["P0_test", "P1_test_simplified", "P2_test_simplified"]
    stage_lbl  = ["P0", "P1", "P2"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    metrics_to_plot = [
        ("ttr",            "Type-Token Ratio (TTR)"),
        ("words_per_sent", "Words per Sentence"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
        std_vals = [
            df_all.loc[df_all["split"] == s, metric].values[0]
            for s in std_splits
        ]
        sim_vals = [
            df_all.loc[df_all["split"] == s, metric].values[0]
            for s in sim_splits
        ]
        ax.plot(stage_lbl, std_vals, marker="o", lw=2,
                label="Standard", color="#ff7f0e")
        ax.plot(stage_lbl, sim_vals, marker="s", lw=2, linestyle="--",
                label="Simplified", color="#9467bd")
        ax.set_xlabel("Paraphrase Stage", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{ylabel} vs Paraphrase Stage", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Linguistic Feature Drift: Standard vs Simplified Paraphrasing",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out1 = os.path.join("figures", "drift_ttr_comparison.png")
    plt.savefig(out1, dpi=200)
    plt.close()
    print(f"Saved: {out1}")

    # ── Plot 2: Human vs LLM TTR drift ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, (track_splits, track_name) in zip(
        axes,
        [(std_splits, "Standard"), (sim_splits, "Simplified")]
    ):
        for lbl, color, marker in [
            ("human", "#1f77b4", "o"),
            ("llm",   "#d62728", "s"),
        ]:
            df_lbl = df[df["label"] == lbl]
            vals   = [
                df_lbl.loc[df_lbl["split"] == s, "ttr"].values[0]
                for s in track_splits
            ]
            ax.plot(
                stage_lbl, vals,
                marker=marker, lw=2, color=color,
                label=f"{lbl.upper()} text",
            )
        ax.set_xlabel("Paraphrase Stage", fontsize=11)
        ax.set_ylabel("Type-Token Ratio (TTR)", fontsize=11)
        ax.set_title(f"{track_name} track: Human vs LLM TTR", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "TTR Drift by Label: Are Human and LLM texts converging?",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out2 = os.path.join("figures", "drift_human_vs_llm_ttr.png")
    plt.savefig(out2, dpi=200)
    plt.close()
    print(f"Saved: {out2}")

    # ── Plot 3: Full feature drift heatmap ────────────────────────────────────
    feat_cols = ["words", "sents", "words_per_sent", "ttr",
                 "punct_rate", "avg_word_len", "sent_len_std"]

    df_heat = df_all.set_index("split")[feat_cols]
    # Normalize each column to [0, 1] for heatmap
    df_norm = (df_heat - df_heat.min()) / (df_heat.max() - df_heat.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(df_norm.values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(feat_cols)))
    ax.set_xticklabels(feat_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(df_norm)))
    ax.set_yticklabels(df_norm.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Normalized value")
    ax.set_title(
        "Feature Drift Heatmap Across Paraphrase Conditions\n"
        "(normalized per feature; green=low, red=high)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    out3 = os.path.join("figures", "drift_heatmap.png")
    plt.savefig(out3, dpi=200)
    plt.close()
    print(f"Saved: {out3}")


if __name__ == "__main__":
    main()
