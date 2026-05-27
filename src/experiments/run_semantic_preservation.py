"""
run_semantic_preservation.py
----------------------------
Semantic-preservation gate. Computes SBERT cosine similarity between every
P0 sample and its paraphrased counterparts at P1/P2 of both tracks.

Why this exists
---------------
A reviewer attack: "If your paraphraser produces semantic drift or
incoherent text, classifier confusion is meaningless — you'd be measuring
nonsense detection, not paraphrase robustness."

This script provides the defense. For each row:

  sim_p0_p1 = cosine( SBERT(P0_text), SBERT(P1_text) )
  sim_p0_p2 = cosine( SBERT(P0_text), SBERT(P2_text) )

Report distributions and flag rows with low semantic similarity
(< 0.70 with the default mpnet model). Stratify by label, by track,
and by detector hardness bucket.

Model: sentence-transformers/all-mpnet-base-v2 (768d).

Outputs
-------
    results/eval/semantic_preservation.csv     -- per-row similarity matrix
    results/eval/semantic_summary.csv          -- per-split summary
    figures/fig11_semantic_preservation.png    -- distribution + label split
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl, load_test_ids


mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LOW_SIM_THRESHOLD = 0.70   # samples below this are flagged as poor paraphrases
SPLITS = ["P1_test_standard", "P2_test_standard",
          "P1_test_simplified", "P2_test_simplified"]


def cosine_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally-shaped matrices."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return (a_norm * b_norm).sum(axis=1)


def main():
    print(f"Loading SBERT model ({MODEL_NAME})... (first run downloads ~420MB)")
    model = SentenceTransformer(MODEL_NAME)

    # Load P0 test rows (the reference)
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all = load_jsonl(paths.P0_PATH)
    p0_test = [r for r in p0_all if r["id"] in test_ids]
    p0_by_id = {r["id"]: r for r in p0_test}

    # Load paraphrase splits
    para_data = {sp: load_jsonl(paths.PARAPHRASE_PATHS[sp]) for sp in SPLITS}

    # Match by ID — verify alignment first
    print("\nID alignment check:")
    for sp in SPLITS:
        rows = para_data[sp]
        matched = sum(1 for r in rows if r["id"] in p0_by_id)
        print(f"  {sp}: {matched}/{len(rows)} rows match P0 IDs")

    # Encode P0 texts ONCE
    p0_ids = list(p0_by_id.keys())
    p0_texts = [p0_by_id[i]["text"] for i in p0_ids]
    print(f"\nEncoding {len(p0_texts)} P0 texts...")
    p0_embs = model.encode(p0_texts, batch_size=16, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=False)
    p0_emb_by_id = {i: e for i, e in zip(p0_ids, p0_embs)}

    # Now encode each paraphrase split and compute per-row cosine
    rows_out = []
    for sp in SPLITS:
        rows = para_data[sp]
        # Filter to rows with valid P0 counterpart
        rows = [r for r in rows if r["id"] in p0_emb_by_id]
        ids = [r["id"] for r in rows]
        labels = [r["label"] for r in rows]
        texts = [r["text"] for r in rows]
        print(f"\nEncoding {sp} ({len(texts)} texts)...")
        p_embs = model.encode(texts, batch_size=16, show_progress_bar=True,
                              convert_to_numpy=True, normalize_embeddings=False)

        # Stack P0 embeddings in the same order
        p0_aligned = np.stack([p0_emb_by_id[i] for i in ids])
        sims = cosine_pairwise(p0_aligned, p_embs)

        for i, lbl, s, tx in zip(ids, labels, sims, texts):
            rows_out.append({
                "id": i,
                "split": sp,
                "label": lbl,
                "sbert_cosine": round(float(s), 6),
                "p0_word_count": len(p0_by_id[i]["text"].split()),
                "paraphrase_word_count": len(tx.split()),
            })

    df = pd.DataFrame(rows_out)
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "semantic_preservation.csv", index=False)
    print(f"\nSaved: {out_dir / 'semantic_preservation.csv'}")

    # ── per-split summary ─────────────────────────────────────────────────────
    print("\nSemantic preservation summary:")
    print(f"  {'split':<25} {'n':>4}  {'mean':>7}  {'med':>7}  {'std':>7}  "
          f"{'min':>7}  {'p10':>7}  {'#<0.70':>7}")
    summary_rows = []
    for sp in SPLITS:
        sub = df[df["split"] == sp]
        sims = sub["sbert_cosine"].values
        for lbl in ["all", "human", "llm"]:
            if lbl == "all":
                vals = sims
                lbl_sub = sub
            else:
                lbl_sub = sub[sub["label"] == lbl]
                vals = lbl_sub["sbert_cosine"].values
            if len(vals) == 0:
                continue
            low_count = int((vals < LOW_SIM_THRESHOLD).sum())
            row = {
                "split": sp,
                "label": lbl,
                "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "p10": round(float(np.percentile(vals, 10)), 4),
                "p90": round(float(np.percentile(vals, 90)), 4),
                "max": round(float(np.max(vals)), 4),
                "n_low": low_count,
                "pct_low": round(100.0 * low_count / len(vals), 2),
            }
            summary_rows.append(row)
            if lbl == "all":
                print(f"  {sp:<25} {row['n']:>4}  {row['mean']:>7.4f}  "
                      f"{row['median']:>7.4f}  {row['std']:>7.4f}  "
                      f"{row['min']:>7.4f}  {row['p10']:>7.4f}  {low_count:>7}")
    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(out_dir / "semantic_summary.csv", index=False)
    print(f"Saved: {out_dir / 'semantic_summary.csv'}")

    # ── label-stratified statement ────────────────────────────────────────────
    print("\nDoes paraphraser preserve LLM-text semantics differently than human-text?")
    for sp in SPLITS:
        sub = df[df["split"] == sp]
        h = sub[sub["label"] == "human"]["sbert_cosine"].values
        l = sub[sub["label"] == "llm"]["sbert_cosine"].values
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(h, l, alternative="two-sided")
        print(f"  {sp:<25} human mean={np.mean(h):.4f}  llm mean={np.mean(l):.4f}  "
              f"diff={np.mean(h) - np.mean(l):+.4f}  Mann-Whitney p={p:.4f}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, sp in zip(axes.flat, SPLITS):
        sub = df[df["split"] == sp]
        h = sub[sub["label"] == "human"]["sbert_cosine"].values
        l = sub[sub["label"] == "llm"]["sbert_cosine"].values
        ax.hist(h, bins=20, alpha=0.55, color="#1f77b4", label=f"Human (n={len(h)})", edgecolor="white")
        ax.hist(l, bins=20, alpha=0.55, color="#d62728", label=f"LLM (n={len(l)})", edgecolor="white")
        ax.axvline(LOW_SIM_THRESHOLD, color="black", linestyle="--", lw=1, alpha=0.7,
                   label=f"flag<{LOW_SIM_THRESHOLD}")
        ax.set_title(f"{sp}\nmean = {sub['sbert_cosine'].mean():.3f}", fontsize=10)
        ax.set_xlabel("SBERT cosine vs P0")
        ax.set_ylabel("# samples")
        ax.set_xlim(0.0, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Semantic preservation: SBERT cosine between paraphrase and P0\n"
                 "(mpnet-base-v2; higher = more preserved)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig_out = paths.FIGURES / "fig11_semantic_preservation.png"
    plt.savefig(fig_out, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {fig_out}")

    # ── headline gate decision ───────────────────────────────────────────────
    print("\n*** Semantic preservation gate decision ***")
    for sp in SPLITS:
        sub = df[df["split"] == sp]
        med = sub["sbert_cosine"].median()
        low = (sub["sbert_cosine"] < LOW_SIM_THRESHOLD).sum()
        verdict = (
            "PASS" if med >= 0.80 and low / len(sub) < 0.10
            else "REVIEW" if med >= 0.70
            else "FAIL"
        )
        print(f"  {sp:<25}  median={med:.3f}  flagged={low}/{len(sub)}  ->  {verdict}")


if __name__ == "__main__":
    main()
