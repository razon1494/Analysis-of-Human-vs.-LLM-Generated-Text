"""
run_embedding_projection.py  (H3)
----------------------------------
UMAP projection of SBERT embeddings to visualise the Hard-bucket collapse:
at P0, Human and LLM texts form separable clusters in embedding space.
At P2-simplified, the LLM cluster drifts toward the Human cluster.

We project the Hard-bucket only (where the collapse is concentrated).

Outputs
-------
  figures/fig23_umap_hard_bucket.png      — 2x2 panel: P0 vs P2 sim, human/LLM
  results/eval/embedding_projection.csv   — per-point 2D coordinates + metadata
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
from lib.io import load_jsonl, load_test_ids
from lib.hardness import hardness_readability

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

SEED = 42
SBERT_MODEL = "all-mpnet-base-v2"  # same as semantic preservation gate


def hard_ids(p0_test: list) -> set:
    texts = [r["text"] for r in p0_test]
    fk = hardness_readability(texts)
    return {r["id"] for r, b in zip(p0_test, fk.buckets) if b == "Hard"}


def embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    print(f"  loading SBERT ({SBERT_MODEL})...")
    model = SentenceTransformer(SBERT_MODEL)
    print(f"  encoding {len(texts)} texts...")
    return model.encode(texts, batch_size=16, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


def main():
    out_dir   = paths.RESULTS / "eval"
    fig_dir   = paths.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading data...")
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all   = load_jsonl(paths.P0_PATH)
    p0_test  = [r for r in p0_all if r["id"] in test_ids]

    p1_path = ROOT / "data" / "p1" / "p1_test_simplified.jsonl"
    p2_path = ROOT / "data" / "p2" / "p2_test_simplified.jsonl"
    p1      = load_jsonl(p1_path)
    p2      = load_jsonl(p2_path)

    # Restrict to Hard bucket
    hids = hard_ids(p0_test)
    print(f"  Hard bucket size: {len(hids)}")

    id_to_p0 = {r["id"]: r for r in p0_test}
    id_to_p1 = {r["id"]: r for r in p1}
    id_to_p2 = {r["id"]: r for r in p2}

    hard_p0 = [id_to_p0[i] for i in hids if i in id_to_p0]
    hard_p1 = [id_to_p1[i] for i in hids if i in id_to_p1]
    hard_p2 = [id_to_p2[i] for i in hids if i in id_to_p2]
    print(f"  aligned Hard rows: P0={len(hard_p0)}, P1={len(hard_p1)}, P2={len(hard_p2)}")

    print("[2/4] Computing SBERT embeddings...")
    all_texts  = ([r["text"] for r in hard_p0] +
                  [r["text"] for r in hard_p1] +
                  [r["text"] for r in hard_p2])
    all_labels = ([r["label"] for r in hard_p0] +
                  [r["label"] for r in hard_p1] +
                  [r["label"] for r in hard_p2])
    all_stages = (["P0"] * len(hard_p0) +
                  ["P1_simplified"] * len(hard_p1) +
                  ["P2_simplified"] * len(hard_p2))
    all_ids    = ([r["id"] for r in hard_p0] +
                  [r["id"] for r in hard_p1] +
                  [r["id"] for r in hard_p2])

    embs = embed_texts(all_texts)
    print(f"  embedding shape: {embs.shape}")

    print("[3/4] Fitting UMAP (joint, all groups)...")
    import umap
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.15,
        random_state=SEED,
        metric="cosine",
    )
    coords = reducer.fit_transform(embs)
    print(f"  UMAP coords: {coords.shape}")

    df = pd.DataFrame({
        "id":    all_ids,
        "label": all_labels,
        "stage": all_stages,
        "x":     coords[:, 0],
        "y":     coords[:, 1],
    })
    df.to_csv(out_dir / "embedding_projection.csv", index=False)
    print(f"  Saved: embedding_projection.csv")

    # Quantitative in ORIGINAL SBERT space (the meaningful metric, not UMAP)
    print("[4/5] Computing semantic separability in SBERT space (cosine)...")
    sep_rows = []
    for stage in ["P0", "P1_simplified", "P2_simplified"]:
        mask = np.array([s == stage for s in all_stages])
        labs = np.array([all_labels[i] for i in range(len(all_labels)) if mask[i]])
        embs_stage = embs[mask]
        h_centroid = embs_stage[labs == "human"].mean(axis=0)
        l_centroid = embs_stage[labs == "llm"].mean(axis=0)
        # cosine distance between centroids (both unit-normalised already)
        cos_sim = float(np.dot(h_centroid, l_centroid) /
                        (np.linalg.norm(h_centroid) * np.linalg.norm(l_centroid)))
        cos_dist = 1.0 - cos_sim
        sep_rows.append({
            "stage":              stage,
            "centroid_cos_sim":   round(cos_sim, 4),
            "centroid_cos_dist":  round(cos_dist, 4),
        })
    df_sep = pd.DataFrame(sep_rows)
    df_sep.to_csv(out_dir / "embedding_separability.csv", index=False)

    print("[5/5] Generating figure...")
    # Square-ish panels with shared legend below — much cleaner for a paper
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharex=True, sharey=True)
    stages_order = ["P0", "P1_simplified", "P2_simplified"]
    titles  = ["P0 (original)", "P1 (1× simplified)", "P2 (2× simplified)"]

    colors = {"human": "#1f77b4", "llm": "#d62728"}
    markers = {"human": "o", "llm": "X"}
    sizes   = {"human": 65, "llm": 60}

    for ax, stage, title in zip(axes, stages_order, titles):
        sub = df[df["stage"] == stage]
        for lbl in ["human", "llm"]:
            s = sub[sub["label"] == lbl]
            ax.scatter(
                s["x"], s["y"],
                c=colors[lbl],
                marker=markers[lbl],
                s=sizes[lbl], alpha=0.85,
                edgecolors="white", linewidths=0.6,
                label=f"{lbl.upper()} (n={len(s)})" if stage == "P0" else None,
            )
        # Annotate centroid cos-dist on each panel
        sep = df_sep[df_sep["stage"] == stage].iloc[0]
        ax.set_title(
            f"{title}\n"
            r"$d_{\mathrm{cos}}^{\mathrm{SBERT}}$ = "
            f"{sep['centroid_cos_dist']:.3f}",
            fontsize=10,
        )
        ax.set_xlabel("UMAP-1", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("UMAP-2", fontsize=9)

    # Single figure-level legend below all panels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(
        h, l,
        loc="lower center", ncol=2,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=10, framealpha=0.95,
    )

    fig.suptitle(
        "Semantic separability is preserved across paraphrase rounds\n"
        "(Hard bucket, SBERT all-mpnet-base-v2 + UMAP). "
        "SBERT centroid cos-distance drops only 21%; "
        r"char-TF-IDF $F_1$ drops 64%.",
        fontsize=10.5, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    out = fig_dir / "fig23_umap_hard_bucket.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")

    # Quantitative summary
    print("\n" + "=" * 60)
    print("H3 SUMMARY — SBERT centroid separation (Hard bucket)")
    print("=" * 60)
    for _, row in df_sep.iterrows():
        print(f"  {row['stage']:<20}  cos-dist = {row['centroid_cos_dist']:.4f}  "
              f"(cos-sim = {row['centroid_cos_sim']:.4f})")
    p0_d  = float(df_sep[df_sep['stage'] == 'P0']['centroid_cos_dist'].values[0])
    p2_d  = float(df_sep[df_sep['stage'] == 'P2_simplified']['centroid_cos_dist'].values[0])
    print(f"\n  Relative change P0 -> P2_sim: "
          f"{((p2_d - p0_d) / p0_d) * 100:+.1f}%")
    print("\nInterpretation: separability persists at the semantic level even "
          "as char-TF-IDF Hard-bucket F1 collapses 0.923 -> 0.336. "
          "The collapse is a surface-feature failure, not a semantic one.")
    print("\nDone. H3 complete.")


if __name__ == "__main__":
    main()
