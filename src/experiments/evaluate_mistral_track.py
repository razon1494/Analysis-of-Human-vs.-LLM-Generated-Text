"""
evaluate_mistral_track.py
-------------------------
Evaluate all detectors on the Mistral-7B paraphrase tracks (standard +
simplified) and produce a 3-paraphraser comparison table and figure.

Prerequisite: run src/paraphrase_test_mistral.py first.

Outputs (results/eval/)
-----------------------
  mistral_metrics_flat.csv         — acc/F1/AUROC per detector per split
  mistral_hardness_buckets.csv     — Hard-bucket F1 per detector
  mistral_aurc.csv                 — AURC per detector x track
  paraphraser_comparison_full.csv  — Llama vs NLLB vs Mistral side-by-side

Figures
-------
  figures/fig16_mistral_robustness.png       — trajectory on Mistral tracks
  figures/fig17_three_paraphraser_comparison.png — full 3-way comparison
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
from lib.io import load_jsonl, load_test_ids, to_xy
from lib.detectors import build_word_tfidf_lr, build_char_tfidf_lr
from lib.metrics import point_metrics, area_under_robustness_curve
from lib.hardness import hardness_readability

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

MISTRAL_STANDARD_TRACK   = ["P0_test", "P1_test_mistral",            "P2_test_mistral"]
MISTRAL_SIMPLIFIED_TRACK = ["P0_test", "P1_test_mistral_simplified",  "P2_test_mistral_simplified"]
ALL_MISTRAL_SPLITS = {
    "P1_test_mistral":           ROOT / "data" / "p1" / "p1_test_mistral.jsonl",
    "P2_test_mistral":           ROOT / "data" / "p2" / "p2_test_mistral.jsonl",
    "P1_test_mistral_simplified": ROOT / "data" / "p1" / "p1_test_mistral_simplified.jsonl",
    "P2_test_mistral_simplified": ROOT / "data" / "p2" / "p2_test_mistral_simplified.jsonl",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_splits() -> dict:
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all   = load_jsonl(paths.P0_PATH)
    p0_test  = [r for r in p0_all if r["id"] in test_ids]

    splits = {"P0_test": p0_test}
    for name, fpath in ALL_MISTRAL_SPLITS.items():
        if not fpath.exists():
            print(f"ERROR: {fpath} not found.")
            print("Run: python src/paraphrase_test_mistral.py first.")
            sys.exit(1)
        splits[name] = load_jsonl(fpath)
    return splits


def train_detectors() -> dict:
    train_ids  = load_test_ids(paths.TRAIN_IDS)
    p0_all     = load_jsonl(paths.P0_PATH)
    train_rows = [r for r in p0_all if r["id"] in train_ids]
    X_tr, y_tr = to_xy(train_rows)
    return {
        "word_tfidf_lr": build_word_tfidf_lr(X_tr, y_tr),
        "char_tfidf_lr": build_char_tfidf_lr(X_tr, y_tr),
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_split(rows: list, detector, det_name: str, split: str) -> dict:
    X, y_true = to_xy(rows)
    y_prob = detector.predict_proba(X)
    y_pred = (y_prob >= 0.5).astype(int)
    m = point_metrics(y_true, y_pred, y_prob)
    m.update({"detector": det_name, "split": split})
    return m


def compute_aurc(df: pd.DataFrame, track: list, det_name: str) -> list:
    rows = []
    sub = df[df["detector"] == det_name].set_index("split")
    for metric in ["acc", "f1", "auroc"]:
        vals = [sub.loc[s, metric] for s in track if s in sub.index]
        if len(vals) >= 2:
            rows.append({
                "detector": det_name,
                "track":    "mistral_standard" if "mistral_simplified" not in track[1] else "mistral_simplified",
                "metric":   metric,
                "aurc":     round(area_under_robustness_curve(vals), 4),
            })
    return rows


def compute_hardness_buckets(splits_data: dict, detectors: dict) -> list:
    p0_rows  = splits_data["P0_test"]
    p0_texts = [r["text"] for r in p0_rows]
    fk_ha    = hardness_readability(p0_texts)
    id_to_bucket = {r["id"]: b for r, b in zip(p0_rows, fk_ha.buckets)}

    rows_out = []
    for det_name, det in detectors.items():
        for split_name, split_rows in splits_data.items():
            id_to_row = {r["id"]: r for r in split_rows}
            aligned   = [id_to_row[r["id"]] for r in p0_rows if r["id"] in id_to_row]
            if len(aligned) < len(p0_rows):
                continue
            X, y = to_xy(aligned)
            y_prob = det.predict_proba(X)
            y_pred = (y_prob >= 0.5).astype(int)

            for bucket in ["Easy", "Medium", "Hard"]:
                mask = np.array([id_to_bucket.get(r["id"], "") == bucket
                                 for r in p0_rows if r["id"] in id_to_row])
                if mask.sum() < 3:
                    continue
                m = point_metrics(y[mask], y_pred[mask], y_prob[mask])
                rows_out.append({
                    "detector": det_name, "split": split_name,
                    "bucket": bucket, "n": int(mask.sum()),
                    "f1": round(m["f1"], 4), "acc": round(m["acc"], 4),
                })
    return rows_out


# ── 3-paraphraser comparison ─────────────────────────────────────────────────

def build_comparison_table(df_mistral: pd.DataFrame) -> pd.DataFrame:
    """Load Llama + NLLB results and merge with Mistral into one table."""
    out_dir = paths.RESULTS / "eval"
    rows = []

    # Mistral simplified P2
    for det in ["word_tfidf_lr", "char_tfidf_lr"]:
        hard = df_mistral[
            (df_mistral["detector"] == det) &
            (df_mistral["split"]    == "P2_test_mistral_simplified") &
            (df_mistral["bucket"]   == "Hard")
        ]
        if len(hard):
            rows.append({"detector": det, "paraphraser": "Mistral-7B (simplified)",
                         "p2_hard_f1": float(hard["f1"].values[0])})
        hard_std = df_mistral[
            (df_mistral["detector"] == det) &
            (df_mistral["split"]    == "P2_test_mistral") &
            (df_mistral["bucket"]   == "Hard")
        ]
        if len(hard_std):
            rows.append({"detector": det, "paraphraser": "Mistral-7B (standard)",
                         "p2_hard_f1": float(hard_std["f1"].values[0])})

    # NLLB
    nllb_path = out_dir / "nllb_hardness_buckets.csv"
    if nllb_path.exists():
        df_nllb = pd.read_csv(nllb_path)
        for det in ["word_tfidf_lr", "char_tfidf_lr"]:
            hard = df_nllb[
                (df_nllb["detector"] == det) &
                (df_nllb["split"]    == "P2_test_nllb") &
                (df_nllb["bucket"]   == "Hard")
            ]
            if len(hard):
                rows.append({"detector": det, "paraphraser": "NLLB back-translation",
                             "p2_hard_f1": float(hard["f1"].values[0])})

    # Llama (multi-seed)
    llama_path = out_dir / "multiseed_summary_buckets.csv"
    if llama_path.exists():
        df_llama = pd.read_csv(llama_path)
        fk = df_llama[
            (df_llama["hardness"] == "readability_fk") &
            (df_llama["bucket"]   == "Hard") &
            (df_llama["metric"]   == "f1")
        ]
        for det in ["word_tfidf_lr", "char_tfidf_lr"]:
            hard = fk[
                (fk["detector"] == det) & (fk["split"] == "P2_test_simplified")
            ]
            if len(hard):
                rows.append({"detector": det, "paraphraser": "Llama-3.1-8B (simplified)",
                             "p2_hard_f1": float(hard["mean"].values[0])})

    return pd.DataFrame(rows)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_mistral_robustness(df_flat: pd.DataFrame) -> None:
    tracks = {
        "Standard": (MISTRAL_STANDARD_TRACK,   ["P0", "P1\nstd", "P2\nstd"]),
        "Simplified": (MISTRAL_SIMPLIFIED_TRACK, ["P0", "P1\nsim", "P2\nsim"]),
    }
    colors = {"word_tfidf_lr": "#1f77b4", "char_tfidf_lr": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (track_name, (splits, xlabels)) in zip(axes, tracks.items()):
        for det, color in colors.items():
            vals = []
            for sp in splits:
                row = df_flat[(df_flat["detector"] == det) & (df_flat["split"] == sp)]
                vals.append(float(row["f1"].values[0]) if len(row) else np.nan)
            ax.plot(xlabels, vals, marker="o", color=color,
                    label=det.replace("_", " "), linewidth=2)
        ax.set_title(f"F1 — Mistral-7B {track_name} track")
        ax.set_ylabel("F1")
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Detector robustness under Mistral-7B paraphrasing\n"
                 "(different LLM family from Llama-3.1-8B generator)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig16_mistral_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_three_paraphraser_comparison(df_comparison: pd.DataFrame) -> None:
    """Bar chart: Hard-bucket F1 at P2 across all 3 paraphrasers."""
    if df_comparison.empty:
        print("Skipping fig17: comparison table is empty.")
        return

    paraphrasers = [
        "Llama-3.1-8B (simplified)",
        "Mistral-7B (simplified)",
        "NLLB back-translation",
        "Mistral-7B (standard)",
    ]
    colors = {
        "Llama-3.1-8B (simplified)": "#ff7f0e",
        "Mistral-7B (simplified)":   "#2ca02c",
        "NLLB back-translation":     "#9467bd",
        "Mistral-7B (standard)":     "#8c564b",
    }
    detectors = ["word_tfidf_lr", "char_tfidf_lr"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, det in zip(axes, detectors):
        sub = df_comparison[df_comparison["detector"] == det]
        x = np.arange(len(paraphrasers))
        for i, para in enumerate(paraphrasers):
            row = sub[sub["paraphraser"] == para]
            f1  = float(row["p2_hard_f1"].values[0]) if len(row) else np.nan
            ax.bar(i, f1 if not np.isnan(f1) else 0,
                   color=colors.get(para, "gray"), alpha=0.85, width=0.6)
            if not np.isnan(f1):
                ax.text(i, f1 + 0.02, f"{f1:.3f}", ha="center",
                        va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(paraphrasers, rotation=20, ha="right", fontsize=8)
        ax.set_title(det.replace("_", " "), fontsize=10)
        ax.set_ylabel("Hard-bucket F1 at P2\n(readability_fk hardness)")
        ax.set_ylim(0, 1.15)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.4)
        ax.grid(True, alpha=0.2, axis="y")

    fig.suptitle("Hard-bucket F1 collapse: consistent across all paraphrasers\n"
                 "(readability_fk hardness; P0 baseline ≈ 0.88–0.93)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig17_three_paraphraser_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading Mistral paraphrase splits...")
    splits_data = load_splits()

    print("[2/5] Training detectors...")
    detectors = train_detectors()

    print("[3/5] Computing flat metrics...")
    metrics_rows = []
    for split_name, rows in splits_data.items():
        for det_name, det in detectors.items():
            m = evaluate_split(rows, det, det_name, split_name)
            metrics_rows.append(m)
            print(f"  {det_name:<20} {split_name:<30}  "
                  f"acc={m['acc']:.4f}  f1={m['f1']:.4f}  auroc={m['auroc']:.4f}")

    df_flat = pd.DataFrame(metrics_rows)
    df_flat.to_csv(out_dir / "mistral_metrics_flat.csv", index=False)

    print("\n[4/5] Computing AURC and Hard-bucket F1...")
    aurc_rows = []
    for det in detectors:
        aurc_rows += compute_aurc(df_flat, MISTRAL_STANDARD_TRACK, det)
        aurc_rows += compute_aurc(df_flat, MISTRAL_SIMPLIFIED_TRACK, det)
    df_aurc = pd.DataFrame(aurc_rows)
    df_aurc.to_csv(out_dir / "mistral_aurc.csv", index=False)
    print("\nMistral AURC:")
    print(df_aurc.to_string(index=False))

    bucket_rows = compute_hardness_buckets(splits_data, detectors)
    df_buckets  = pd.DataFrame(bucket_rows)
    df_buckets.to_csv(out_dir / "mistral_hardness_buckets.csv", index=False)

    hard = df_buckets[df_buckets["bucket"] == "Hard"][
        ["detector", "split", "f1"]].sort_values(["detector", "split"])
    print("\nHard-bucket F1 (readability_fk) on Mistral tracks:")
    print(hard.to_string(index=False))

    print("\n[5/5] Generating figures...")
    fig_mistral_robustness(df_flat)

    df_comparison = build_comparison_table(df_buckets)
    df_comparison.to_csv(out_dir / "paraphraser_comparison_full.csv", index=False)
    fig_three_paraphraser_comparison(df_comparison)

    # Final summary
    print("\n" + "=" * 65)
    print("3-PARAPHRASER COMPARISON — Hard-bucket F1 at P2 (readability_fk)")
    print("=" * 65)
    for det in detectors:
        sub = df_comparison[df_comparison["detector"] == det].sort_values("paraphraser")
        print(f"\n  {det}:")
        for _, row in sub.iterrows():
            print(f"    {row['paraphraser']:<35} F1={row['p2_hard_f1']:.3f}")

    print("\nDone. P6 complete.")


if __name__ == "__main__":
    main()
