"""
evaluate_nllb_track.py
----------------------
Evaluate all three detectors on the NLLB back-translation paraphrase track
and compare Hard-bucket F1 collapse with the Llama paraphrase results.

Prerequisite: run src/paraphrase_test_nllb.py first to generate:
  data/p1/p1_test_nllb.jsonl
  data/p2/p2_test_nllb.jsonl

Outputs (all written to results/eval/)
---------------------------------------
  nllb_metrics_flat.csv          — acc/F1/AUROC per detector per split
  nllb_hardness_buckets.csv      — Hard-bucket F1 per hardness definition
  nllb_aurc.csv                  — AURC per detector
  nllb_paraphraser_comparison.csv — side-by-side Llama vs NLLB vs Mistral (if available)

Figures
-------
  figures/fig14_nllb_robustness.png         — 3-detector trajectory on NLLB track
  figures/fig15_paraphraser_comparison.png  — Hard-bucket F1 collapse by paraphraser
"""

from __future__ import annotations

import json
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
from lib.metrics import (
    point_metrics, bootstrap_metric_ci, area_under_robustness_curve,
    METRIC_FUNCS,
)
from lib.hardness import (
    hardness_readability,
)

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

NLLB_TRACK = ["P0_test", "P1_test_nllb", "P2_test_nllb"]
HARDNESS_DEFS = ["margin_self", "readability_fk", "ttr", "length"]
N_BOOT = 2000
SEED = 42


# ── Data loading ─────────────────────────────────────────────────────────────

def load_nllb_splits() -> dict:
    """Return {split_name: list_of_rows} for NLLB track."""
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all   = load_jsonl(paths.P0_PATH)
    p0_test  = [r for r in p0_all if r["id"] in test_ids]

    p1_path = ROOT / "data" / "p1" / "p1_test_nllb.jsonl"
    p2_path = ROOT / "data" / "p2" / "p2_test_nllb.jsonl"

    for p in [p1_path, p2_path]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            print("Run: python src/paraphrase_test_nllb.py first.")
            sys.exit(1)

    return {
        "P0_test":      p0_test,
        "P1_test_nllb": load_jsonl(p1_path),
        "P2_test_nllb": load_jsonl(p2_path),
    }


def train_detectors(p0_all: list) -> dict:
    """Train word + char TF-IDF detectors on the fixed train split."""
    train_ids = load_test_ids(paths.TRAIN_IDS)
    train_rows = [r for r in p0_all if r["id"] in train_ids]
    X_tr, y_tr = to_xy(train_rows)

    word_det = build_word_tfidf_lr(X_tr, y_tr)
    char_det = build_char_tfidf_lr(X_tr, y_tr)

    return {"word_tfidf_lr": word_det, "char_tfidf_lr": char_det}


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_split(rows: list, detector, det_name: str, split: str) -> dict:
    X, y_true = to_xy(rows)
    y_prob = detector.predict_proba(X)       # DetectorBundle returns 1-D array
    y_pred = (y_prob >= 0.5).astype(int)
    m = point_metrics(y_true, y_pred, y_prob)
    m.update({"detector": det_name, "split": split})
    return m


def compute_aurc(metrics_rows: list, detectors: list) -> list:
    rows = []
    df = pd.DataFrame(metrics_rows)
    split_order = NLLB_TRACK
    for det in detectors:
        sub = df[df["detector"] == det].set_index("split")
        for metric in ["acc", "f1", "auroc"]:
            vals = [sub.loc[s, metric] for s in split_order if s in sub.index]
            if len(vals) >= 2:
                aurc = area_under_robustness_curve(vals)
                rows.append({"detector": det, "metric": metric, "aurc": round(aurc, 4)})
    return rows


def compute_hardness_buckets(
    splits_data: dict, detectors: dict
) -> list:
    """Hard-bucket F1 under readability_fk (primary non-circular signal)."""
    rows_out = []
    p0_rows = splits_data["P0_test"]
    p0_texts = [r["text"] for r in p0_rows]

    for det_name, det in detectors.items():
        X0, y0 = to_xy(p0_rows)
        probs0  = det.predict_proba(X0)       # 1-D array
        fk_ha      = hardness_readability(p0_texts)   # returns HardnessAssignment
        buckets_fk = fk_ha.buckets                    # list of "Easy"/"Medium"/"Hard"

        id_to_bucket = {r["id"]: b for r, b in zip(p0_rows, buckets_fk)}

        for split_name, split_rows in splits_data.items():
            # align by ID
            id_to_row = {r["id"]: r for r in split_rows}
            aligned = [id_to_row[r["id"]] for r in p0_rows if r["id"] in id_to_row]
            if len(aligned) < len(p0_rows):
                continue
            X, y = to_xy(aligned)
            y_prob = det.predict_proba(X)         # 1-D array
            y_pred = (y_prob >= 0.5).astype(int)

            for bucket in ["Easy", "Medium", "Hard"]:
                mask = np.array([id_to_bucket.get(r["id"], "") == bucket
                                 for r in p0_rows if r["id"] in id_to_row])
                if mask.sum() < 3:
                    continue
                yt = y[mask]; yp = y_pred[mask]; ypr = y_prob[mask]
                m = point_metrics(yt, yp, ypr)
                rows_out.append({
                    "detector": det_name,
                    "hardness":  "readability_fk",
                    "split":    split_name,
                    "bucket":   bucket,
                    "n":        int(mask.sum()),
                    "f1":       round(m["f1"], 4),
                    "acc":      round(m["acc"], 4),
                })
    return rows_out


# ── Paraphraser comparison ────────────────────────────────────────────────────

def load_existing_results() -> pd.DataFrame:
    """Load Hard-bucket F1 from existing multiseed results for comparison."""
    p = paths.RESULTS / "eval" / "multiseed_summary_buckets.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    fk = df[
        (df["hardness"] == "readability_fk") &
        (df["bucket"]   == "Hard") &
        (df["metric"]   == "f1")
    ][["detector", "split", "mean", "ci_lo", "ci_hi"]].copy()
    fk["paraphraser"] = "llama_3.1_8b"
    return fk


# ── Figures ──────────────────────────────────────────────────────────────────

def fig_nllb_robustness(metrics_df: pd.DataFrame) -> None:
    splits = NLLB_TRACK
    x_labels = ["P0", "P1\nnllb", "P2\nnllb"]
    colors = {"word_tfidf_lr": "#1f77b4", "char_tfidf_lr": "#d62728"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric, ylabel in zip(axes, ["acc", "f1"], ["Accuracy", "F1"]):
        for det, color in colors.items():
            vals = []
            for sp in splits:
                row = metrics_df[(metrics_df["detector"] == det) &
                                 (metrics_df["split"]    == sp)]
                vals.append(float(row[metric].values[0]) if len(row) else np.nan)
            ax.plot(x_labels, vals, marker="o", color=color,
                    label=det.replace("_", " "), linewidth=2)
        ax.set_title(f"{ylabel} — NLLB back-translation track")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.4, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle("Detector robustness under NLLB back-translation (en->fr->en)\n"
                 "Same Hard-bucket collapse as Llama paraphraser? → see fig15",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig14_nllb_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_paraphraser_comparison(nllb_buckets: pd.DataFrame,
                                llama_buckets: pd.DataFrame) -> None:
    """Hard-bucket F1 at P2 for Llama vs NLLB, per detector."""
    p2_splits = {
        "llama_3.1_8b": "P2_test_simplified",
        "nllb_backtranslation": "P2_test_nllb",
    }
    detectors   = ["word_tfidf_lr", "char_tfidf_lr"]
    paraphrasers = list(p2_splits.keys())
    colors = {"llama_3.1_8b": "#ff7f0e", "nllb_backtranslation": "#2ca02c"}
    labels = {"llama_3.1_8b": "Llama-3.1-8B\n(P2 simplified)",
              "nllb_backtranslation": "NLLB back-translation\n(P2)"}

    fig, axes = plt.subplots(1, len(detectors), figsize=(10, 4), sharey=True)
    for ax, det in zip(axes, detectors):
        x = np.arange(len(paraphrasers))
        for i, para in enumerate(paraphrasers):
            sp = p2_splits[para]
            if para == "llama_3.1_8b":
                row = llama_buckets[
                    (llama_buckets["detector"] == det) &
                    (llama_buckets["split"]    == sp)
                ]
                f1  = float(row["mean"].values[0]) if len(row) else np.nan
                lo  = float(row["ci_lo"].values[0]) if len(row) else np.nan
                hi  = float(row["ci_hi"].values[0]) if len(row) else np.nan
            else:
                row = nllb_buckets[
                    (nllb_buckets["detector"] == det) &
                    (nllb_buckets["split"]    == sp) &
                    (nllb_buckets["bucket"]   == "Hard")
                ]
                f1  = float(row["f1"].values[0]) if len(row) else np.nan
                lo = hi = f1

            ax.bar(i, f1, color=colors[para], alpha=0.8, label=labels[para],
                   width=0.5)
            if not np.isnan(lo) and lo != f1:
                ax.errorbar(i, f1, yerr=[[f1 - lo], [hi - f1]],
                            fmt="none", color="black", capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels([labels[p] for p in paraphrasers], fontsize=8)
        ax.set_title(det.replace("_", " "), fontsize=10)
        ax.set_ylabel("Hard-bucket F1 (readability_fk)")
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.4, linewidth=0.8)
        ax.grid(True, alpha=0.2, axis="y")

    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in paraphrasers]
    fig.legend(handles, [labels[p] for p in paraphrasers],
               loc="upper right", fontsize=9, framealpha=0.9)
    fig.suptitle("Hard-bucket F1 at P2: Llama vs NLLB back-translation\n"
                 "(readability_fk hardness; same Hard samples across paraphrasers)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = paths.FIGURES / "fig15_paraphraser_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading NLLB paraphrase splits...")
    splits_data = load_nllb_splits()

    print("[2/5] Training detectors on fixed train split...")
    p0_all    = load_jsonl(paths.P0_PATH)
    detectors = train_detectors(p0_all)

    print("[3/5] Computing flat metrics on NLLB track...")
    metrics_rows = []
    for split_name, rows in splits_data.items():
        for det_name, det in detectors.items():
            m = evaluate_split(rows, det, det_name, split_name)
            metrics_rows.append(m)
            print(f"  {det_name:<20} {split_name:<20}  "
                  f"acc={m['acc']:.4f}  f1={m['f1']:.4f}  auroc={m['auroc']:.4f}")

    df_flat = pd.DataFrame(metrics_rows)
    df_flat.to_csv(out_dir / "nllb_metrics_flat.csv", index=False)
    print(f"Saved: nllb_metrics_flat.csv")

    print("[4/5] Computing AURC and Hard-bucket F1 (readability_fk)...")
    aurc_rows = compute_aurc(metrics_rows, list(detectors.keys()))
    df_aurc = pd.DataFrame(aurc_rows)
    df_aurc.to_csv(out_dir / "nllb_aurc.csv", index=False)
    print("\nNLLB track AURC:")
    print(df_aurc.to_string(index=False))

    bucket_rows = compute_hardness_buckets(splits_data, detectors)
    df_buckets = pd.DataFrame(bucket_rows)
    df_buckets.to_csv(out_dir / "nllb_hardness_buckets.csv", index=False)

    print("\nHard-bucket F1 (readability_fk) on NLLB track:")
    hard = df_buckets[df_buckets["bucket"] == "Hard"][
        ["detector", "split", "f1"]
    ].sort_values(["detector", "split"])
    print(hard.to_string(index=False))

    print("[5/5] Generating figures...")
    fig_nllb_robustness(df_flat)

    llama_buckets = load_existing_results()
    if not llama_buckets.empty:
        fig_paraphraser_comparison(df_buckets, llama_buckets)
    else:
        print("Skipping fig15: multiseed_summary_buckets.csv not found.")

    # Paraphraser comparison summary
    print("\n" + "=" * 60)
    print("PARAPHRASER COMPARISON — Hard-bucket F1 at P2 (readability_fk)")
    print("=" * 60)
    for det in detectors:
        row_nllb = df_buckets[
            (df_buckets["detector"] == det) &
            (df_buckets["split"]    == "P2_test_nllb") &
            (df_buckets["bucket"]   == "Hard")
        ]
        f1_nllb = float(row_nllb["f1"].values[0]) if len(row_nllb) else float("nan")

        if not llama_buckets.empty:
            row_llama = llama_buckets[
                (llama_buckets["detector"] == det) &
                (llama_buckets["split"]    == "P2_test_simplified")
            ]
            f1_llama = float(row_llama["mean"].values[0]) if len(row_llama) else float("nan")
        else:
            f1_llama = float("nan")

        print(f"  {det:<20}  Llama P2-sim={f1_llama:.3f}  NLLB P2={f1_nllb:.3f}")

    print("\nDone. Next: run src/paraphrase_test_mistral.py for Part 2 of P6.")


if __name__ == "__main__":
    main()
