"""
run_calibration_depth.py
------------------------
P5 — Calibration depth analysis.

What this adds beyond the existing fig06 reliability diagrams:

  1. Temperature scaling (Platt/logit scaling) — fit temperature T on the
     val split; apply to test splits; compare ECE before vs after correction.

  2. Per-bucket ECE — Easy / Medium / Hard (readability_fk) × stage.
     Tests whether calibration degradation is concentrated in the Hard bucket.

  3. Cross-detector ECE summary — combines TF-IDF (from multiseed) and
     RoBERTa (from roberta_raw_flat.csv) into one comparison table.

Outputs (results/eval/)
-----------------------
  calibration_temperature.csv    — optimal T per detector, ECE before/after
  calibration_buckets.csv        — per-bucket ECE per detector per stage
  calibration_summary.csv        — all-detector ECE comparison table

Figures
-------
  figures/fig18_temperature_scaling.png  — ECE before/after per detector
  figures/fig19_bucket_ece.png           — Hard vs Easy ECE across stages
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl, load_test_ids, to_xy
from lib.detectors import build_word_tfidf_lr, build_char_tfidf_lr
from lib.metrics import expected_calibration_error, point_metrics
from lib.hardness import hardness_readability

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 200,
    "font.size": 10, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False,
})

SPLITS_ORDER = [
    "P0_test", "P1_test_standard", "P2_test_standard",
    "P1_test_simplified", "P2_test_simplified",
]
SPLIT_LABELS = {
    "P0_test":            "P0",
    "P1_test_standard":   "P1\nstd",
    "P2_test_standard":   "P2\nstd",
    "P1_test_simplified": "P1\nsim",
    "P2_test_simplified": "P2\nsim",
}


# ── Temperature scaling helpers ───────────────────────────────────────────────

def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    return np.log(p / (1.0 - p))


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Sigmoid(logit(p) / T). T>1 softens; T<1 sharpens."""
    return 1.0 / (1.0 + np.exp(-logit(probs) / T))


def nll(T: float, probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(apply_temperature(probs, T), 1e-7, 1.0 - 1e-7)
    return -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_temperature(probs_val: np.ndarray, y_val: np.ndarray) -> float:
    """Find T in [0.1, 10] minimising NLL on val set."""
    result = minimize_scalar(
        lambda T: nll(T, probs_val, y_val),
        bounds=(0.1, 10.0),
        method="bounded",
    )
    return float(result.x)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_splits() -> dict:
    test_ids  = load_test_ids(paths.TEST_IDS)
    p0_all    = load_jsonl(paths.P0_PATH)
    p0_test   = [r for r in p0_all if r["id"] in test_ids]

    splits = {"P0_test": p0_test}
    for sp in SPLITS_ORDER[1:]:
        p = paths.PARAPHRASE_PATHS.get(sp)
        if p and Path(p).exists():
            splits[sp] = load_jsonl(p)
    return splits


def train_and_get_val_probs() -> tuple:
    """Return (detectors, val_probs_dict)."""
    p0_all    = load_jsonl(paths.P0_PATH)
    train_ids = load_test_ids(paths.TRAIN_IDS)
    val_ids   = load_test_ids(paths.VAL_IDS)

    train_rows = [r for r in p0_all if r["id"] in train_ids]
    val_rows   = [r for r in p0_all if r["id"] in val_ids]

    X_tr, y_tr = to_xy(train_rows)
    X_val, y_val = to_xy(val_rows)

    word_det = build_word_tfidf_lr(X_tr, y_tr)
    char_det = build_char_tfidf_lr(X_tr, y_tr)

    detectors = {"word_tfidf_lr": word_det, "char_tfidf_lr": char_det}
    val_probs = {
        "word_tfidf_lr": word_det.predict_proba(X_val),
        "char_tfidf_lr": char_det.predict_proba(X_val),
    }
    return detectors, val_probs, y_val


# ── Temperature scaling analysis ──────────────────────────────────────────────

def run_temperature_scaling(
    detectors: dict, val_probs: dict, y_val: np.ndarray, splits_data: dict
) -> pd.DataFrame:
    rows = []
    for det_name, det in detectors.items():
        probs_val = val_probs[det_name]
        T_opt = fit_temperature(probs_val, y_val)
        print(f"  {det_name}: optimal T = {T_opt:.4f}")

        for sp in SPLITS_ORDER:
            if sp not in splits_data:
                continue
            X, y = to_xy(splits_data[sp])
            probs_raw  = det.predict_proba(X)
            probs_cal  = apply_temperature(probs_raw, T_opt)
            preds_raw  = (probs_raw >= 0.5).astype(int)
            preds_cal  = (probs_cal >= 0.5).astype(int)

            ece_before = expected_calibration_error(y, probs_raw)
            ece_after  = expected_calibration_error(y, probs_cal)
            acc_before = float(np.mean(preds_raw == y))
            acc_after  = float(np.mean(preds_cal == y))

            rows.append({
                "detector":   det_name,
                "split":      sp,
                "T_opt":      round(T_opt, 4),
                "ece_before": round(ece_before, 4),
                "ece_after":  round(ece_after, 4),
                "ece_delta":  round(ece_after - ece_before, 4),
                "acc_before": round(acc_before, 4),
                "acc_after":  round(acc_after, 4),
            })

    return pd.DataFrame(rows)


# ── Per-bucket ECE ────────────────────────────────────────────────────────────

def run_bucket_ece(detectors: dict, splits_data: dict) -> pd.DataFrame:
    p0_rows  = splits_data["P0_test"]
    p0_texts = [r["text"] for r in p0_rows]
    fk_ha    = hardness_readability(p0_texts)
    id_to_bucket = {r["id"]: b for r, b in zip(p0_rows, fk_ha.buckets)}

    rows = []
    for det_name, det in detectors.items():
        for sp in SPLITS_ORDER:
            if sp not in splits_data:
                continue
            split_rows = splits_data[sp]
            id_to_row  = {r["id"]: r for r in split_rows}
            aligned    = [id_to_row[r["id"]] for r in p0_rows if r["id"] in id_to_row]
            if len(aligned) < len(p0_rows):
                continue

            X, y = to_xy(aligned)
            probs = det.predict_proba(X)

            for bucket in ["Easy", "Medium", "Hard"]:
                mask = np.array([
                    id_to_bucket.get(r["id"], "") == bucket
                    for r in p0_rows if r["id"] in id_to_row
                ])
                if mask.sum() < 3:
                    continue
                ece_b = expected_calibration_error(y[mask], probs[mask])
                preds_b = (probs[mask] >= 0.5).astype(int)
                acc_b = float(np.mean(preds_b == y[mask]))
                rows.append({
                    "detector": det_name, "split": sp,
                    "bucket": bucket, "n": int(mask.sum()),
                    "ece": round(ece_b, 4), "acc": round(acc_b, 4),
                })
    return pd.DataFrame(rows)


# ── RoBERTa ECE summary ───────────────────────────────────────────────────────

def summarise_roberta_ece() -> pd.DataFrame:
    p = paths.RESULTS / "eval" / "roberta_raw_flat.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    ece = df[df["metric"] == "ece"].copy()
    agg = ece.groupby("split")["value"].agg(
        mean="mean", std="std",
        ci_lo=lambda x: float(np.percentile(x, 2.5)),
        ci_hi=lambda x: float(np.percentile(x, 97.5)),
    ).reset_index()
    agg["detector"] = "roberta_base"
    return agg


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_temperature_scaling(df_temp: pd.DataFrame) -> None:
    detectors = df_temp["detector"].unique()
    splits    = [s for s in SPLITS_ORDER if s in df_temp["split"].values]
    xlabels   = [SPLIT_LABELS[s] for s in splits]
    colors    = {"word_tfidf_lr": "#1f77b4", "char_tfidf_lr": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

    # Left: ECE before/after per split per detector
    ax = axes[0]
    x = np.arange(len(splits))
    width = 0.2
    offsets = {"word_tfidf_lr": -0.2, "char_tfidf_lr": 0.2}
    for det in detectors:
        sub = df_temp[df_temp["detector"] == det].set_index("split")
        before = [float(sub.loc[s, "ece_before"]) if s in sub.index else np.nan for s in splits]
        after  = [float(sub.loc[s, "ece_after"])  if s in sub.index else np.nan for s in splits]
        off = offsets[det]
        ax.bar(x + off - 0.1, before, width, color=colors[det], alpha=0.4,
               label=f"{det.replace('_lr','').replace('_',' ')} (raw)")
        ax.bar(x + off + 0.1, after,  width, color=colors[det], alpha=0.9,
               label=f"{det.replace('_lr','').replace('_',' ')} (T-scaled)")

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("ECE (lower = better calibrated)")
    ax.set_title("ECE before/after temperature scaling")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")

    # Right: ECE delta (improvement)
    ax2 = axes[1]
    for det in detectors:
        sub = df_temp[df_temp["detector"] == det].set_index("split")
        deltas = [float(sub.loc[s, "ece_delta"]) if s in sub.index else np.nan for s in splits]
        ax2.plot(xlabels, deltas, marker="o", color=colors[det],
                 label=det.replace("_", " "), linewidth=2)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax2.set_ylabel("ECE delta (after - before); negative = improvement")
    ax2.set_title("Temperature scaling improvement per stage")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # Print optimal T values
    t_word = df_temp[df_temp["detector"] == "word_tfidf_lr"]["T_opt"].iloc[0]
    t_char = df_temp[df_temp["detector"] == "char_tfidf_lr"]["T_opt"].iloc[0]
    fig.suptitle(
        f"Calibration: temperature scaling (T_word={t_word:.3f}, T_char={t_char:.3f})\n"
        "T>1 softens predictions; T<1 sharpens",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = paths.FIGURES / "fig18_temperature_scaling.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def fig_bucket_ece(df_buckets: pd.DataFrame) -> None:
    detectors = ["word_tfidf_lr", "char_tfidf_lr"]
    splits    = [s for s in SPLITS_ORDER if s in df_buckets["split"].values]
    xlabels   = [SPLIT_LABELS[s] for s in splits]
    colors    = {"Easy": "#2ca02c", "Medium": "#ff7f0e", "Hard": "#d62728"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, det in zip(axes, detectors):
        for bucket, color in colors.items():
            sub = df_buckets[
                (df_buckets["detector"] == det) &
                (df_buckets["bucket"]   == bucket)
            ].set_index("split")
            vals = [float(sub.loc[s, "ece"]) if s in sub.index else np.nan for s in splits]
            ax.plot(xlabels, vals, marker="o", color=color,
                    label=bucket, linewidth=2)
        ax.set_title(det.replace("_", " "))
        ax.set_ylabel("ECE (per hardness bucket)")
        ax.set_ylim(0, 0.6)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Per-bucket ECE across paraphrase stages (readability_fk hardness)\n"
        "Does calibration degrade more in Hard samples?",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    out = paths.FIGURES / "fig19_bucket_ece.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading data and training detectors...")
    splits_data = load_all_splits()
    detectors, val_probs, y_val = train_and_get_val_probs()
    print(f"  Val set: {len(y_val)} rows  "
          f"(class balance: {y_val.mean():.2f} LLM)")

    print("\n[2/5] Fitting temperature scaling on val set...")
    df_temp = run_temperature_scaling(detectors, val_probs, y_val, splits_data)
    df_temp.to_csv(out_dir / "calibration_temperature.csv", index=False)
    print("\nTemperature scaling results:")
    print(df_temp[["detector", "split", "T_opt", "ece_before",
                   "ece_after", "ece_delta"]].to_string(index=False))

    print("\n[3/5] Computing per-bucket ECE (readability_fk)...")
    df_buckets = run_bucket_ece(detectors, splits_data)
    df_buckets.to_csv(out_dir / "calibration_buckets.csv", index=False)
    print("\nPer-bucket ECE summary (Hard bucket):")
    hard = df_buckets[df_buckets["bucket"] == "Hard"][
        ["detector", "split", "ece", "acc"]].sort_values(["detector", "split"])
    print(hard.to_string(index=False))

    print("\n[4/5] Summarising RoBERTa ECE...")
    df_roberta = summarise_roberta_ece()
    if not df_roberta.empty:
        print(df_roberta[["split", "mean", "ci_lo", "ci_hi"]].to_string(index=False))

    # Combined summary
    tfidf_ece = pd.read_csv(out_dir / "multiseed_summary_flat.csv")
    tfidf_ece = tfidf_ece[tfidf_ece["metric"] == "ece"][
        ["detector", "split", "mean", "ci_lo", "ci_hi"]].copy()
    if not df_roberta.empty:
        rob_rows = df_roberta[["detector", "split", "mean", "ci_lo", "ci_hi"]]
        df_summary = pd.concat([tfidf_ece, rob_rows], ignore_index=True)
    else:
        df_summary = tfidf_ece
    df_summary.to_csv(out_dir / "calibration_summary.csv", index=False)

    print("\n[5/5] Generating figures...")
    fig_temperature_scaling(df_temp)
    fig_bucket_ece(df_buckets)

    # Key findings printout
    print("\n" + "=" * 60)
    print("P5 KEY FINDINGS")
    print("=" * 60)
    for det in ["word_tfidf_lr", "char_tfidf_lr"]:
        sub = df_temp[df_temp["detector"] == det]
        T = sub["T_opt"].iloc[0]
        p0_before = float(sub[sub["split"] == "P0_test"]["ece_before"].values[0])
        p0_after  = float(sub[sub["split"] == "P0_test"]["ece_after"].values[0])
        p2s_row   = sub[sub["split"] == "P2_test_simplified"]
        p2s_before = float(p2s_row["ece_before"].values[0]) if len(p2s_row) else float("nan")
        p2s_after  = float(p2s_row["ece_after"].values[0])  if len(p2s_row) else float("nan")
        print(f"\n  {det}  (T_opt={T:.3f})")
        print(f"    P0  ECE: {p0_before:.3f} -> {p0_after:.3f}  "
              f"(delta={p0_after-p0_before:+.3f})")
        print(f"    P2s ECE: {p2s_before:.3f} -> {p2s_after:.3f}  "
              f"(delta={p2s_after-p2s_before:+.3f})")

    if not df_roberta.empty:
        print(f"\n  roberta_base (no temperature scaling applied)")
        for _, row in df_roberta.iterrows():
            print(f"    {row['split']:<25} ECE={row['mean']:.3f} "
                  f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]")

    print("\nDone. P5 complete.")


if __name__ == "__main__":
    main()
