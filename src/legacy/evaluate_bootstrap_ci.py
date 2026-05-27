"""
evaluate_bootstrap_ci.py
------------------------
Computes bootstrap confidence intervals (95%) for all metrics
(Accuracy, Precision, Recall, F1, AUROC, MCC) across every
evaluation condition:

    P0_test | P1_test_standard | P2_test_standard
            | P1_test_simplified | P2_test_simplified

Also runs McNemar's test between every adjacent pair
(P0 vs P1, P1 vs P2) within each track, to check whether
performance drops are statistically significant.

Outputs
-------
    results/bootstrap_ci.csv   -- wide table, one row per condition
    results/bootstrap_ci.json  -- full detail including all CI bounds
    results/mcnemar_tests.csv  -- pairwise McNemar results

Usage
-----
    python src/evaluate_bootstrap_ci.py

Requirements
------------
    pip install scikit-learn joblib numpy pandas statsmodels
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
)
from statsmodels.stats.contingency_tables import mcnemar

# ── reproducibility ──────────────────────────────────────────────────────────
RNG_SEED = 42
N_BOOTSTRAP = 2000          # 2 000 resamples → stable 95% CI
CI_ALPHA    = 0.05          # two-sided 95%

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR   = os.path.join("results")
VEC_PATH      = os.path.join(RESULTS_DIR, "vectorizer.joblib")
MODEL_PATH    = os.path.join(RESULTS_DIR, "model.joblib")
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


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    """Return dict of point-estimate metrics."""
    acc  = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    mcc = matthews_corrcoef(y_true, y_pred)
    return dict(acc=acc, precision=p, recall=r, f1=f1, auroc=auc, mcc=mcc)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    seed: int = RNG_SEED,
):
    """
    Percentile bootstrap CI for all metrics.
    Returns dict: metric -> (lower, upper)
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    boot_acc  = np.empty(n_boot)
    boot_prec = np.empty(n_boot)
    boot_rec  = np.empty(n_boot)
    boot_f1   = np.empty(n_boot)
    boot_auc  = np.empty(n_boot)
    boot_mcc  = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)          # sample WITH replacement
        yt, yp, ypr = y_true[idx], y_pred[idx], y_prob[idx]

        boot_acc[i]  = accuracy_score(yt, yp)
        pr, rc, f, _ = precision_recall_fscore_support(
            yt, yp, average="binary", zero_division=0
        )
        boot_prec[i] = pr
        boot_rec[i]  = rc
        boot_f1[i]   = f
        try:
            boot_auc[i] = roc_auc_score(yt, ypr)
        except ValueError:
            boot_auc[i] = float("nan")
        boot_mcc[i] = matthews_corrcoef(yt, yp)

    lo, hi = alpha / 2, 1 - alpha / 2

    def ci(arr):
        return float(np.nanpercentile(arr, lo * 100)), float(np.nanpercentile(arr, hi * 100))

    return {
        "acc":       ci(boot_acc),
        "precision": ci(boot_prec),
        "recall":    ci(boot_rec),
        "f1":        ci(boot_f1),
        "auroc":     ci(boot_auc),
        "mcc":       ci(boot_mcc),
    }


def mcnemar_test(pred_a: np.ndarray, pred_b: np.ndarray, y_true: np.ndarray):
    """
    McNemar's test comparing two sets of binary predictions.
    Returns chi2 statistic and p-value.
    """
    # Contingency table: rows = pred_a correct/wrong, cols = pred_b correct/wrong
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    # b = A correct, B wrong; c = A wrong, B correct
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    table = np.array([[0, b], [c, 0]])   # only off-diagonals matter
    result = mcnemar(table, exact=False, correction=True)
    return float(result.statistic), float(result.pvalue)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load model
    vectorizer = joblib.load(VEC_PATH)
    clf        = joblib.load(MODEL_PATH)

    # Load all test splits
    test_ids = set(
        Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines()
    )
    p0_rows = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    splits_def = [
        ("P0_test",            p0_rows),
        ("P1_test_standard",   load_jsonl(os.path.join("data", "p1", "p1_test.jsonl"))),
        ("P2_test_standard",   load_jsonl(os.path.join("data", "p2", "p2_test.jsonl"))),
        ("P1_test_simplified", load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl"))),
        ("P2_test_simplified", load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl"))),
    ]

    # ── compute predictions for every split ───────────────────────────────────
    split_data = {}   # name -> (y_true, y_pred, y_prob)
    for name, rows in splits_def:
        X, y = to_xy(rows)
        Xv   = vectorizer.transform(X)
        pred = clf.predict(Xv)
        prob = clf.predict_proba(Xv)[:, 1]
        split_data[name] = (y, pred, prob)

    # ── point estimates + bootstrap CIs ───────────────────────────────────────
    records     = []
    full_detail = []

    print(f"\nBootstrap CI  (n_boot={N_BOOTSTRAP}, alpha={CI_ALPHA})\n")
    print(f"{'Split':<25} {'Acc':>6}  {'95% CI':^15}  {'F1':>6}  {'95% CI':^15}  {'AUROC':>6}  {'MCC':>6}")
    print("-" * 90)

    for name, (y_true, y_pred, y_prob) in split_data.items():
        pt  = point_metrics(y_true, y_pred, y_prob)
        cis = bootstrap_ci(y_true, y_pred, y_prob)

        print(
            f"{name:<25} "
            f"{pt['acc']:.4f}  [{cis['acc'][0]:.4f}, {cis['acc'][1]:.4f}]  "
            f"{pt['f1']:.4f}  [{cis['f1'][0]:.4f}, {cis['f1'][1]:.4f}]  "
            f"{pt['auroc']:.4f}  {pt['mcc']:.4f}"
        )

        row = {"split": name, "n": int(len(y_true))}
        for metric in ("acc", "precision", "recall", "f1", "auroc", "mcc"):
            row[metric]          = round(pt[metric],       4)
            row[f"{metric}_ci_lo"] = round(cis[metric][0], 4)
            row[f"{metric}_ci_hi"] = round(cis[metric][1], 4)
        records.append(row)

        full_detail.append({
            "split":         name,
            "n":             int(len(y_true)),
            "point_estimates": {k: round(v, 6) for k, v in pt.items()},
            "ci_95": {k: [round(v, 6) for v in cis[k]] for k in cis},
        })

    # ── McNemar tests ─────────────────────────────────────────────────────────
    mcnemar_pairs = [
        # Standard track
        ("P0_test",          "P1_test_standard",   "std: P0 vs P1"),
        ("P1_test_standard", "P2_test_standard",   "std: P1 vs P2"),
        ("P0_test",          "P2_test_standard",   "std: P0 vs P2"),
        # Simplified track
        ("P0_test",            "P1_test_simplified", "sim: P0 vs P1"),
        ("P1_test_simplified", "P2_test_simplified", "sim: P1 vs P2"),
        ("P0_test",            "P2_test_simplified", "sim: P0 vs P2"),
        # Cross-track at P1 and P2
        ("P1_test_standard",   "P1_test_simplified", "P1: std vs sim"),
        ("P2_test_standard",   "P2_test_simplified", "P2: std vs sim"),
    ]

    mcnemar_records = []
    print("\nMcNemar's tests (H0: no difference in error rates)\n")
    print(f"{'Comparison':<30} {'chi2':>8}  {'p-value':>10}  {'significant':>12}")
    print("-" * 65)

    # NOTE: McNemar requires matching sample indices.
    # All splits share the same P0 test IDs, so y_true is identical.
    # We compare predictions only — y_true is the same object.
    y_true_ref = split_data["P0_test"][0]   # all splits share same y_true

    for name_a, name_b, label in mcnemar_pairs:
        _, pred_a, _ = split_data[name_a]
        _, pred_b, _ = split_data[name_b]
        chi2, pval   = mcnemar_test(pred_a, pred_b, y_true_ref)
        sig          = "YES *" if pval < 0.05 else "no"
        print(f"{label:<30} {chi2:>8.4f}  {pval:>10.4f}  {sig:>12}")
        mcnemar_records.append({
            "comparison": label,
            "split_a":    name_a,
            "split_b":    name_b,
            "chi2":       round(chi2, 6),
            "pvalue":     round(pval, 6),
            "significant_p05": pval < 0.05,
        })

    # ── save outputs ──────────────────────────────────────────────────────────
    df_ci = pd.DataFrame(records)
    ci_csv  = os.path.join(RESULTS_DIR, "bootstrap_ci.csv")
    ci_json = os.path.join(RESULTS_DIR, "bootstrap_ci.json")
    df_ci.to_csv(ci_csv, index=False)
    with open(ci_json, "w", encoding="utf-8") as f:
        json.dump(full_detail, f, indent=2)

    df_mc = pd.DataFrame(mcnemar_records)
    mc_csv = os.path.join(RESULTS_DIR, "mcnemar_tests.csv")
    df_mc.to_csv(mc_csv, index=False)

    print(f"\nSaved: {ci_csv}")
    print(f"Saved: {ci_json}")
    print(f"Saved: {mc_csv}")


if __name__ == "__main__":
    main()
