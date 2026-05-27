"""Metrics + uncertainty quantification.

Includes:
  - point metrics (acc/P/R/F1/AUROC/MCC/Brier/ECE)
  - paired-bootstrap CI on metric DIFFERENCES (the proper way to
    report degradation Δ = metric(P_k) − metric(P_0))
  - independent bootstrap CI for stand-alone metrics
  - Area Under the Robustness Curve (AURC) summarising total
    degradation across paraphrase stages as a single scalar
  - per-bucket evaluation helper

References
----------
- Efron & Tibshirani (1993) — paired bootstrap for differences
- Naeini et al. (AAAI 2015) — expected calibration error
- Snoek et al. (2019) — reliability diagrams
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


# ── point metrics ─────────────────────────────────────────────────────────────


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """All metrics in one dict. NaN-safe on degenerate inputs (single class)."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else float("nan")
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)
    return dict(
        acc=float(acc),
        precision=float(p),
        recall=float(r),
        f1=float(f1),
        auroc=float(auc),
        mcc=float(mcc),
        brier=float(brier),
        ece=float(ece),
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average gap between mean predicted probability and observed accuracy
    in each confidence bin. Standard binary-classification definition."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = (y_true[mask] == (y_prob[mask] >= 0.5).astype(int)).mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


# ── bootstrap CIs ─────────────────────────────────────────────────────────────


@dataclass
class CI:
    point: float
    lo: float
    hi: float

    def fmt(self, places: int = 4) -> str:
        return f"{self.point:.{places}f} [{self.lo:.{places}f}, {self.hi:.{places}f}]"


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CI:
    """Percentile bootstrap CI for a single metric on a single split."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred, y_prob)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            samples[i] = metric_fn(y_true[idx], y_pred[idx], y_prob[idx])
        except ValueError:
            samples[i] = np.nan
    lo, hi = np.nanpercentile(samples, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return CI(point=float(point), lo=float(lo), hi=float(hi))


def paired_bootstrap_diff_ci(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_prob_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> CI:
    """
    Paired bootstrap on the DIFFERENCE metric_fn(B) − metric_fn(A) where A and B
    are evaluated on the SAME paired samples (e.g., P0 vs P1 for the same docs).
    Resamples the shared sample index, then computes both metrics on the same
    resample. CI is on Δ = B − A.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point_a = metric_fn(y_true, y_pred_a, y_prob_a)
    point_b = metric_fn(y_true, y_pred_b, y_prob_b)
    point = point_b - point_a

    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        try:
            ma = metric_fn(yt, y_pred_a[idx], y_prob_a[idx])
            mb = metric_fn(yt, y_pred_b[idx], y_prob_b[idx])
            diffs[i] = mb - ma
        except ValueError:
            diffs[i] = np.nan
    lo, hi = np.nanpercentile(diffs, [alpha / 2 * 100, (1 - alpha / 2) * 100])
    return CI(point=float(point), lo=float(lo), hi=float(hi))


# ── metric extractors (compatible with bootstrap_metric_ci signature) ────────


def _acc(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    return float(accuracy_score(yt, yp))


def _f1(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    return float(precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)[2])


def _auroc(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(yt, yprob))
    except ValueError:
        return float("nan")


def _mcc(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    try:
        return float(matthews_corrcoef(yt, yp))
    except ValueError:
        return float("nan")


def _brier(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    return float(brier_score_loss(yt, yprob))


def _ece(yt: np.ndarray, yp: np.ndarray, yprob: np.ndarray) -> float:
    return float(expected_calibration_error(yt, yprob, n_bins=10))


METRIC_FUNCS = {
    "acc":   _acc,
    "f1":    _f1,
    "auroc": _auroc,
    "mcc":   _mcc,
    "brier": _brier,
    "ece":   _ece,
}


# ── AURC: area under the degradation curve ───────────────────────────────────


def area_under_robustness_curve(
    stage_values: list[float],
    stage_x: list[float] | None = None,
) -> float:
    """
    AURC = trapezoidal area under metric vs paraphrase-stage curve, normalised
    so AURC=1 means metric stayed at 1.0 through all stages (perfect robustness)
    and AURC=0 means metric was 0 throughout.

    If stage_x is None, uses [0, 1, 2, ...] (equally spaced).
    """
    arr = np.array(stage_values, dtype=float)
    if stage_x is None:
        x = np.arange(len(arr), dtype=float)
    else:
        x = np.array(stage_x, dtype=float)
    if len(arr) < 2:
        return float(arr[0]) if len(arr) else float("nan")
    # Trapezoidal area / span — equals mean of midpoint values for equal spacing.
    span = x[-1] - x[0]
    if span <= 0:
        return float(arr.mean())
    area = np.trapezoid(arr, x)
    return float(area / span)


def relative_degradation_slope(
    stage_values: list[float],
    stage_x: list[float] | None = None,
) -> float:
    """Least-squares slope of metric vs stage. Negative = degradation."""
    arr = np.array(stage_values, dtype=float)
    if stage_x is None:
        x = np.arange(len(arr), dtype=float)
    else:
        x = np.array(stage_x, dtype=float)
    if len(arr) < 2 or np.all(np.isnan(arr)):
        return float("nan")
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(x[mask], arr[mask], deg=1)[0]
    return float(slope)
