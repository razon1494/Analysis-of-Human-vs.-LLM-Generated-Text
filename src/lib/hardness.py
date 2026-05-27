"""Hardness definitions and bucket assignment.

WHY MULTIPLE DEFINITIONS:
The naive "margin from the evaluated detector" definition is circular —
samples near a decision boundary moving across that boundary under
perturbation is mathematically expected, not a property of the data.

A reviewer-proof claim needs to show that the same "Hard" samples
collapse even when Hardness is defined by something the evaluated
detector cannot see. We provide several:

  - margin           : 1 − 2·|p(LLM|x) − 0.5| from the evaluated detector
                       (CIRCULAR; included for reproducing prior work)
  - cross_margin     : margin from a SEPARATE detector (e.g., char vs word)
                       Non-circular for the detector being evaluated.
  - readability      : Flesch-Reading-Ease-style score on the raw text;
                       fully detector-independent.
  - length           : word-count-based hardness (shorter = less signal);
                       fully detector-independent.
  - lexical_diversity: TTR-based; fully detector-independent.

Bucket assignment is fixed from a SCORE VECTOR on a reference split (typically
P0_test). The same bucket label is then transported to all perturbation
splits by ID (preferred) or by position (fallback).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np


WORD_RE = re.compile(r"\b\w+\b")
SENT_RE = re.compile(r"[.!?]+")
VOWEL_GROUPS = re.compile(r"[aeiouAEIOU]+")


@dataclass
class HardnessAssignment:
    score_name: str          # "margin", "entropy", "abs_margin", etc.
    scores: np.ndarray       # one score per sample on the reference split
    buckets: list[str]       # parallel list of "Easy"/"Medium"/"Hard"
    cutoffs: tuple[float, float]   # the q1/q2 thresholds used


def margin_score(probs: np.ndarray) -> np.ndarray:
    """1 − 2·|p − 0.5|. Larger = harder. Range [0, 1]."""
    return 1.0 - 2.0 * np.abs(probs - 0.5)


def abs_margin_score(probs: np.ndarray) -> np.ndarray:
    """|p − 0.5|. Smaller = harder. Range [0, 0.5]."""
    return np.abs(probs - 0.5)


def entropy_score(probs: np.ndarray) -> np.ndarray:
    """Binary entropy H(p) = −p log p − (1−p) log(1−p). Larger = harder."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / math.log(2)


def assign_tertiles(
    scores: np.ndarray,
    hard_is_high: bool = True,
) -> tuple[list[str], tuple[float, float]]:
    """
    Stratify into Easy/Medium/Hard tertiles.

    If hard_is_high=True (default for margin_score, entropy_score), the top
    tertile is "Hard". For abs_margin_score (smaller = harder), pass False.
    """
    q1 = float(np.quantile(scores, 1.0 / 3.0))
    q2 = float(np.quantile(scores, 2.0 / 3.0))
    buckets: list[str] = []
    for s in scores:
        if hard_is_high:
            if s >= q2:
                buckets.append("Hard")
            elif s >= q1:
                buckets.append("Medium")
            else:
                buckets.append("Easy")
        else:
            if s <= q1:
                buckets.append("Hard")
            elif s <= q2:
                buckets.append("Medium")
            else:
                buckets.append("Easy")
    return buckets, (q1, q2)


def hardness_from_probs(
    probs: np.ndarray,
    method: str = "margin",
) -> HardnessAssignment:
    if method == "margin":
        scores = margin_score(probs)
        buckets, cutoffs = assign_tertiles(scores, hard_is_high=True)
    elif method == "abs_margin":
        scores = abs_margin_score(probs)
        buckets, cutoffs = assign_tertiles(scores, hard_is_high=False)
    elif method == "entropy":
        scores = entropy_score(probs)
        buckets, cutoffs = assign_tertiles(scores, hard_is_high=True)
    else:
        raise ValueError(f"unknown hardness method: {method}")
    return HardnessAssignment(
        score_name=method,
        scores=scores,
        buckets=buckets,
        cutoffs=cutoffs,
    )


# ── detector-independent hardness ────────────────────────────────────────────


def _count_syllables(word: str) -> int:
    """Crude syllable count: number of vowel groups, minimum 1."""
    n = len(VOWEL_GROUPS.findall(word))
    return max(1, n)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level. Higher = harder to read."""
    words = WORD_RE.findall(text)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    n_w = len(words)
    n_s = max(1, len(sents))
    if n_w == 0:
        return float("nan")
    n_syl = sum(_count_syllables(w) for w in words)
    return 0.39 * (n_w / n_s) + 11.8 * (n_syl / n_w) - 15.59


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease. Higher = easier (lower hardness)."""
    words = WORD_RE.findall(text)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    n_w = len(words)
    n_s = max(1, len(sents))
    if n_w == 0:
        return float("nan")
    n_syl = sum(_count_syllables(w) for w in words)
    return 206.835 - 1.015 * (n_w / n_s) - 84.6 * (n_syl / n_w)


def lexical_diversity_ttr(text: str) -> float:
    """Type-token ratio. Lower diversity could be a hardness signal
    (less distinctive content) — we treat LOW TTR as HARD."""
    words = WORD_RE.findall(text.lower())
    if not words:
        return float("nan")
    return len(set(words)) / len(words)


def text_length_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def hardness_readability(texts: list[str]) -> HardnessAssignment:
    """Use Flesch-Kincaid grade — HIGHER grade = HARDER reading,
    so we mark high-FK as "Hard" tertile."""
    scores = np.array([flesch_kincaid_grade(t) for t in texts], dtype=float)
    # Handle nan (very short texts) by ranking nan as median
    if np.isnan(scores).any():
        med = np.nanmedian(scores)
        scores = np.where(np.isnan(scores), med, scores)
    buckets, cutoffs = assign_tertiles(scores, hard_is_high=True)
    return HardnessAssignment("readability_fk", scores, buckets, cutoffs)


def hardness_length(texts: list[str]) -> HardnessAssignment:
    """Use word count — SHORTER = LESS SIGNAL = HARDER to classify."""
    scores = np.array([text_length_words(t) for t in texts], dtype=float)
    buckets, cutoffs = assign_tertiles(scores, hard_is_high=False)
    return HardnessAssignment("length", scores, buckets, cutoffs)


def hardness_lexical(texts: list[str]) -> HardnessAssignment:
    """Use TTR — LOW diversity = harder (less distinctive content)."""
    scores = np.array([lexical_diversity_ttr(t) for t in texts], dtype=float)
    if np.isnan(scores).any():
        med = np.nanmedian(scores)
        scores = np.where(np.isnan(scores), med, scores)
    buckets, cutoffs = assign_tertiles(scores, hard_is_high=False)
    return HardnessAssignment("ttr", scores, buckets, cutoffs)


# ── cross-detector hardness ──────────────────────────────────────────────────


def hardness_cross_detector(
    cross_probs: np.ndarray,
    method: str = "margin",
) -> HardnessAssignment:
    """
    Compute hardness using probabilities from a DIFFERENT detector than the
    one being evaluated. Defeats the circularity attack — if a sample is
    Hard *according to detector B* but the question is "does detector A
    catastrophically fail on Hard samples?", there is no boundary-tautology
    argument.

    Pass probabilities from the OTHER detector (e.g., char detector's
    P(LLM|x) when evaluating the word detector).
    """
    h = hardness_from_probs(cross_probs, method=method)
    h.score_name = f"cross_{method}"
    return h


# ── unified factory ──────────────────────────────────────────────────────────


def all_text_based_hardness(texts: list[str]) -> dict[str, HardnessAssignment]:
    """Compute every text-only (detector-independent) hardness in one call."""
    return {
        "readability_fk": hardness_readability(texts),
        "length":         hardness_length(texts),
        "ttr":            hardness_lexical(texts),
    }
