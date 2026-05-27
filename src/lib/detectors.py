"""Detector factory. Keeps hyperparameters in one place so all experiments
use identical configurations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


@dataclass
class DetectorBundle:
    name: str
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        Xv = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(Xv)[:, 1]

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)


def build_word_tfidf_lr(
    X_train: list[str],
    y_train: np.ndarray,
    seed: int = 42,
) -> DetectorBundle:
    vec = TfidfVectorizer(
        lowercase=True,
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
    )
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    return DetectorBundle("word_tfidf_lr", vec, clf)


def build_char_tfidf_lr(
    X_train: list[str],
    y_train: np.ndarray,
    seed: int = 42,
) -> DetectorBundle:
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100_000,
        min_df=2,
        lowercase=True,
        sublinear_tf=True,
    )
    Xtr = vec.fit_transform(X_train)
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    clf.fit(Xtr, y_train)
    return DetectorBundle("char_tfidf_lr", vec, clf)


DETECTOR_FACTORY = {
    "word_tfidf_lr": build_word_tfidf_lr,
    "char_tfidf_lr": build_char_tfidf_lr,
}
