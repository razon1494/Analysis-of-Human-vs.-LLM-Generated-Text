"""
train_char_ngram_detector.py
-----------------------------
Trains a second detector using CHARACTER n-gram TF-IDF (3–5 grams)
+ Logistic Regression, as a comparison baseline against the word
n-gram detector in train_detector.py.

Character n-grams are known to be more robust to vocabulary-level
paraphrasing because they capture subword morphological patterns
and writing style rather than exact token identity.

Outputs
-------
    results/vectorizer_char.joblib   -- fitted char TF-IDF vectorizer
    results/model_char.joblib        -- fitted logistic regression model
    results/metrics_char_p0.json     -- val + test metrics on P0

Usage
-----
    python src/train_char_ngram_detector.py

Requirements
------------
    pip install scikit-learn joblib numpy
"""

import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)

# ── paths ─────────────────────────────────────────────────────────────────────
P0_PATH   = os.path.join("data", "p0",     "p0.jsonl")
SPLIT_DIR = os.path.join("data", "splits")
OUT_DIR   = os.path.join("results")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ids(path: str):
    return set(Path(path).read_text(encoding="utf-8").splitlines())


def load_split_jsonl(path: str, ids_set: set):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] not in ids_set:
                continue
            X.append(obj["text"])
            y.append(1 if obj["label"] == "llm" else 0)
    return X, np.array(y, dtype=int)


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def to_xy(rows):
    X = [r["text"] for r in rows]
    y = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)
    return X, y


def eval_split(name: str, X_vec, y: np.ndarray, clf) -> dict:
    pred = clf.predict(X_vec)
    prob = clf.predict_proba(X_vec)[:, 1]

    acc         = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc         = roc_auc_score(y, prob)
    mcc         = matthews_corrcoef(y, pred)
    cm          = confusion_matrix(y, pred).tolist()

    print(
        f"  {name:<25} acc={acc:.4f}  p={p:.4f}  r={r:.4f}  "
        f"f1={f1:.4f}  auroc={auc:.4f}  mcc={mcc:.4f}"
    )
    return {
        "split": name, "n": int(len(y)),
        "acc": float(acc), "precision": float(p), "recall": float(r),
        "f1": float(f1), "auroc": float(auc), "mcc": float(mcc),
        "confusion_matrix": cm,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── load P0 splits ────────────────────────────────────────────────────────
    train_ids = load_ids(os.path.join(SPLIT_DIR, "train_ids.txt"))
    val_ids   = load_ids(os.path.join(SPLIT_DIR, "val_ids.txt"))
    test_ids  = load_ids(os.path.join(SPLIT_DIR, "test_ids.txt"))

    X_train, y_train = load_split_jsonl(P0_PATH, train_ids)
    X_val,   y_val   = load_split_jsonl(P0_PATH, val_ids)
    X_test,  y_test  = load_split_jsonl(P0_PATH, test_ids)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # ── char n-gram TF-IDF ────────────────────────────────────────────────────
    # analyzer="char_wb" respects word boundaries (more informative than "char")
    # ngram_range=(3,5): trigrams through 5-grams capture morphology + style
    # max_features=100_000: char vocab is larger than word vocab
    print("\nFitting char n-gram TF-IDF (3-5 grams, char_wb) …")
    vec_char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=100_000,
        min_df=2,
        lowercase=True,
        sublinear_tf=True,      # log-scale TF dampens very frequent n-grams
    )

    Xtr = vec_char.fit_transform(X_train)
    Xva = vec_char.transform(X_val)
    Xte = vec_char.transform(X_test)

    print(f"Vocab size: {len(vec_char.vocabulary_):,}")

    # ── logistic regression ───────────────────────────────────────────────────
    clf_char = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    clf_char.fit(Xtr, y_train)

    # ── evaluate on P0 splits ─────────────────────────────────────────────────
    print("\nChar n-gram detector — P0 splits:")
    metrics = {
        "val":  eval_split("P0_val",  Xva, y_val,  clf_char),
        "test": eval_split("P0_test", Xte, y_test, clf_char),
        "model":    "CharTFIDF(3-5gram,char_wb)+LogReg",
        "dataset":  "P0",
        "label_map": {"human": 0, "llm": 1},
    }

    # ── robustness evaluation on paraphrase tracks ────────────────────────────
    paraphrase_splits = [
        ("P1_test_standard",   os.path.join("data", "p1", "p1_test.jsonl")),
        ("P2_test_standard",   os.path.join("data", "p2", "p2_test.jsonl")),
        ("P1_test_simplified", os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        ("P2_test_simplified", os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    ]

    robustness_results = []

    print("\nChar n-gram detector — Robustness under paraphrasing:")
    for split_name, path in paraphrase_splits:
        rows   = load_jsonl(path)
        X, y   = to_xy(rows)
        Xv     = vec_char.transform(X)
        result = eval_split(split_name, Xv, y, clf_char)
        robustness_results.append(result)

    # ── save artifacts ────────────────────────────────────────────────────────
    joblib.dump(vec_char,  os.path.join(OUT_DIR, "vectorizer_char.joblib"))
    joblib.dump(clf_char,  os.path.join(OUT_DIR, "model_char.joblib"))

    with open(os.path.join(OUT_DIR, "metrics_char_p0.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    rob_csv = os.path.join(OUT_DIR, "robustness_char.csv")
    with open(rob_csv, "w", encoding="utf-8") as f:
        f.write("split,n,acc,precision,recall,f1,auroc,mcc\n")
        # include P0_test baseline row
        t = metrics["test"]
        f.write(
            f"P0_test,{t['n']},{t['acc']:.4f},{t['precision']:.4f},"
            f"{t['recall']:.4f},{t['f1']:.4f},{t['auroc']:.4f},{t['mcc']:.4f}\n"
        )
        for r in robustness_results:
            f.write(
                f"{r['split']},{r['n']},{r['acc']:.4f},{r['precision']:.4f},"
                f"{r['recall']:.4f},{r['f1']:.4f},{r['auroc']:.4f},{r['mcc']:.4f}\n"
            )

    print(f"\nSaved: {os.path.join(OUT_DIR, 'vectorizer_char.joblib')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'model_char.joblib')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'metrics_char_p0.json')}")
    print(f"Saved: {rob_csv}")

    # ── comparison summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY  (char n-gram vs word n-gram)")
    print("=" * 70)
    print(
        "Load results/robustness_test_dualtrack.csv and results/robustness_char.csv\n"
        "and compare columns side-by-side to see which detector is more robust\n"
        "under paraphrasing. Use plot_compare_detectors.py to visualise."
    )


if __name__ == "__main__":
    main()
