import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def load_ids(path: str):
    return set(Path(path).read_text(encoding="utf-8").splitlines())


def load_split_jsonl(p0_path: str, ids_set: set):
    X, y = [], []
    with open(p0_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["id"] not in ids_set:
                continue
            X.append(obj["text"])
            y.append(1 if obj["label"] == "llm" else 0)  # llm=1, human=0
    return X, np.array(y, dtype=int)


def main():
    P0_PATH = os.path.join("data", "p0", "p0.jsonl")
    SPLIT_DIR = os.path.join("data", "splits")
    OUT_DIR = os.path.join("results")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_ids = load_ids(os.path.join(SPLIT_DIR, "train_ids.txt"))
    val_ids = load_ids(os.path.join(SPLIT_DIR, "val_ids.txt"))
    test_ids = load_ids(os.path.join(SPLIT_DIR, "test_ids.txt"))

    X_train, y_train = load_split_jsonl(P0_PATH, train_ids)
    X_val, y_val = load_split_jsonl(P0_PATH, val_ids)
    X_test, y_test = load_split_jsonl(P0_PATH, test_ids)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # TF-IDF baseline (strong, interpretable)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=2,
    )

    Xtr = vectorizer.fit_transform(X_train)
    Xva = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    # Logistic regression (well-behaved baseline)
    clf = LogisticRegression(
        max_iter=2000,
        n_jobs=1,
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
    )

    clf.fit(Xtr, y_train)

    # Evaluate
    def eval_split(name, X, y):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        cm = confusion_matrix(y, pred).tolist()
        print(f"{name} | acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f}")
        return {"acc": acc, "precision": p, "recall": r, "f1": f1, "confusion_matrix": cm}

    metrics = {
        "val": eval_split("VAL", Xva, y_val),
        "test": eval_split("TEST", Xte, y_test),
        "label_map": {"human": 0, "llm": 1},
        "model": "TFIDF(1-2gram)+LogReg",
        "dataset": "P0",
    }

    # Save artifacts
    joblib.dump(vectorizer, os.path.join(OUT_DIR, "vectorizer.joblib"))
    joblib.dump(clf, os.path.join(OUT_DIR, "model.joblib"))

    with open(os.path.join(OUT_DIR, "metrics_p0.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {os.path.join(OUT_DIR, 'vectorizer.joblib')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'model.joblib')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'metrics_p0.json')}")


if __name__ == "__main__":
    main()
