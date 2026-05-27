import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_xy(rows):
    X = [r["text"] for r in rows]
    y = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)
    return X, y


def eval_one(name, X, y, vectorizer, model):
    Xv = vectorizer.transform(X)
    pred = model.predict(Xv)

    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred).tolist()

    return {
        "split": name,
        "n": int(len(y)),
        "acc": float(acc),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def main():
    VEC_PATH = os.path.join("results", "vectorizer.joblib")
    MODEL_PATH = os.path.join("results", "model.joblib")

    vectorizer = joblib.load(VEC_PATH)
    clf = joblib.load(MODEL_PATH)

    # P0 full file (we will filter by test IDs)
    P0_PATH = os.path.join("data", "p0", "p0.jsonl")
    TEST_IDS_PATH = os.path.join("data", "splits", "test_ids.txt")

    test_ids = set(Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines())

    p0_rows = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    # These are already test-only files (100 rows each)
    P1_TEST = os.path.join("data", "p1", "p1_test.jsonl")
    P2_TEST = os.path.join("data", "p2", "p2_test.jsonl")

    p1_rows = load_jsonl(P1_TEST)
    p2_rows = load_jsonl(P2_TEST)

    results = []
    for name, rows in [("P0_test", p0_rows), ("P1_test", p1_rows), ("P2_test", p2_rows)]:
        X, y = to_xy(rows)
        results.append(eval_one(name, X, y, vectorizer, clf))

    out_json = os.path.join("results", "robustness_test.json")
    out_csv = os.path.join("results", "robustness_test.csv")
    os.makedirs("results", exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("split,n,acc,precision,recall,f1\n")
        for r in results:
            f.write(f"{r['split']},{r['n']},{r['acc']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")

    print("Robustness results (test):")
    for r in results:
        print(f"{r['split']} | n={r['n']} acc={r['acc']:.4f} p={r['precision']:.4f} r={r['recall']:.4f} f1={r['f1']:.4f}")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
