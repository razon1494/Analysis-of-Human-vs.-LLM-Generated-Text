import json
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def to_xy(rows):
    X = [r["text"] for r in rows]
    y = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)  # llm=1, human=0
    return X, y

def eval_one(name, rows, vectorizer, model):
    X, y = to_xy(rows)
    Xv = vectorizer.transform(X)
    pred = model.predict(Xv)

    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    cm = confusion_matrix(y, pred).tolist()

    return {"split": name, "n": int(len(y)), "acc": float(acc), "precision": float(p), "recall": float(r), "f1": float(f1), "confusion_matrix": cm}

def main():
    vectorizer = joblib.load(os.path.join("results", "vectorizer.joblib"))
    clf = joblib.load(os.path.join("results", "model.joblib"))

    # P0_test via IDs
    P0_PATH = os.path.join("data", "p0", "p0.jsonl")
    test_ids = set(Path(os.path.join("data", "splits", "test_ids.txt")).read_text(encoding="utf-8").splitlines())
    p0_test = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    # Standard track (already created earlier)
    p1_std = load_jsonl(os.path.join("data", "p1", "p1_test.jsonl"))
    p2_std = load_jsonl(os.path.join("data", "p2", "p2_test.jsonl"))

    # Simplified track (new)
    p1_sim = load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl"))
    p2_sim = load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl"))

    results = [
        eval_one("P0_test", p0_test, vectorizer, clf),
        eval_one("P1_test_standard", p1_std, vectorizer, clf),
        eval_one("P2_test_standard", p2_std, vectorizer, clf),
        eval_one("P1_test_simplified", p1_sim, vectorizer, clf),
        eval_one("P2_test_simplified", p2_sim, vectorizer, clf),
    ]

    os.makedirs("results", exist_ok=True)
    out_json = os.path.join("results", "robustness_test_dualtrack.json")
    out_csv = os.path.join("results", "robustness_test_dualtrack.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("split,n,acc,precision,recall,f1\n")
        for r in results:
            f.write(f"{r['split']},{r['n']},{r['acc']:.4f},{r['precision']:.4f},{r['recall']:.4f},{r['f1']:.4f}\n")

    print("Dual-track robustness (test):")
    for r in results:
        print(f"{r['split']} | n={r['n']} acc={r['acc']:.4f} p={r['precision']:.4f} r={r['recall']:.4f} f1={r['f1']:.4f}")

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")

if __name__ == "__main__":
    main()
