import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def to_xy(rows):
    X = [r["text"] for r in rows]
    y = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)
    ids = [r["id"] for r in rows]
    return ids, X, y

def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return acc, p, r, f1

def assign_tertiles(margins):
    q1 = np.quantile(margins, 1/3)
    q2 = np.quantile(margins, 2/3)
    buckets = []
    for m in margins:
        if m <= q1:
            buckets.append("Hard")
        elif m <= q2:
            buckets.append("Medium")
        else:
            buckets.append("Easy")
    return buckets

def main():
    vec = joblib.load(os.path.join("results", "vectorizer.joblib"))
    clf = joblib.load(os.path.join("results", "model.joblib"))

    # --- Load P0_test from IDs ---
    p0_path = os.path.join("data", "p0", "p0.jsonl")
    test_ids = set(Path(os.path.join("data", "splits", "test_ids.txt")).read_text(encoding="utf-8").splitlines())
    p0_all = load_jsonl(p0_path)
    p0_test_rows = [r for r in p0_all if r["id"] in test_ids]

    # --- Load other test sets ---
    splits = {
        "P0_test": p0_test_rows,
        "P1_test_standard": load_jsonl(os.path.join("data", "p1", "p1_test.jsonl")),
        "P2_test_standard": load_jsonl(os.path.join("data", "p2", "p2_test.jsonl")),
        "P1_test_simplified": load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        "P2_test_simplified": load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    }

    # --- Compute hardness buckets using P0 margins ---
    p0_ids, p0_X, p0_y = to_xy(p0_test_rows)
    p0_Xv = vec.transform(p0_X)
    p0_probs = clf.predict_proba(p0_Xv)[:, 1]  # prob of LLM
    p0_margins = np.abs(p0_probs - 0.5)
    p0_bucket = assign_tertiles(p0_margins)

    bucket_map = {pid: b for pid, b in zip(p0_ids, p0_bucket)}

    # --- Evaluate each split by bucket ---
    records = []
    for split_name, rows in splits.items():
        ids, X, y = to_xy(rows)
        Xv = vec.transform(X)
        pred = clf.predict(Xv)

        for b in ["Easy", "Medium", "Hard"]:
            idx = [i for i, pid in enumerate(ids) if bucket_map.get(pid) == b]
            if not idx:
                continue
            acc, p, r, f1 = metrics(y[idx], pred[idx])
            records.append({
                "split": split_name,
                "bucket": b,
                "n": len(idx),
                "acc": round(acc, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            })

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/hardness_buckets_test.csv", index=False)

    with open("results/hardness_buckets_test.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(df)

    # quick sanity: bucket sizes in P0
    p0_counts = pd.Series(p0_bucket).value_counts()
    print("\nBucket sizes (P0_test):")
    print(p0_counts)

if __name__ == "__main__":
    main()
