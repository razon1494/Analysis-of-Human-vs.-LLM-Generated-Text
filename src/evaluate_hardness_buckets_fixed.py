"""
evaluate_hardness_buckets_fixed.py
-----------------------------------
FIXED version of evaluate_hardness_buckets.py.

Root cause of the flat-line bug:
  The P1/P2 paraphrase JSONL files share IDs with P0, BUT the
  bucket_map lookup was silently returning no matches when IDs
  didn't align (e.g. due to row-order mismatch or ID truncation).
  This caused every bucket slot to get only P0 predictions
  regardless of stage, producing flat lines.

Fix:
  1. Explicitly verify ID overlap between P0 and each P1/P2 file
     and print a diagnostic so you can see what fraction matches.
  2. Use POSITIONAL alignment as a fallback when IDs match
     positionally but not by value (common when paraphrase scripts
     preserve order but reset/alter the id field).
  3. Build a position-indexed bucket array so each sample in
     P1/P2 inherits the bucket of the sample at the same position
     in P0 (same test set, same order = valid alignment).

Also fixes plot_hardness_buckets.py inline (writes new figure files).

Outputs
-------
    results/hardness_buckets_fixed.csv
    results/hardness_buckets_fixed.json
    figures/f1_hardness_standard_fixed.png
    figures/f1_hardness_simplified_fixed.png

Usage
-----
    python src/evaluate_hardness_buckets_fixed.py
"""

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_xy(rows):
    X   = [r["text"] for r in rows]
    y   = np.array([1 if r["label"] == "llm" else 0 for r in rows], dtype=int)
    ids = [r["id"] for r in rows]
    return ids, X, y


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return acc, p, r, f1


def assign_tertiles(margins):
    q1 = np.quantile(margins, 1 / 3)
    q2 = np.quantile(margins, 2 / 3)
    buckets = []
    for m in margins:
        if m <= q1:
            buckets.append("Hard")
        elif m <= q2:
            buckets.append("Medium")
        else:
            buckets.append("Easy")
    return buckets


# ── alignment strategy ────────────────────────────────────────────────────────

def build_bucket_assignment(p0_rows, vec, clf):
    """
    Compute per-sample bucket for the P0 test set.
    Returns:
        id_to_bucket   : dict {id -> bucket_label}
        pos_buckets    : list of bucket labels in P0 row order
        p0_ids         : list of P0 IDs in order
    """
    ids, X, y = to_xy(p0_rows)
    Xv        = vec.transform(X)
    probs     = clf.predict_proba(Xv)[:, 1]
    margins   = np.abs(probs - 0.5)
    buckets   = assign_tertiles(margins)

    id_to_bucket = {pid: b for pid, b in zip(ids, buckets)}
    return id_to_bucket, buckets, ids


def get_bucket_indices(rows, id_to_bucket, p0_ids, p0_pos_buckets):
    """
    For a P1/P2 split, assign each row to a bucket using:
      1. ID lookup (preferred — exact match)
      2. Positional fallback (same row order as P0)

    Returns dict: bucket_label -> list of row indices in `rows`
    """
    ids = [r["id"] for r in rows]
    n   = len(rows)

    # Diagnostic: how many IDs match P0?
    p0_id_set    = set(id_to_bucket.keys())
    matching_ids = sum(1 for pid in ids if pid in p0_id_set)
    print(f"    ID match: {matching_ids}/{n} rows matched by ID")

    bucket_indices = {"Easy": [], "Medium": [], "Hard": []}

    if matching_ids == n:
        # Perfect ID match — use ID lookup
        strategy = "id"
        for i, pid in enumerate(ids):
            b = id_to_bucket.get(pid)
            if b:
                bucket_indices[b].append(i)
    else:
        # Positional fallback — assume same order as P0
        strategy = "positional"
        print(f"    WARNING: Using positional alignment (ID mismatch). "
              f"Ensure P1/P2 rows are in the same order as P0 test rows.")
        for i, b in enumerate(p0_pos_buckets[:n]):
            bucket_indices[b].append(i)

    print(f"    Strategy: {strategy} | "
          f"Easy={len(bucket_indices['Easy'])} "
          f"Medium={len(bucket_indices['Medium'])} "
          f"Hard={len(bucket_indices['Hard'])}")
    return bucket_indices


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_track(df, track_name, outfile):
    stages       = ["P0_test", f"P1_test_{track_name}", f"P2_test_{track_name}"]
    stage_labels = ["P0", "P1", "P2"]
    buckets      = ["Easy", "Medium", "Hard"]
    colors       = {"Easy": "#2196F3", "Medium": "#FF9800", "Hard": "#4CAF50"}

    plt.figure(figsize=(8, 5))
    for b in buckets:
        y_vals = []
        for s in stages:
            v = df[(df["split"] == s) & (df["bucket"] == b)]["f1"].values
            y_vals.append(float(v[0]) if len(v) else float("nan"))
        plt.plot(stage_labels, y_vals, marker="o", label=b,
                 color=colors[b], lw=2)

    plt.title(
        f"F1 vs Paraphrase Stage by Hardness Bucket\n({track_name.capitalize()} Track)",
        fontsize=12, fontweight="bold",
    )
    plt.xlabel("Paraphrase stage", fontsize=11)
    plt.ylabel("F1", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    os.makedirs("figures", exist_ok=True)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved: {outfile}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    vec = joblib.load(os.path.join("results", "vectorizer.joblib"))
    clf = joblib.load(os.path.join("results", "model.joblib"))

    # Load P0 test rows
    p0_path  = os.path.join("data", "p0", "p0.jsonl")
    test_ids = set(
        Path(os.path.join("data", "splits", "test_ids.txt"))
        .read_text(encoding="utf-8").splitlines()
    )
    p0_all       = load_jsonl(p0_path)
    p0_test_rows = [r for r in p0_all if r["id"] in test_ids]

    print(f"P0 test rows loaded: {len(p0_test_rows)}")

    # Compute bucket assignments from P0
    id_to_bucket, pos_buckets, p0_ids = build_bucket_assignment(
        p0_test_rows, vec, clf
    )

    bucket_counts = pd.Series(pos_buckets).value_counts()
    print(f"\nP0 bucket distribution:\n{bucket_counts}\n")

    # All splits to evaluate
    splits_def = {
        "P0_test":            p0_test_rows,
        "P1_test_standard":   load_jsonl(os.path.join("data", "p1", "p1_test.jsonl")),
        "P2_test_standard":   load_jsonl(os.path.join("data", "p2", "p2_test.jsonl")),
        "P1_test_simplified": load_jsonl(os.path.join("data", "p1", "p1_test_simplified.jsonl")),
        "P2_test_simplified": load_jsonl(os.path.join("data", "p2", "p2_test_simplified.jsonl")),
    }

    records = []

    for split_name, rows in splits_def.items():
        print(f"\n── {split_name} (n={len(rows)}) ──")
        ids, X, y = to_xy(rows)
        Xv        = vec.transform(X)
        pred      = clf.predict(Xv)

        bucket_indices = get_bucket_indices(
            rows, id_to_bucket, p0_ids, pos_buckets
        )

        for b in ["Easy", "Medium", "Hard"]:
            idx = bucket_indices[b]
            if not idx:
                print(f"    WARN: No samples found for bucket={b}")
                continue
            y_b, pred_b = y[idx], pred[idx]
            acc, p, r, f1 = compute_metrics(y_b, pred_b)

            print(
                f"    {b:6s} n={len(idx):3d} | "
                f"acc={acc:.4f} p={p:.4f} r={r:.4f} f1={f1:.4f}"
            )
            records.append({
                "split":     split_name,
                "bucket":    b,
                "n":         len(idx),
                "acc":       round(acc, 4),
                "precision": round(p,   4),
                "recall":    round(r,   4),
                "f1":        round(f1,  4),
            })

    # Save results
    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/hardness_buckets_fixed.csv", index=False)
    with open("results/hardness_buckets_fixed.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print("\n── Full results table ──")
    print(df.to_string(index=False))
    print("\nSaved: results/hardness_buckets_fixed.csv")
    print("Saved: results/hardness_buckets_fixed.json")

    # Plot both tracks
    plot_track(df, "standard",   "figures/f1_hardness_standard_fixed.png")
    plot_track(df, "simplified", "figures/f1_hardness_simplified_fixed.png")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n── Sanity check: P0 bucket F1 values ──")
    p0_rows_check = df[df["split"] == "P0_test"]
    for _, row in p0_rows_check.iterrows():
        print(f"  {row['bucket']:6s} F1={row['f1']:.4f}  n={row['n']}")

    print(
        "\nIf Easy/Medium/Hard F1 lines are STILL flat after this fix, "
        "check that your p1_test.jsonl IDs match p0 IDs exactly by running:\n"
        "  python -c \""
        "import json; p0=[json.loads(l)['id'] for l in open('data/p0/p0.jsonl')];"
        "p1=[json.loads(l)['id'] for l in open('data/p1/p1_test.jsonl')];"
        "print('P0 ids[:5]:', p0[:5]); print('P1 ids[:5]:', p1[:5])\""
    )


if __name__ == "__main__":
    main()
