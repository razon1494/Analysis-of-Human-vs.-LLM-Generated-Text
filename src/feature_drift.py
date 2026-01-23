import json
import os
import re
import pandas as pd
from pathlib import Path

SENT_SPLIT = re.compile(r"[.!?]+")

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def features(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    n_words = len(words)

    sents = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    n_sents = max(1, len(sents))

    unique_words = len(set(words)) if words else 0
    ttr = (unique_words / n_words) if n_words else 0.0

    punct = sum(1 for ch in text if ch in ".,;:!?")
    punct_rate = punct / max(1, len(text))

    uniq_ratio = unique_words / max(1, n_words)  # similar to TTR but explicit naming

    return {
        "words": n_words,
        "sents": n_sents,
        "words_per_sent": n_words / n_sents,
        "ttr": ttr,
        "punct_rate": punct_rate,
        "uniq_word_ratio": uniq_ratio,
    }

def summarize(name, rows):
    feats = [features(r["text"]) for r in rows]
    df = pd.DataFrame(feats)
    out = df.mean(numeric_only=True).to_dict()
    out["split"] = name
    out["n"] = len(rows)
    return out

def main():
    # P0_test from ids
    P0_PATH = os.path.join("data", "p0", "p0.jsonl")
    TEST_IDS_PATH = os.path.join("data", "splits", "test_ids.txt")
    test_ids = set(Path(TEST_IDS_PATH).read_text(encoding="utf-8").splitlines())
    p0_rows = [r for r in load_jsonl(P0_PATH) if r["id"] in test_ids]

    p1_rows = load_jsonl(os.path.join("data", "p1", "p1_test.jsonl"))
    p2_rows = load_jsonl(os.path.join("data", "p2", "p2_test.jsonl"))

    summary = [
        summarize("P0_test", p0_rows),
        summarize("P1_test", p1_rows),
        summarize("P2_test", p2_rows),
    ]

    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(summary)
    df.to_csv("results/feature_drift_test.csv", index=False)
    print(df)
    print("\nSaved: results/feature_drift_test.csv")

if __name__ == "__main__":
    main()
