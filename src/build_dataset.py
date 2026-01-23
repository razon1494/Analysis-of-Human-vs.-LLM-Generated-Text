import json
import os
import random
import hashlib


def wc(text: str) -> int:
    return len(text.split())


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def dedup_and_filter(rows, label: str, min_words: int, max_words: int):
    kept = []
    seen = set()  # IMPORTANT: per-class dedup
    total = 0
    inrange = 0
    empty = 0
    for obj in rows:
        total += 1
        text = (obj.get("text") or "").strip()
        if not text:
            empty += 1
            continue
        n = wc(text)
        if n < min_words or n > max_words:
            continue
        inrange += 1
        h = stable_hash(text)
        if h in seen:
            continue
        seen.add(h)

        if label == "human":
            kept.append({
                "id": obj.get("id"),
                "text": text,
                "label": "human",
                "meta": {
                    "source": obj.get("source"),
                    "title": obj.get("title"),
                    "url": obj.get("url"),
                    "config": obj.get("config"),
                }
            })
        else:
            kept.append({
                "id": obj.get("id"),
                "text": text,
                "label": "llm",
                "meta": {
                    "source": obj.get("source"),
                    "model": obj.get("model"),
                    "temperature": obj.get("temperature"),
                    "topic": obj.get("topic"),
                }
            })

    print(f"\n[{label.upper()}] total lines read: {total}")
    print(f"[{label.upper()}] in {min_words}–{max_words} words: {inrange}")
    print(f"[{label.upper()}] empty text skipped: {empty}")
    print(f"[{label.upper()}] after dedup kept: {len(kept)}")
    return kept


def main():
    HUMAN_PATH = os.path.join("data", "raw_human", "human.jsonl")
    LLM_PATH = os.path.join("data", "raw_llm", "llm.jsonl")

    OUT_ALL = os.path.join("data", "processed", "all.jsonl")
    OUT_P0 = os.path.join("data", "p0", "p0.jsonl")

    TARGET_PER_CLASS = 500
    MIN_WORDS = 100
    MAX_WORDS = 200
    SEED = 42

    rng = random.Random(SEED)
    os.makedirs(os.path.dirname(OUT_ALL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_P0), exist_ok=True)

    print("HUMAN_PATH =", os.path.abspath(HUMAN_PATH))
    print("LLM_PATH   =", os.path.abspath(LLM_PATH))

    human_rows = list(load_jsonl(HUMAN_PATH))
    llm_rows = list(load_jsonl(LLM_PATH))

    human = dedup_and_filter(human_rows, "human", MIN_WORDS, MAX_WORDS)
    llm = dedup_and_filter(llm_rows, "llm", MIN_WORDS, MAX_WORDS)

    if len(human) < TARGET_PER_CLASS:
        raise RuntimeError(f"Not enough human samples after filtering: {len(human)}")
    if len(llm) < TARGET_PER_CLASS:
        raise RuntimeError(f"Not enough llm samples after filtering: {len(llm)}")

    rng.shuffle(human)
    rng.shuffle(llm)
    human = human[:TARGET_PER_CLASS]
    llm = llm[:TARGET_PER_CLASS]

    all_data = human + llm
    rng.shuffle(all_data)

    with open(OUT_ALL, "w", encoding="utf-8") as f:
        for row in all_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(OUT_P0, "w", encoding="utf-8") as f:
        for row in all_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_ALL}")
    print(f"Saved: {OUT_P0}")
    print(f"Total: {len(all_data)} (human={len(human)}, llm={len(llm)})")


if __name__ == "__main__":
    main()
