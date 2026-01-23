import json
import os
import re
import random
import hashlib
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_into_paragraphs(text: str) -> List[str]:
    chunks = re.split(r"\n\s*\n", text)
    if len(chunks) <= 1:
        chunks = text.split("\n")
    paras = [normalize_ws(p) for p in chunks if normalize_ws(p)]
    return paras


def word_count(s: str) -> int:
    return len(s.split())


def stable_hash(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def reservoir_add(reservoir: List[Dict], item: Dict, seen: int, k: int, rng: random.Random) -> None:
    if len(reservoir) < k:
        reservoir.append(item)
    else:
        j = rng.randint(0, seen - 1)
        if j < k:
            reservoir[j] = item


def main():
    OUT_PATH = os.path.join("data", "raw_human", "human.jsonl")
    TARGET_N = 500
    MIN_WORDS = 100
    MAX_WORDS = 200
    SEED = 42
    WIKI_CONFIG = "20231101.en"      # English Wikipedia config
    MAX_ARTICLES_TO_SCAN = 50_000

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rng = random.Random(SEED)

    # Streaming avoids downloading the full dump
    ds = load_dataset("wikimedia/wikipedia", WIKI_CONFIG, split="train", streaming=True)

    reservoir: List[Dict] = []
    dedup = set()
    seen_paras = 0
    scanned_articles = 0

    pbar = tqdm(total=TARGET_N, desc="Collecting human paragraphs", unit="para")

    for ex in ds:
        scanned_articles += 1
        if scanned_articles > MAX_ARTICLES_TO_SCAN:
            break

        text = ex.get("text", "")
        title = ex.get("title", "")
        url = ex.get("url", "")

        if not text or not isinstance(text, str):
            continue

        for para in split_into_paragraphs(text):
            wc = word_count(para)
            if wc < MIN_WORDS or wc > MAX_WORDS:
                continue

            h = stable_hash(para)
            if h in dedup:
                continue
            dedup.add(h)

            seen_paras += 1
            item = {
                "id": f"human_{seen_paras:07d}",
                "source": "wikimedia/wikipedia",
                "config": WIKI_CONFIG,
                "title": title,
                "url": url,
                "text": para,
                "label": "human"
            }

            reservoir_add(reservoir, item, seen_paras, TARGET_N, rng)

            if len(reservoir) <= TARGET_N:
                pbar.n = len(reservoir)
                pbar.refresh()

        if len(reservoir) >= TARGET_N and scanned_articles >= 5000:
            break

    pbar.close()
    rng.shuffle(reservoir)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in reservoir:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Collected paragraphs: {len(reservoir)}")
    print(f"Articles scanned: {scanned_articles}")


if __name__ == "__main__":
    main()
