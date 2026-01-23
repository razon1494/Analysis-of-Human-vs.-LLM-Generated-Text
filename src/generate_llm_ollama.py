import json
import os
import random
import time
from typing import List, Dict

import requests
from tqdm import tqdm


OLLAMA_URL = "http://localhost:11434/api/generate"


def normalize_one_paragraph(text: str) -> str:
    # Collapse newlines into one paragraph
    return " ".join([t.strip() for t in text.splitlines() if t.strip()]).strip()


def wc(text: str) -> int:
    return len(text.split())


def load_topics_from_human(path: str, n: int) -> List[str]:
    # Use Wikipedia titles as generation topics (keeps topical distribution aligned)
    topics = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            title = (obj.get("title") or "").strip()
            if title:
                topics.append(title)
            if len(topics) >= n:
                break
    return topics


def prompt_for(topic: str) -> str:
    return (
        f"Write ONE neutral encyclopedic paragraph (120–180 words) explaining: {topic}.\n"
        "Constraints:\n"
        "- Factual tone, no opinions, no first-person.\n"
        "- No headings, no bullet points, no lists.\n"
        "- No citations.\n"
        "- Single paragraph only.\n"
    )


def ollama_generate(model: str, topic: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt_for(topic),
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def main():
    HUMAN_PATH = os.path.join("data", "raw_human", "human.jsonl")
    OUT_PATH = os.path.join("data", "raw_llm", "llm.jsonl")
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    MODEL = "llama3.1:8b"
    TARGET_N = 500
    MIN_WORDS = 100
    MAX_WORDS = 200
    TEMPERATURE = 0.0
    SEED = 42

    rng = random.Random(SEED)

    topics = load_topics_from_human(HUMAN_PATH, TARGET_N)
    if len(topics) < TARGET_N:
        raise RuntimeError(f"Found only {len(topics)} topics in {HUMAN_PATH}; need {TARGET_N}.")

    rng.shuffle(topics)

    results: List[Dict] = []
    skipped = 0

    pbar = tqdm(total=TARGET_N, desc="Generating LLM paragraphs", unit="para")

    for idx, topic in enumerate(topics, start=1):
        best = ""

        for attempt in range(3):
            try:
                text = ollama_generate(MODEL, topic, TEMPERATURE)
                text = normalize_one_paragraph(text)
            except Exception:
                time.sleep(2)
                continue

            n_words = wc(text)

            # If model returns weird stuff (too short/long), try again
            if MIN_WORDS <= n_words <= MAX_WORDS:
                best = text
                break

            time.sleep(0.6)

        if not best:
            skipped += 1
            continue

        results.append({
            "id": f"llm_{len(results)+1:07d}",
            "source": "ollama",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "topic": topic,
            "text": best,
            "label": "llm"
        })
        pbar.update(1)

        if len(results) >= TARGET_N:
            break

    pbar.close()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved: {OUT_PATH}")
    print(f"Generated: {len(results)} / {TARGET_N}")
    print(f"Skipped: {skipped}")
    if len(results) < TARGET_N:
        print("Tip: If too many skips, lower constraints or allow 4–5 attempts per topic.")


if __name__ == "__main__":
    main()
