import json
import os
import time
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

def wc(text: str) -> int:
    return len(text.split())

def normalize_one_paragraph(text: str) -> str:
    return " ".join([t.strip() for t in text.splitlines() if t.strip()]).strip()

def prompt_for(topic: str) -> str:
    # Stronger length control: ask for EXACT word range and single paragraph.
    return (
        f"Write ONE neutral encyclopedic paragraph about: {topic}.\n"
        "Hard constraints:\n"
        "- Single paragraph only (no newlines).\n"
        "- 130 to 170 words.\n"
        "- Factual tone, no opinions, no first-person.\n"
        "- No headings, no lists, no bullet points, no citations.\n"
    )

def ollama_generate(model: str, topic: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "prompt": prompt_for(topic),
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def main():
    LLM_PATH = os.path.join("data", "raw_llm", "llm.jsonl")
    HUMAN_PATH = os.path.join("data", "raw_human", "human.jsonl")
    
    MODEL = "llama3.1:8b"
    TEMPERATURE = 0.0
    MIN_WORDS = 150
    MAX_WORDS = 190

    # Count current valid samples
    llm_rows = []
    with open(LLM_PATH, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            llm_rows.append(obj)

    valid = [r for r in llm_rows if MIN_WORDS <= wc(r["text"]) <= MAX_WORDS]
    need = 17

    print(f"Valid LLM in given range of words: {len(valid)}")
    if need <= 0:
        print("No top-up needed.")
        return

    print(f"Top-up needed: {need}")

    # Load topics from human titles (reuse same distribution)
    topics = []
    with open(HUMAN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            t = (json.loads(line).get("title") or "").strip()
            if t:
                topics.append(t)

    # Start generating from the tail of topics to reduce repeats
    start_idx = len(llm_rows) % len(topics)

    new_rows = []
    pbar = tqdm(total=need, desc="Top-up LLM samples", unit="para")

    i = 0
    while len(new_rows) < need:
        topic = topics[(start_idx + i) % len(topics)]
        i += 1

        best = ""
        for _ in range(4):
            try:
                text = normalize_one_paragraph(ollama_generate(MODEL, topic, TEMPERATURE))
            except Exception:
                time.sleep(2)
                continue

            if MIN_WORDS <= wc(text) <= MAX_WORDS:
                best = text
                break
            time.sleep(0.6)

        if not best:
            continue

        new_rows.append({
            "id": f"llm_topup_{len(new_rows)+1:05d}",
            "source": "ollama",
            "model": MODEL,
            "temperature": TEMPERATURE,
            "topic": topic,
            "text": best,
            "label": "llm"
        })
        pbar.update(1)

    pbar.close()

    # Append to file
    with open(LLM_PATH, "a", encoding="utf-8") as f:
        for row in new_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Appended {len(new_rows)} new rows to {LLM_PATH}")

if __name__ == "__main__":
    main()
