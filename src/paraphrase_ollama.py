import json
import os
import time
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"

def wc(text: str) -> int:
    return len(text.split())

def one_paragraph(text: str) -> str:
    return " ".join([t.strip() for t in text.splitlines() if t.strip()]).strip()

def paraphrase_prompt(text: str) -> str:
    return (
        "Paraphrase the paragraph while preserving meaning and factual content.\n"
        "Hard constraints:\n"
        "- Output must be ONE paragraph only (no newlines).\n"
        "- Keep length between 100 and 200 words.\n"
        "- No lists, no bullet points, no headings, no citations.\n"
        "- Keep all key facts; do not add new facts.\n\n"
        f"Paragraph:\n{text}"
    )

def ollama_generate(model: str, prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature}
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def paraphrase_text(model: str, text: str) -> str:
    prompt = paraphrase_prompt(text)
    out = ollama_generate(model, prompt)
    return one_paragraph(out)

def build_paraphrase(in_path: str, out_path: str, model: str, seed_tag: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    MIN_WORDS, MAX_WORDS = 100, 200
    TEMPERATURE = 0.2

    out_rows = []
    pbar = tqdm(total=len(rows), desc=f"Paraphrasing -> {os.path.basename(out_path)}", unit="row")

    for obj in rows:
        text = obj["text"]
        best = ""
        for _ in range(4):
            try:
                cand = paraphrase_text(model, text)
            except Exception:
                time.sleep(2)
                continue

            n = wc(cand)
            if MIN_WORDS <= n <= MAX_WORDS and cand and cand != text:
                best = cand
                break
            time.sleep(0.6)

        if not best:
            # fallback: keep original (rare, but keeps dataset size consistent)
            best = text

        new_obj = dict(obj)
        new_obj["text"] = best
        new_obj["meta"] = dict(obj.get("meta", {}))
        new_obj["meta"]["paraphrase_model"] = model
        new_obj["meta"]["paraphrase_temperature"] = TEMPERATURE
        new_obj["meta"]["paraphrase_stage"] = seed_tag  # "p1" or "p2"
        out_rows.append(new_obj)
        pbar.update(1)

    pbar.close()

    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved: {out_path} ({len(out_rows)} rows)")

def main():
    MODEL = "llama3.1:8b"
    P0 = os.path.join("data", "p0", "p0.jsonl")
    P1 = os.path.join("data", "p1", "p1.jsonl")
    P2 = os.path.join("data", "p2", "p2.jsonl")

    # Build P1 from P0, then P2 from P1 (iterative paraphrasing)
    build_paraphrase(P0, P1, MODEL, "p1")
    build_paraphrase(P1, P2, MODEL, "p2")

if __name__ == "__main__":
    main()
