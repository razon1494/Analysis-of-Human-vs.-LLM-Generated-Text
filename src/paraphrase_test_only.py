import json
import os
import time
import requests
from pathlib import Path
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
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()

def paraphrase_once(model: str, text: str) -> str:
    out = ollama_generate(model, paraphrase_prompt(text), temperature=0.2)
    return one_paragraph(out)

def load_ids(path: str):
    return set(Path(path).read_text(encoding="utf-8").splitlines())

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def paraphrase_rows(rows, model: str, stage: str):
    MIN_WORDS, MAX_WORDS = 100, 200
    out = []
    pbar = tqdm(total=len(rows), desc=f"Paraphrasing {stage}", unit="row")
    for obj in rows:
        text = obj["text"]
        best = ""
        for _ in range(4):
            try:
                cand = paraphrase_once(model, text)
            except Exception:
                time.sleep(2)
                continue
            n = wc(cand)
            if MIN_WORDS <= n <= MAX_WORDS and cand and cand != text:
                best = cand
                break
            time.sleep(0.6)
        if not best:
            best = text

        new_obj = dict(obj)
        new_obj["text"] = best
        new_obj["meta"] = dict(obj.get("meta", {}))
        new_obj["meta"]["paraphrase_model"] = model
        new_obj["meta"]["paraphrase_stage"] = stage
        out.append(new_obj)
        pbar.update(1)
    pbar.close()
    return out

def main():
    MODEL = "llama3.1:8b"
    P0 = os.path.join("data", "p0", "p0.jsonl")
    TEST_IDS = os.path.join("data", "splits", "test_ids.txt")

    ids = load_ids(TEST_IDS)
    test_rows = [r for r in load_jsonl(P0) if r["id"] in ids]

    out_p1 = os.path.join("data", "p1", "p1_test.jsonl")
    out_p2 = os.path.join("data", "p2", "p2_test.jsonl")

    p1 = paraphrase_rows(test_rows, MODEL, "p1_test")
    save_jsonl(out_p1, p1)

    p2 = paraphrase_rows(p1, MODEL, "p2_test")
    save_jsonl(out_p2, p2)

    print(f"Saved: {out_p1} rows={len(p1)}")
    print(f"Saved: {out_p2} rows={len(p2)}")

if __name__ == "__main__":
    main()
