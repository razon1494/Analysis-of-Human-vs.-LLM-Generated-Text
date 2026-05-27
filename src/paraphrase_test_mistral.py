"""
paraphrase_test_mistral.py
--------------------------
Paraphrase the test set using Mistral-7B via Ollama (different LLM family
from the Llama-3.1-8B generator/paraphraser used in the main pipeline).

Why this exists
---------------
P6 paraphraser diversity: show that the Hard-bucket collapse is not a
Llama-specific artifact. Mistral-7B is a different model family (Mistral AI
vs Meta), trained on different data with different RLHF. If the collapse
persists under Mistral paraphrasing, it rules out the LLM-family confound.

Together with NLLB back-translation (P6 Part 1), we have three paraphrasers:
  Llama-3.1-8B    — original (LLM, Meta)
  Mistral-7B      — different LLM family (Mistral AI)         ← this script
  NLLB-200        — MT system, no LLM at all                  ← already done

Both standard and simplified prompts are run in one script.

Outputs
-------
    data/p1/p1_test_mistral.jsonl              standard track, round 1
    data/p2/p2_test_mistral.jsonl              standard track, round 2
    data/p1/p1_test_mistral_simplified.jsonl   simplified track, round 1
    data/p2/p2_test_mistral_simplified.jsonl   simplified track, round 2

Resumable: each file is skipped if it already exists.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path

import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "mistral:latest"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ── prompts (identical wording to Llama tracks for comparability) ─────────────

def prompt_standard(text: str) -> str:
    return (
        "Paraphrase the paragraph while preserving meaning and factual content.\n"
        "Hard constraints:\n"
        "- Output must be ONE paragraph only (no newlines).\n"
        "- Keep length between 100 and 200 words.\n"
        "- No lists, no bullet points, no headings, no citations.\n"
        "- Keep all key facts; do not add new facts.\n\n"
        f"Paragraph:\n{text}"
    )


def prompt_simplified(text: str) -> str:
    return (
        "Rewrite the paragraph in a simplified, non-expert style (easy to understand), "
        "while preserving meaning and factual content.\n"
        "Hard constraints:\n"
        "- Output must be ONE paragraph only (no newlines).\n"
        "- Keep length between 100 and 200 words.\n"
        "- Use simple vocabulary and shorter sentences.\n"
        "- Avoid technical jargon unless necessary.\n"
        "- No lists, no bullet points, no headings, no citations.\n"
        "- Do not add new facts.\n\n"
        f"Paragraph:\n{text}"
    )


# ── Ollama helpers ────────────────────────────────────────────────────────────

def wc(text: str) -> int:
    return len(text.split())


def one_paragraph(text: str) -> str:
    return " ".join([t.strip() for t in text.splitlines() if t.strip()]).strip()


def ollama_generate(prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=240)
    r.raise_for_status()
    return (r.json().get("response") or "").strip()


def paraphrase_once(text: str, prompt_fn) -> str:
    out = ollama_generate(prompt_fn(text), temperature=0.2)
    return one_paragraph(out)


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_ids(path: Path) -> set:
    return set(path.read_text(encoding="utf-8").splitlines())


def load_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── paraphrase loop ───────────────────────────────────────────────────────────

def paraphrase_rows(rows: list, prompt_fn, track: str, stage: str) -> list:
    MIN_WORDS, MAX_WORDS = 100, 200
    out = []
    pbar = tqdm(total=len(rows), desc=f"{track}/{stage}", unit="row")
    for obj in rows:
        text = obj["text"]
        best = ""
        for _ in range(4):
            try:
                cand = paraphrase_once(text, prompt_fn)
            except Exception:
                time.sleep(2)
                continue
            n = wc(cand)
            if MIN_WORDS <= n <= MAX_WORDS and cand and cand != text:
                best = cand
                break
            time.sleep(0.5)
        if not best:
            best = text   # fallback (logged below)

        new_obj = dict(obj)
        new_obj["text"] = best
        new_obj["meta"] = dict(obj.get("meta", {}))
        new_obj["meta"]["paraphrase_model"] = MODEL
        new_obj["meta"]["paraphrase_track"] = track
        new_obj["meta"]["paraphrase_stage"] = stage
        out.append(new_obj)
        pbar.update(1)
    pbar.close()
    return out


# ── diagnostics ───────────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def jaccard_report(p0_by_id: dict, rows: list, label: str) -> None:
    vals = [
        token_jaccard(p0_by_id[r["id"]]["text"], r["text"])
        for r in rows if r["id"] in p0_by_id
    ]
    unchanged  = sum(1 for v in vals if v >= 0.99)
    near_copy  = sum(1 for v in vals if v >= 0.85)
    print(f"  {label}: median Jaccard={statistics.median(vals):.3f}  "
          f"unchanged={unchanged}  near-copy(>=0.85)={near_copy}/{len(vals)}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_track(
    p0_test: list,
    prompt_fn,
    track: str,
    out_p1: Path,
    out_p2: Path,
) -> tuple:
    if out_p1.exists():
        print(f"  {out_p1.name} already exists — loading.")
        p1 = load_jsonl(out_p1)
    else:
        p1 = paraphrase_rows(p0_test, prompt_fn, track, "P1")
        save_jsonl(out_p1, p1)
        print(f"  Saved {out_p1.name} ({len(p1)} rows)")

    if out_p2.exists():
        print(f"  {out_p2.name} already exists — loading.")
        p2 = load_jsonl(out_p2)
    else:
        p2 = paraphrase_rows(p1, prompt_fn, track, "P2")
        save_jsonl(out_p2, p2)
        print(f"  Saved {out_p2.name} ({len(p2)} rows)")

    return p1, p2


def main():
    # Load test rows
    test_ids = load_ids(DATA / "splits" / "test_ids.txt")
    p0_all   = load_jsonl(DATA / "p0" / "p0.jsonl")
    p0_test  = [r for r in p0_all if r["id"] in test_ids]
    p0_by_id = {r["id"]: r for r in p0_test}
    print(f"Loaded {len(p0_test)} test rows. Model: {MODEL}\n")

    # ── Standard track ────────────────────────────────────────────────────────
    print("=== Standard track (preserve meaning) ===")
    p1_std, p2_std = run_track(
        p0_test,
        prompt_standard,
        "mistral_standard",
        DATA / "p1" / "p1_test_mistral.jsonl",
        DATA / "p2" / "p2_test_mistral.jsonl",
    )

    # ── Simplified track ──────────────────────────────────────────────────────
    print("\n=== Simplified track (non-expert style) ===")
    p1_sim, p2_sim = run_track(
        p0_test,
        prompt_simplified,
        "mistral_simplified",
        DATA / "p1" / "p1_test_mistral_simplified.jsonl",
        DATA / "p2" / "p2_test_mistral_simplified.jsonl",
    )

    # ── Diagnostics ───────────────────────────────────────────────────────────
    print("\nToken Jaccard (vs P0 originals):")
    jaccard_report(p0_by_id, p1_std, "P1_mistral_standard   ")
    jaccard_report(p0_by_id, p2_std, "P2_mistral_standard   ")
    jaccard_report(p0_by_id, p1_sim, "P1_mistral_simplified ")
    jaccard_report(p0_by_id, p2_sim, "P2_mistral_simplified ")

    fallbacks_std = sum(1 for r0, r in zip(p0_test, p2_std)
                        if r["text"] == r0["text"])
    fallbacks_sim = sum(1 for r0, r in zip(p0_test, p2_sim)
                        if r["text"] == r0["text"])
    print(f"\nFallback-to-original at P2: standard={fallbacks_std}, "
          f"simplified={fallbacks_sim}")

    print("\nDone. Next step:")
    print("  python src/experiments/evaluate_mistral_track.py")


if __name__ == "__main__":
    main()
