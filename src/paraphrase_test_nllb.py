"""
paraphrase_test_nllb.py
-----------------------
Back-translation paraphraser: English -> French -> English (NLLB-200).

Why this exists
---------------
The main pipeline uses Llama-3.1-8B for both text generation AND paraphrasing.
A reviewer can argue the Hard-bucket collapse is a Llama-specific artifact.
This script uses a completely different mechanism — seq2seq machine translation,
no instruction-following LLM — to test whether the same collapse appears.

If Hard-bucket F1 collapses under NLLB back-translation too, the single-model
confound is defeated.

Pipeline
--------
P0 (100 test rows) -[en->fr->en]-> P1_nllb -[en->fr->en]-> P2_nllb

Model: facebook/nllb-200-distilled-600M  (~1.2 GB download, CPU-feasible)
Time: ~90 minutes on CPU for 200 translations (100 rows x 2 rounds).

Outputs
-------
    data/p1/p1_test_nllb.jsonl
    data/p2/p2_test_nllb.jsonl

Resumable: if P1 file already exists it is loaded rather than recomputed.
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG   = "eng_Latn"
INT_LANG   = "fra_Latn"   # intermediate: French
MAX_LEN    = 512
BATCH_SIZE = 4            # safe for CPU RAM; increase to 8 if you have 16GB+ RAM


# ── I/O helpers ──────────────────────────────────────────────────────────────

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


# ── Translation ───────────────────────────────────────────────────────────────

def translate_batch(
    texts: list,
    tokenizer,
    model,
    src_lang: str,
    tgt_lang: str,
) -> list:
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=MAX_LEN,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def back_translate(texts: list, tokenizer, model) -> list:
    """English -> French -> English, in batches."""
    all_en = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="  batches", leave=False):
        batch = texts[i : i + BATCH_SIZE]
        fr = translate_batch(batch, tokenizer, model, SRC_LANG, INT_LANG)
        en = translate_batch(fr,    tokenizer, model, INT_LANG, SRC_LANG)
        all_en.extend(en)
    return all_en


def paraphrase_rows(rows: list, tokenizer, model, stage: str) -> list:
    print(f"\nBack-translating {stage} ({len(rows)} rows, en->fr->en)...")
    texts = [r["text"] for r in rows]
    translated = back_translate(texts, tokenizer, model)

    out = []
    for row, new_text in zip(rows, translated):
        new_row = dict(row)
        new_row["text"] = new_text.strip()
        new_row["meta"] = dict(row.get("meta", {}))
        new_row["meta"]["paraphrase_model"]  = MODEL_NAME
        new_row["meta"]["paraphrase_track"]  = "nllb_backtranslation"
        new_row["meta"]["paraphrase_stage"]  = stage
        new_row["meta"]["intermediate_lang"] = INT_LANG
        out.append(new_row)
    return out


# ── Diagnostics ───────────────────────────────────────────────────────────────

def token_jaccard(a: str, b: str) -> float:
    sa = set(re.findall(r"\w+", a.lower()))
    sb = set(re.findall(r"\w+", b.lower()))
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def jaccard_report(p0_by_id: dict, rows: list, label: str) -> None:
    vals = [
        token_jaccard(p0_by_id[r["id"]]["text"], r["text"])
        for r in rows if r["id"] in p0_by_id
    ]
    unchanged = sum(1 for v in vals if v >= 0.99)
    near_copy = sum(1 for v in vals if v >= 0.85)
    print(f"  {label}: median Jaccard={statistics.median(vals):.3f}  "
          f"unchanged={unchanged}  near-copy(>=0.85)={near_copy}/{len(vals)}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading NLLB model: {MODEL_NAME}")
    print("First run downloads ~1.2 GB and caches to HuggingFace cache.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()
    print("Model loaded. Running on CPU.")

    # Load P0 test rows
    test_ids = load_ids(DATA / "splits" / "test_ids.txt")
    p0_all   = load_jsonl(DATA / "p0" / "p0.jsonl")
    p0_test  = [r for r in p0_all if r["id"] in test_ids]
    p0_by_id = {r["id"]: r for r in p0_test}
    print(f"Loaded {len(p0_test)} test rows from P0.")

    # P1 — back-translate P0
    out_p1 = DATA / "p1" / "p1_test_nllb.jsonl"
    if out_p1.exists():
        print(f"\nP1 already exists ({out_p1.name}). Loading...")
        p1 = load_jsonl(out_p1)
    else:
        p1 = paraphrase_rows(p0_test, tokenizer, model, "P1_nllb")
        save_jsonl(out_p1, p1)
        print(f"Saved: {out_p1} ({len(p1)} rows)")

    # P2 — back-translate P1
    out_p2 = DATA / "p2" / "p2_test_nllb.jsonl"
    if out_p2.exists():
        print(f"\nP2 already exists ({out_p2.name}). Loading...")
        p2 = load_jsonl(out_p2)
    else:
        p2 = paraphrase_rows(p1, tokenizer, model, "P2_nllb")
        save_jsonl(out_p2, p2)
        print(f"Saved: {out_p2} ({len(p2)} rows)")

    # Diagnostics
    print("\nToken Jaccard (vs P0 originals):")
    jaccard_report(p0_by_id, p1, "P1_nllb")
    jaccard_report(p0_by_id, p2, "P2_nllb")

    print("\nDone. Next step:")
    print("  python src/experiments/evaluate_nllb_track.py")


if __name__ == "__main__":
    main()
