"""
diagnose_paraphrase_quality.py
-------------------------------
Stdlib-only diagnostic. Quantifies paraphrase-fallback contamination
and lexical change rates that the current evaluation does not report.

Why this matters
----------------
paraphrase_test_only.py:69-70 (and the simplified variant) silently fall
back to the ORIGINAL text when the LLM fails to produce a constrained
paraphrase within 4 retries:

    if not best:
        best = text                 # <— silent contamination

If even a few percent of P1/P2 samples are unchanged, every robustness
number is biased upward. Reviewers will reject any claim that doesn't
quantify this.

This script reports for each paraphrase split:
  - n_total
  - n_unchanged   (text identical to its P0 source)
  - n_low_change  (Jaccard token overlap > 0.85 with P0 — near-copy)
  - Jaccard distribution (min/median/max)
  - length-delta distribution
  - per-label breakdown (does fallback hit LLM rows more than human?)

Output
------
    results/paraphrase_quality.csv     -- per-split summary
    results/paraphrase_quality.json    -- full distributions

Usage
-----
    python src/diagnose_paraphrase_quality.py
"""

from __future__ import annotations

import json
import os
import re
import statistics
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SPLITS = [
    ("P1_test_standard",   DATA_DIR / "p1" / "p1_test.jsonl"),
    ("P2_test_standard",   DATA_DIR / "p2" / "p2_test.jsonl"),
    ("P1_test_simplified", DATA_DIR / "p1" / "p1_test_simplified.jsonl"),
    ("P2_test_simplified", DATA_DIR / "p2" / "p2_test_simplified.jsonl"),
]

WORD_RE = re.compile(r"\b\w+\b")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))


def summarize_split(name: str, p0_text_by_id: dict[str, dict], rows: list[dict]) -> dict:
    n_total = len(rows)
    n_unchanged = 0
    n_low_change = 0
    n_id_missing = 0
    per_label_unchanged = {"human": 0, "llm": 0}
    per_label_total = {"human": 0, "llm": 0}

    jaccards: list[float] = []
    len_deltas: list[int] = []

    examples_unchanged: list[dict] = []

    for r in rows:
        rid = r.get("id")
        text = r.get("text", "")
        label = r.get("label", "unknown")
        per_label_total[label] = per_label_total.get(label, 0) + 1

        if rid not in p0_text_by_id:
            n_id_missing += 1
            continue

        p0_text = p0_text_by_id[rid]["text"]
        unchanged = (text.strip() == p0_text.strip())
        if unchanged:
            n_unchanged += 1
            per_label_unchanged[label] = per_label_unchanged.get(label, 0) + 1
            if len(examples_unchanged) < 5:
                examples_unchanged.append({
                    "id": rid,
                    "label": label,
                    "len_words": len(text.split()),
                    "preview": text[:160] + ("..." if len(text) > 160 else ""),
                })

        j = jaccard(p0_text, text)
        jaccards.append(j)
        if j > 0.85 and not unchanged:
            n_low_change += 1

        len_deltas.append(len(text.split()) - len(p0_text.split()))

    return {
        "split": name,
        "n_total": n_total,
        "n_id_missing": n_id_missing,
        "n_unchanged": n_unchanged,
        "pct_unchanged": round(100.0 * n_unchanged / max(1, n_total), 2),
        "n_low_change_jaccard_gt_0.85": n_low_change,
        "pct_low_change": round(100.0 * n_low_change / max(1, n_total), 2),
        "jaccard_min":    round(min(jaccards), 4) if jaccards else None,
        "jaccard_median": round(statistics.median(jaccards), 4) if jaccards else None,
        "jaccard_mean":   round(statistics.mean(jaccards), 4) if jaccards else None,
        "jaccard_max":    round(max(jaccards), 4) if jaccards else None,
        "len_delta_median": int(statistics.median(len_deltas)) if len_deltas else None,
        "len_delta_mean":   round(statistics.mean(len_deltas), 2) if len_deltas else None,
        "len_delta_min":    min(len_deltas) if len_deltas else None,
        "len_delta_max":    max(len_deltas) if len_deltas else None,
        "per_label_unchanged": per_label_unchanged,
        "per_label_total":     per_label_total,
        "examples_unchanged":  examples_unchanged,
        "jaccard_distribution": jaccards,
        "len_delta_distribution": len_deltas,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test_ids = set(
        (DATA_DIR / "splits" / "test_ids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    p0_all = load_jsonl(DATA_DIR / "p0" / "p0.jsonl")
    p0_test = [r for r in p0_all if r["id"] in test_ids]
    p0_by_id = {r["id"]: r for r in p0_test}

    print(f"P0 test rows: {len(p0_test)}")
    print(f"Diagnosing paraphrase contamination on {len(SPLITS)} splits...\n")

    summaries = []
    for name, path in SPLITS:
        if not path.exists():
            print(f"  SKIP {name}: {path} does not exist")
            continue
        rows = load_jsonl(path)
        s = summarize_split(name, p0_by_id, rows)
        summaries.append(s)

        print(f"== {name} (n={s['n_total']}) ==")
        print(f"   unchanged: {s['n_unchanged']}/{s['n_total']} ({s['pct_unchanged']}%)")
        print(f"   low-change (Jaccard>0.85): {s['n_low_change_jaccard_gt_0.85']} "
              f"({s['pct_low_change']}%)")
        print(f"   Jaccard P0 vs paraphrase: min={s['jaccard_min']} "
              f"median={s['jaccard_median']} max={s['jaccard_max']}")
        print(f"   word-count delta: median={s['len_delta_median']} "
              f"mean={s['len_delta_mean']} range=[{s['len_delta_min']},{s['len_delta_max']}]")
        print(f"   unchanged by label: {s['per_label_unchanged']} "
              f"of {s['per_label_total']}")
        if s['examples_unchanged']:
            print(f"   examples of unchanged: ")
            for e in s['examples_unchanged'][:3]:
                print(f"     [{e['label']:5s}] id={e['id']} ({e['len_words']}w): {e['preview']}")
        print()

    # ── save outputs ──────────────────────────────────────────────────────────
    out_json = RESULTS_DIR / "paraphrase_quality.json"
    out_json.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    out_csv = RESULTS_DIR / "paraphrase_quality.csv"
    csv_cols = [
        "split", "n_total", "n_unchanged", "pct_unchanged",
        "n_low_change_jaccard_gt_0.85", "pct_low_change",
        "jaccard_min", "jaccard_median", "jaccard_mean", "jaccard_max",
        "len_delta_median", "len_delta_mean", "len_delta_min", "len_delta_max",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(csv_cols) + "\n")
        for s in summaries:
            f.write(",".join(str(s[c]) for c in csv_cols) + "\n")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_csv}")

    # ── headline summary ──────────────────────────────────────────────────────
    if any(s["n_unchanged"] > 0 for s in summaries):
        print("\n*** CONTAMINATION DETECTED ***")
        print("    One or more splits contain P0-identical rows.")
        print("    All P1/P2 metrics are biased toward P0 baseline and must be re-reported")
        print("    with these rows EXCLUDED or with the paraphraser re-run on failed samples.")
    else:
        print("\nNo identical-text contamination detected.")


if __name__ == "__main__":
    main()
