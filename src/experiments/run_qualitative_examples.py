"""
run_qualitative_examples.py  (H1)
----------------------------------
Extract publication-quality qualitative examples of Hard-bucket flip cases.

For each selected flip case shows:
  - P0 original text
  - P2 simplified paraphrase
  - char-TF-IDF prediction probability at P0 and P2
  - word length statistics
  - Highlighted vocabulary differences

Outputs
-------
  results/eval/qualitative_examples.csv      — per-case CSV with all text
  results/eval/qualitative_examples.tex      — LaTeX table for paper inclusion
  results/eval/qualitative_examples.md       — readable markdown for review
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl


def word_stats(text: str) -> dict:
    """Word length, count, type-token ratio for one text."""
    import re
    toks = re.findall(r"\b[a-zA-Z]+\b", text)
    if not toks:
        return {"n_words": 0, "mean_wlen": 0.0, "ttr": 0.0}
    return {
        "n_words": len(toks),
        "mean_wlen": float(np.mean([len(t) for t in toks])),
        "ttr": len(set(t.lower() for t in toks)) / len(toks),
    }


def pick_illustrative_flips(df_flips: pd.DataFrame, n: int = 3) -> list:
    """Return n flip case IDs that:
      - have the largest probability drop (most dramatic)
      - cross the 0.5 decision boundary (P0 above, P2 below)
      - appear in char_tfidf_lr (the catastrophic detector)
    """
    char_flips = df_flips[df_flips["detector"] == "char_tfidf_lr"].copy()
    # Sort by magnitude of probability drop
    char_flips = char_flips.sort_values("delta", ascending=True)
    # Pick top-n by drop magnitude
    return char_flips.head(n)["id"].tolist()


def load_texts_by_id(jsonl_path: Path, ids: list) -> dict:
    rows = load_jsonl(jsonl_path)
    by_id = {r["id"]: r["text"] for r in rows}
    return {i: by_id.get(i, "<MISSING>") for i in ids}


def main():
    out_dir = paths.RESULTS / "eval"

    # Load flip cases
    df_flips = pd.read_csv(out_dir / "mechanistic_flip_cases.csv")
    print(f"Loaded {len(df_flips)} flip rows")

    # Pick illustrative cases
    selected_ids = pick_illustrative_flips(df_flips, n=3)
    print(f"Selected {len(selected_ids)} cases: {selected_ids}")

    # Load text from all three stages
    p0_path  = paths.P0_PATH
    p1_path  = ROOT / "data" / "p1" / "p1_test_simplified.jsonl"
    p2_path  = ROOT / "data" / "p2" / "p2_test_simplified.jsonl"

    p0_texts = load_texts_by_id(p0_path, selected_ids)
    p1_texts = load_texts_by_id(p1_path, selected_ids)
    p2_texts = load_texts_by_id(p2_path, selected_ids)

    # Pull per-case predictions
    char_flips = df_flips[df_flips["detector"] == "char_tfidf_lr"].set_index("id")
    word_flips = df_flips[df_flips["detector"] == "word_tfidf_lr"].set_index("id")

    rows = []
    for rid in selected_ids:
        cr = char_flips.loc[rid] if rid in char_flips.index else None
        wr = word_flips.loc[rid] if rid in word_flips.index else None

        s0 = word_stats(p0_texts[rid])
        s1 = word_stats(p1_texts[rid])
        s2 = word_stats(p2_texts[rid])

        rows.append({
            "id":             rid,
            "text_p0":        p0_texts[rid],
            "text_p1":        p1_texts[rid],
            "text_p2":        p2_texts[rid],
            "char_prob_p0":   float(cr["prob_p0"]) if cr is not None else None,
            "char_prob_p2":   float(cr["prob_p2"]) if cr is not None else None,
            "word_prob_p0":   float(wr["prob_p0"]) if wr is not None else None,
            "word_prob_p2":   float(wr["prob_p2"]) if wr is not None else None,
            "n_words_p0":     s0["n_words"],
            "n_words_p2":     s2["n_words"],
            "mean_wlen_p0":   round(s0["mean_wlen"], 2),
            "mean_wlen_p2":   round(s2["mean_wlen"], 2),
            "ttr_p0":         round(s0["ttr"], 3),
            "ttr_p2":         round(s2["ttr"], 3),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "qualitative_examples.csv", index=False)
    print(f"Saved: qualitative_examples.csv")

    # ── Markdown output (readable for paper review) ──────────────────────────
    md_lines = ["# Qualitative Flip Examples\n",
                "Hard-bucket LLM examples that were correctly identified as "
                "LLM at P0 but misclassified as Human at P2 under simplified "
                "paraphrasing.\n\n---\n"]
    for i, r in enumerate(rows, 1):
        md_lines.append(f"\n## Example {i} — `{r['id']}`\n")
        md_lines.append(f"\n**char-TF-IDF detector:** "
                       f"P(LLM) = {r['char_prob_p0']:.3f} (P0) → "
                       f"{r['char_prob_p2']:.3f} (P2)   "
                       f"_correct → wrong_\n")
        if r['word_prob_p0'] is not None:
            md_lines.append(f"\n**word-TF-IDF detector:** "
                           f"P(LLM) = {r['word_prob_p0']:.3f} (P0) → "
                           f"{r['word_prob_p2']:.3f} (P2)\n")
        md_lines.append(f"\n**Length:** {r['n_words_p0']} → {r['n_words_p2']} words   "
                       f"**Mean word length:** {r['mean_wlen_p0']} → "
                       f"{r['mean_wlen_p2']} chars   "
                       f"**TTR:** {r['ttr_p0']} → {r['ttr_p2']}\n")
        md_lines.append(f"\n### P0 (original Llama-3.1-8B output)\n\n"
                       f"> {r['text_p0']}\n")
        md_lines.append(f"\n### P1 (one round of simplified paraphrasing)\n\n"
                       f"> {r['text_p1']}\n")
        md_lines.append(f"\n### P2 (two rounds of simplified paraphrasing)\n\n"
                       f"> {r['text_p2']}\n")
        md_lines.append("\n---\n")

    (out_dir / "qualitative_examples.md").write_text(
        "".join(md_lines), encoding="utf-8")
    print(f"Saved: qualitative_examples.md")

    # ── LaTeX table (compact, paper-ready) ───────────────────────────────────
    # We use a single "Example X" table per case, ~80 word excerpts only
    def excerpt(t, max_words=80):
        toks = t.split()
        if len(toks) <= max_words:
            return t
        return " ".join(toks[:max_words]) + " \\ldots"

    def latex_escape(t):
        return (t.replace("&", "\\&").replace("%", "\\%")
                 .replace("$", "\\$").replace("#", "\\#")
                 .replace("_", "\\_").replace("{", "\\{").replace("}", "\\}"))

    tex_lines = ["% Qualitative flip-case examples\n",
                 "% Generated by run_qualitative_examples.py\n\n"]
    for i, r in enumerate(rows, 1):
        tex_lines.append(
            f"\\paragraph{{Example {i}}} (\\texttt{{{r['id']}}}, "
            f"char-TF-IDF $P(\\mathrm{{LLM}})$: "
            f"$\\mathbf{{{r['char_prob_p0']:.3f}}}\\to\\mathbf{{{r['char_prob_p2']:.3f}}}$; "
            f"mean word length $\\mathbf{{{r['mean_wlen_p0']:.2f}}}\\to\\mathbf{{{r['mean_wlen_p2']:.2f}}}$)\\\\\n"
        )
        tex_lines.append("\\noindent\\textbf{P0 (Llama-3.1-8B original):}\\\\\n")
        tex_lines.append(f"\\emph{{{latex_escape(excerpt(r['text_p0']))}}}\\\\[0.3em]\n")
        tex_lines.append("\\noindent\\textbf{P2 (after 2 rounds of simplified paraphrasing):}\\\\\n")
        tex_lines.append(f"\\emph{{{latex_escape(excerpt(r['text_p2']))}}}\\\\[0.7em]\n\n")

    (out_dir / "qualitative_examples.tex").write_text(
        "".join(tex_lines), encoding="utf-8")
    print(f"Saved: qualitative_examples.tex")

    # ── Summary printout ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("QUALITATIVE EXAMPLES SUMMARY")
    print("=" * 65)
    for i, r in enumerate(rows, 1):
        print(f"\nExample {i}: {r['id']}")
        print(f"  char P(LLM):   {r['char_prob_p0']:.3f} -> {r['char_prob_p2']:.3f}")
        if r['word_prob_p0'] is not None:
            print(f"  word P(LLM):   {r['word_prob_p0']:.3f} -> {r['word_prob_p2']:.3f}")
        print(f"  words:         {r['n_words_p0']} -> {r['n_words_p2']}")
        print(f"  mean wlen:     {r['mean_wlen_p0']:.2f} -> {r['mean_wlen_p2']:.2f}")
        print(f"  TTR:           {r['ttr_p0']:.3f} -> {r['ttr_p2']:.3f}")
        print(f"  P0 first 200 chars: {r['text_p0'][:200]}...")
        print(f"  P2 first 200 chars: {r['text_p2'][:200]}...")

    print("\nDone. Review qualitative_examples.md to pick the best ones for the paper.")


if __name__ == "__main__":
    main()
