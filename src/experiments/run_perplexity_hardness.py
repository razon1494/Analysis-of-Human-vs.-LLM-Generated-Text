"""
run_perplexity_hardness.py  (H2)
---------------------------------
Add GPT-2 perplexity as a fourth, independent hardness signal.

Why this matters
----------------
The existing text-only hardness definitions (readability_fk, ttr, length)
all measure surface-level lexical/syntactic complexity. A reviewer can ask:
"What if 'Hard' just means 'lexically formal'?" Perplexity from an external
LM measures a different axis — how predictable the text is to a separate
generative model trained on web text. Reproducing the Hard-bucket F1
collapse under GPT-2 perplexity hardness rules out the lexical-formality
confound.

Convention
----------
HIGH GPT-2 perplexity = the LM finds the text unusual = HARD.
We also report the LOW-is-hard variant as a sanity check.

Outputs
-------
  results/eval/perplexity_scores.csv          — per-text PPL and tertile
  results/eval/perplexity_concordance.csv     — Kendall tau vs other hardness
  results/eval/perplexity_hardness_buckets.csv — Hard F1 under PPL hardness
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from lib import paths
from lib.io import load_jsonl, load_test_ids, to_xy
from lib.detectors import build_word_tfidf_lr, build_char_tfidf_lr
from lib.metrics import point_metrics
from lib.hardness import (
    hardness_readability, hardness_length, hardness_lexical,
    hardness_perplexity,
)

GPT2_MODEL = "gpt2"     # 124M params, CPU-friendly
MAX_LEN    = 1024       # GPT-2 context window
SEED       = 42


def compute_gpt2_perplexities(texts: list[str]) -> np.ndarray:
    """Return per-text GPT-2 perplexity (one float per input)."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    print(f"  loading {GPT2_MODEL}...")
    tok = GPT2TokenizerFast.from_pretrained(GPT2_MODEL)
    mdl = GPT2LMHeadModel.from_pretrained(GPT2_MODEL)
    mdl.eval()

    ppls = []
    with torch.no_grad():
        for i, t in enumerate(texts):
            enc = tok(t, return_tensors="pt", truncation=True, max_length=MAX_LEN)
            input_ids = enc["input_ids"]
            if input_ids.shape[1] < 2:
                ppls.append(float("nan"))
                continue
            # GPT-2: cross-entropy loss == mean negative-log-likelihood
            out = mdl(input_ids, labels=input_ids)
            loss = float(out.loss)              # mean NLL per token (nats)
            ppls.append(float(np.exp(loss)))    # perplexity
            if (i + 1) % 25 == 0:
                print(f"    [{i+1}/{len(texts)}]  last PPL = {ppls[-1]:.1f}")
    return np.array(ppls, dtype=float)


def kendall_tau(a: list[str], b: list[str]) -> tuple[float, float]:
    """Bucket-level Kendall tau on Easy/Medium/Hard label vectors."""
    from scipy.stats import kendalltau
    rank = {"Easy": 0, "Medium": 1, "Hard": 2}
    ra = [rank[x] for x in a]
    rb = [rank[x] for x in b]
    tau, p = kendalltau(ra, rb)
    return float(tau), float(p)


def evaluate_hard_buckets(
    p0_test: list,
    splits: dict,
    detectors: dict,
    hardness_obj,
    hardness_name: str,
) -> list:
    """Run the standard Easy/Medium/Hard F1 evaluation using a pre-built
    HardnessAssignment as the bucket source."""
    id_to_bucket = {r["id"]: b for r, b in zip(p0_test, hardness_obj.buckets)}

    rows = []
    for det_name, det in detectors.items():
        for split_name, split_rows in splits.items():
            id_to_row = {r["id"]: r for r in split_rows}
            aligned = [id_to_row[r["id"]] for r in p0_test if r["id"] in id_to_row]
            if len(aligned) < len(p0_test):
                continue
            X, y = to_xy(aligned)
            y_prob = det.predict_proba(X)
            y_pred = (y_prob >= 0.5).astype(int)

            for bucket in ["Easy", "Medium", "Hard"]:
                mask = np.array([id_to_bucket.get(r["id"], "") == bucket
                                 for r in p0_test if r["id"] in id_to_row])
                if mask.sum() < 3:
                    continue
                m = point_metrics(y[mask], y_pred[mask], y_prob[mask])
                rows.append({
                    "detector": det_name,
                    "hardness": hardness_name,
                    "split":    split_name,
                    "bucket":   bucket,
                    "n":        int(mask.sum()),
                    "f1":       round(m["f1"], 4),
                    "acc":      round(m["acc"], 4),
                })
    return rows


def main():
    out_dir = paths.RESULTS / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading P0 test rows...")
    test_ids = load_test_ids(paths.TEST_IDS)
    p0_all   = load_jsonl(paths.P0_PATH)
    p0_test  = [r for r in p0_all if r["id"] in test_ids]
    p0_texts = [r["text"] for r in p0_test]
    print(f"  P0_test: {len(p0_test)} paragraphs")

    print("[2/5] Computing GPT-2 perplexities (CPU)...")
    ppl_csv = out_dir / "perplexity_scores.csv"
    if ppl_csv.exists():
        print(f"  cached: {ppl_csv}")
        df_ppl = pd.read_csv(ppl_csv)
        # Align by id
        id_to_ppl = dict(zip(df_ppl["id"], df_ppl["perplexity"]))
        ppls = np.array([id_to_ppl[r["id"]] for r in p0_test])
    else:
        ppls = compute_gpt2_perplexities(p0_texts)
        df_ppl = pd.DataFrame({
            "id":         [r["id"] for r in p0_test],
            "label":      [r["label"] for r in p0_test],
            "perplexity": np.round(ppls, 3),
        })
        df_ppl.to_csv(ppl_csv, index=False)
        print(f"  saved: {ppl_csv}")

    # Sanity check: LLM text should have LOWER perplexity than human
    llm_ppl   = [p for p, r in zip(ppls, p0_test) if r["label"] == "llm"]
    human_ppl = [p for p, r in zip(ppls, p0_test) if r["label"] == "human"]
    print(f"  median PPL: human={np.median(human_ppl):.2f}  "
          f"llm={np.median(llm_ppl):.2f}  "
          f"(LLM should be lower if GPT-2 finds its register easier)")

    print("[3/5] Building hardness assignments...")
    ha_ppl_high = hardness_perplexity(ppls, high_is_hard=True)   # primary
    ha_ppl_low  = hardness_perplexity(ppls, high_is_hard=False)  # sanity check
    ha_fk       = hardness_readability(p0_texts)
    ha_len      = hardness_length(p0_texts)
    ha_ttr      = hardness_lexical(p0_texts)

    # Bucket distribution
    from collections import Counter
    print(f"  PPL high-is-hard buckets: {dict(Counter(ha_ppl_high.buckets))}")
    print(f"  PPL low-is-hard buckets:  {dict(Counter(ha_ppl_low.buckets))}")

    print("[4/5] Concordance vs other hardness definitions (Kendall tau)...")
    conc_rows = []
    for name, ha in [
        ("readability_fk", ha_fk),
        ("length", ha_len),
        ("ttr", ha_ttr),
    ]:
        tau_high, p_high = kendall_tau(ha_ppl_high.buckets, ha.buckets)
        tau_low,  p_low  = kendall_tau(ha_ppl_low.buckets,  ha.buckets)
        conc_rows.append({
            "vs":                name,
            "perplexity_high":   round(tau_high, 4),
            "perplexity_high_p": round(p_high, 4),
            "perplexity_low":    round(tau_low, 4),
            "perplexity_low_p":  round(p_low, 4),
        })
    df_conc = pd.DataFrame(conc_rows)
    df_conc.to_csv(out_dir / "perplexity_concordance.csv", index=False)
    print(df_conc.to_string(index=False))

    print("[5/5] Hard-bucket F1 under perplexity hardness...")
    train_ids  = load_test_ids(paths.TRAIN_IDS)
    train_rows = [r for r in p0_all if r["id"] in train_ids]
    X_tr, y_tr = to_xy(train_rows)
    detectors = {
        "word_tfidf_lr": build_word_tfidf_lr(X_tr, y_tr),
        "char_tfidf_lr": build_char_tfidf_lr(X_tr, y_tr),
    }

    splits = {
        "P0_test":            p0_test,
        "P1_test_simplified": load_jsonl(ROOT / "data" / "p1" / "p1_test_simplified.jsonl"),
        "P2_test_simplified": load_jsonl(ROOT / "data" / "p2" / "p2_test_simplified.jsonl"),
        "P1_test_standard":   load_jsonl(ROOT / "data" / "p1" / "p1_test.jsonl"),
        "P2_test_standard":   load_jsonl(ROOT / "data" / "p2" / "p2_test.jsonl"),
    }

    # Run both directions; reviewers will care about high-is-hard most
    all_rows = []
    all_rows += evaluate_hard_buckets(p0_test, splits, detectors,
                                       ha_ppl_high, "perplexity_high")
    all_rows += evaluate_hard_buckets(p0_test, splits, detectors,
                                       ha_ppl_low,  "perplexity_low")
    df_buckets = pd.DataFrame(all_rows)
    df_buckets.to_csv(out_dir / "perplexity_hardness_buckets.csv", index=False)

    # ── Summary printout ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("H2 PERPLEXITY HARDNESS — Hard-bucket F1 trajectories")
    print("=" * 72)
    for direction in ["perplexity_high", "perplexity_low"]:
        print(f"\n  Direction: {direction}")
        for det in ["char_tfidf_lr", "word_tfidf_lr"]:
            sub = df_buckets[
                (df_buckets["detector"] == det) &
                (df_buckets["hardness"] == direction) &
                (df_buckets["bucket"]   == "Hard")
            ].sort_values("split")
            print(f"    {det}:")
            for _, r in sub.iterrows():
                print(f"      {r['split']:<22} n={r['n']:>2}  F1={r['f1']:.4f}")

    print("\nDone. H2 complete.")


if __name__ == "__main__":
    main()
