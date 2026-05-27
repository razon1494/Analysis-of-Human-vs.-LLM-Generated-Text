# Hardness-Aware Robustness of LLM Text Detection Under Iterative Paraphrasing

> **Mohammad Arifur Rahman** · rahman.arif.cse@gmail.com
> *Hardness-Aware Robustness of LLM-Text Detection Under Iterative Paraphrasing*

A fully local, end-to-end pipeline that studies **how detector robustness collapses under iterative paraphrasing**, and shows the collapse is **highly non-uniform across samples**. The headline contribution is methodological: aggregate metrics hide catastrophic local failure, and this failure persists even when *hardness* is defined by signals independent of the evaluated detector.

---

## Headline (multi-seed, n_seeds=20, 95% across-seed percentile interval)

**Aggregate accuracy collapses under simplified paraphrasing:**

| Stage | word TF-IDF Acc | char TF-IDF Acc |
|---|---|---|
| P0 (original)        | 0.836 [0.820, 0.855] | 0.940 [0.930, 0.950] |
| P1 standard          | 0.775 [0.755, 0.805] | 0.793 [0.770, 0.820] |
| P2 standard          | 0.764 [0.745, 0.796] | 0.762 [0.740, 0.790] |
| P1 simplified        | 0.725 [0.700, 0.755] | 0.612 [0.585, 0.640] |
| P2 simplified        | 0.704 [0.685, 0.720] | 0.572 [0.550, 0.600] |

**Area Under Robustness Curve (accuracy):**

| Detector × Track | AURC |
|---|---|
| word TF-IDF × standard      | 0.788 [0.771, 0.812] |
| word TF-IDF × simplified    | 0.747 [0.731, 0.761] |
| char TF-IDF × standard      | 0.835 [0.820, 0.858] |
| **char TF-IDF × simplified** | **0.691 [0.685, 0.703]** ← worst overall |
| RoBERTa-base × standard     | 0.849 [0.803, 0.893] |
| RoBERTa-base × simplified   | 0.908 [0.867, 0.950] ← most robust overall |

**Hard-bucket F1 collapse persists under non-circular hardness (word TF-IDF detector):**

| Hardness definition | P0 F1 | P2 simplified F1 | Drop |
|---|---|---|---|
| margin: word self (CIRCULAR baseline)  | 0.654 [0.61, 0.73] | 0.213 [0.12, 0.25] | −0.441 |
| margin: cross-detector (NON-CIRCULAR)  | 0.640 [0.59, 0.72] | 0.269 [0.13, 0.38] | **−0.371** |
| Flesch-Kincaid grade (text-only)       | 0.882 [0.86, 0.90] | 0.681 [0.63, 0.71] | −0.201 |
| Type-Token Ratio (text-only)           | 0.741 [0.72, 0.78] | 0.648 [0.61, 0.68] | −0.093 |
| Word count (text-only)                 | 0.419 [0.36, 0.55] | 0.325 [0.29, 0.37] | −0.094 |

> Hard-bucket collapse holds (−0.371 F1) when "Hard" is defined by a *different* detector. The paper does not rest on a circular definition.

**Collapse profile is detector-family-specific (Flesch-Kincaid hardness, Hard bucket, multi-seed):**

| Detector | P0 F1 | P2 simplified F1 | Drop |
|---|---|---|---|
| **char TF-IDF** | 0.923 [0.909, 0.930] | 0.336 [0.308, 0.370] | **−0.587** ← catastrophic |
| word TF-IDF    | 0.882 [0.857, 0.905] | 0.681 [0.626, 0.714] | −0.201 |
| RoBERTa-base   | 1.000 [1.000, 1.000] | 0.920 [0.865, 0.995] | −0.080 ← robust |

> The same Hard samples (defined by text complexity alone — no detector involved) cause catastrophic collapse in char TF-IDF but barely affect RoBERTa. This detector-family asymmetry is invisible to aggregate metrics.

---

## Why this design

Five reviewer-level concerns drove the methodology:

1. **Hardness must not be tautological.** Margin-from-the-evaluated-detector is circular. We add **cross-detector margin** (Hard for word detector defined by char detector's margin) and **text-only** hardness (Flesch-Kincaid, word count, TTR). All show the collapse.

2. **n=100 test set is small.** We add **multi-seed training** (k=20 stratified train/val splits) and **paired bootstrap on differences** (2000 resamples), reporting both within-seed test-bootstrap CIs and across-seed training-variance CIs.

3. **A single significance test per stage is not enough.** Every Δ is reported with a paired-bootstrap 95% CI; we also report **AURC** (area under the degradation curve) as a scalar summary per (detector, track).

4. **The "paraphrase fallback" risk in the prior pipeline.** The paraphrase scripts silently fall back to the original text on LLM failure. We verified that **0/100 rows are unchanged in any paraphrase split**, and quantify lexical change with **token Jaccard**: standard track has 21% near-copies (Jaccard > 0.85) at P1; simplified track has 0% (median Jaccard = 0.45 at P1, 0.41 at P2).

5. **Test class composition is reported transparently.** The 80/10/10 random split (seed=42, no stratification) produced **38 human / 62 LLM** in the test set. All metrics are reported with this in mind; multi-seed stratification is applied to train/val. See [figures/fig07_class_imbalance_warning.png](figures/fig07_class_imbalance_warning.png).

---

## Key scientific findings

### F1 — Iterative paraphrasing is mostly single-shot.

Paired-bootstrap on Δ accuracy / F1 shows **P0 → P1 is significant** for both tracks and both detectors, but **P1 → P2 is NOT significant** on accuracy/F1 for either detector. Most of the damage happens at the first paraphrase round. The "iterative" framing in the paper title needs caveating: iteration plateaus quickly.

### F2 — Char n-gram detector is more robust to lexical paraphrasing, **less** robust to simplification.

Char detector AURC: 0.835 (standard) vs 0.691 (simplified) — a 0.144 gap.
Word detector AURC: 0.788 (standard) vs 0.747 (simplified) — only 0.041 gap.

Char n-grams capture morphological style → survive synonym swap → collapse when simplification changes vocabulary form. Word n-grams care about token identity → harder to retain under any paraphrase. Contradicts the common claim that char n-grams are uniformly more robust.

### F3 — Detector-margin hardness and text-complexity hardness measure **different things**.

Kendall's τ between margin (any detector) and {Flesch-Kincaid, word count, TTR} is statistically zero (|τ| < 0.1, p > 0.05). But Word and Char detectors agree strongly on margin-hardness (τ = +0.58, p < 1e-17). There is a *shared intrinsic detector hardness* that is not captured by surface text complexity.

### F4 — The collapse is largest under detector-hardness, but persists under text-hardness.

Hard-bucket F1 drops:
- under margin_self_word: −0.441 (circular)
- under margin_cross: −0.371 (non-circular detector signal)
- under readability_fk: −0.201 (pure text signal)
- under TTR: −0.093 (pure text signal)

Two distinct vulnerability regimes:
- *Detector-defined hardness* identifies samples where the **classifier** is uncertain — these collapse most.
- *Text-defined hardness* identifies **inherently harder-to-classify** prose (more complex, denser) — these also collapse but less.

### F5 — Calibration degrades significantly under paraphrasing.

ECE for word detector: 0.106 (P0) → 0.207 (P2 standard) → 0.199 (P2 simplified).
Paired-bootstrap Δ ECE for P0 → P2 standard: −0.122 [−0.168, −0.042], significant.

Suggests a second, separable contribution: **paraphrasing damages calibration, not just accuracy**.

### F6 — The collapse profile is detector-family-specific under text-complexity hardness.

Under simplified paraphrasing, char TF-IDF drops from F1=0.95 to 0.57 (aggregate); RoBERTa drops from 0.99 to 0.92. The aggregate gap looks modest. But stratified by Flesch-Kincaid hardness:

- Char TF-IDF Hard bucket: F1 0.923 → 0.336 (Δ = −0.587)
- RoBERTa Hard bucket: F1 1.000 → 0.920 (Δ = −0.080)

Char n-grams encode morphological surface patterns destroyed by vocabulary-form simplification. RoBERTa's contextual representations are invariant to the same perturbation. This directly contradicts the common assumption that char n-grams are uniformly more robust than neural detectors — they are more robust to synonym swap (standard track) but catastrophically less robust to simplification.

The methodology contribution is validated by this finding: **aggregate AURC (0.691 vs 0.908) hints at the gap, but only hardness stratification reveals its catastrophic, selective nature.**

### F7 — Semantic preservation is asymmetric across classes.

SBERT cosine similarity (all-mpnet-base-v2) between P0 and paraphrased text: LLM rows are preserved better than human rows at P2 simplified (Δ ≈ +0.05–0.08, Mann-Whitney p < 0.0001). All four paraphrase splits pass the semantic gate (median cosine ≥ 0.80, < 10% flagged below 0.70). The evasion observed in the Hard bucket is not driven by semantic drift — LLM samples are paraphrased less aggressively yet still evade detection.

---

## Dataset & pipeline

- **P0:** 500 Wikipedia paragraphs (human) + 500 Llama-3.1-8B paragraphs (LLM) on matched topics, 100–200 words each.
- **Train/Val/Test (single-seed for paraphrasing):** 800 / 100 / 100 with seed=42.
  - Multi-seed framework uses 20 stratified resamplings of train/val (test fixed because paraphrases are generated for these 100 rows only).
- **Paraphrase tracks:**
  - **Standard:** "Paraphrase preserving meaning and facts." Llama-3.1-8B, temperature=0.2. Median P0↔P1 Jaccard = 0.76 (relatively mild).
  - **Simplified:** "Rewrite in simplified, non-expert style." Llama-3.1-8B, temperature=0.2. Median P0↔P1 Jaccard = 0.45 (substantially more aggressive).
- **Iteration:** Two rounds per track (P1, P2). P2 of each track is P1 paraphrased again.

---

## Repository structure

```
src/
├── lib/                                       # shared library
│   ├── io.py                                  # JSONL loaders, label coding
│   ├── detectors.py                           # word- and char-TFIDF + LR factories
│   ├── metrics.py                             # bootstrap, paired-bootstrap, AURC, ECE
│   ├── hardness.py                            # margin / cross-margin / readability / length / TTR
│   └── paths.py                               # centralized paths
├── experiments/
│   ├── run_evaluation.py                      # main eval: point + CI + paired-diff + AURC + buckets
│   ├── run_multiseed.py                       # 20-seed training-variance experiment
│   ├── make_figures.py                        # figures 01–07
│   └── make_multiseed_figures.py              # figures 08–10
├── diagnose_paraphrase_quality.py             # paraphrase-contamination and Jaccard diagnostic
├── (legacy single-purpose scripts kept for reproducibility)
data/
├── p0/p0.jsonl                                # 1000 rows
├── p1/{p1_test, p1_test_simplified}.jsonl     # 100 rows × 2 tracks
├── p2/{p2_test, p2_test_simplified}.jsonl     # 100 rows × 2 tracks
└── splits/{train,val,test}_ids.txt            # fixed reference split
results/
├── eval/                                      # all multi-seed and paired-bootstrap outputs
└── (legacy artifacts kept for backward compat)
figures/
├── fig01_aggregate_robustness.png             # acc+F1 trajectory with bootstrap CIs
├── fig02_hardness_trajectory.png              # per-bucket F1 across stages, 6 hardness defs
├── fig03_hardness_concordance_heatmap.png     # Kendall's tau between hardness definitions
├── fig04_aurc_bars.png                        # AURC summary
├── fig05_paired_diff_forest.png               # forest plot of paired-bootstrap deltas
├── fig06_calibration_reliability.png          # reliability diagrams per stage
├── fig07_class_imbalance_warning.png          # 38/62 test split disclosure
├── fig08_multiseed_robustness.png             # acc/F1/AUROC with across-seed CIs
├── fig09_multiseed_hardness_grid.png          # bucket F1 across hardness × stage × detector × seed
├── fig10_multiseed_aurc_bars.png              # AURC with across-seed CIs
├── fig11_semantic_preservation.png            # SBERT cosine distributions (passes gate)
├── fig12_three_detector_robustness.png        # word + char + RoBERTa overlay (after Colab)
└── fig13_universal_hardness_collapse.png      # Hard-bucket collapse on all 3 detectors

colab/
├── roberta_detector_finetune.ipynb            # self-contained Colab notebook
├── COLAB_INSTRUCTIONS.md                      # step-by-step guide
└── bundle/                                    # 8 files to upload to Colab in one go
```

---

## Reproducing the multi-seed pipeline

```powershell
# 1. fresh venv (Windows; Linux/macOS use python -m venv .venv && source bin/activate)
python -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt

# 2. (if data not already present) regenerate from scratch
python src/collect_human_wikipedia.py
python src/generate_llm_ollama.py
python src/build_dataset.py
python src/repair_ids.py
python src/make_splits.py
python src/paraphrase_test_only.py
python src/generate_paraphrase_test_simplified.py

# 3. diagnostic — verify no paraphrase contamination, log Jaccard distribution
python src/diagnose_paraphrase_quality.py

# 4. core: paired-bootstrap + AURC + 6-way hardness analysis (single seed)
$env:PYTHONIOENCODING="utf-8"
python src/experiments/run_evaluation.py

# 5. multi-seed (20 stratified resamples of train/val; ~30s on CPU)
python src/experiments/run_multiseed.py

# 6. figures
python src/experiments/make_figures.py
python src/experiments/make_multiseed_figures.py

# 7. semantic preservation gate (SBERT, CPU; ~2 min first run downloads model)
python src/experiments/run_semantic_preservation.py

# 8. (optional) neural detector via Colab — see colab/COLAB_INSTRUCTIONS.md
# After downloading roberta_predictions.csv into results/eval/:
python src/experiments/integrate_neural_predictions.py
```

All outputs are deterministic given the seed.

---

## Known limitations and follow-ups

| Limitation | Mitigation status |
|---|---|
| Single generator+paraphraser (Llama-3.1-8B) — confound | OPEN. Adding Mistral or NLLB back-translation is the next priority. |
| Paraphraser preserves LLM-text semantics more than human-text (SBERT cosine Δ ≈ +0.05; Mann-Whitney p < 0.0001) | OPEN. Reported in [results/eval/semantic_summary.csv](results/eval/semantic_summary.csv); needs follow-up paraphraser-control study. |
| Neural detector complete — collapse is detector-family-specific | DONE. RoBERTa-base fine-tuned (3 seeds, Colab T4). Hard-bucket F1 under readability_fk: char TF-IDF 0.923→0.336 (Δ=−0.587), RoBERTa 1.000→0.920 (Δ=−0.080). Universality claim revised: collapse is real across all detectors but magnitude is family-specific. See fig12, fig13. |
| Test n=100, fixed | Mitigated by multi-seed; full mitigation requires regenerating paraphrases on a larger pool. |
| No external-LLM perplexity hardness | OPEN. GPT-2 PPL as fourth non-circular hardness signal. |
| Wikipedia-only domain | OPEN. News (XSum) or essays (Brown) sanity check at n=200 each is the cheapest external-validity move. |
| Calibration not yet corrected (no temperature scaling) | OPEN. Affects whether margin == confidence in the strict sense. |
| Single-track paraphrase quality unequal | Mitigated: Jaccard analysis exposes "standard track" as a mild perturbation (21% near-copy at P1); simplified is the real stress test. |

---

## Position vs prior work

This work sits adjacent to:
- **Krishna et al. (2023), DIPPER** — adversarial paraphrasing for detector evasion. We do not adversarially train a paraphraser; instead we use an off-the-shelf LLM under two natural prompts and stratify by detector hardness.
- **Mitchell et al. (2023), DetectGPT** and **Fast-DetectGPT (2024)** — zero-shot detection via curvature in model log-probabilities. Complementary; we evaluate fine-tuned and shallow detectors but plan to add Fast-DetectGPT as a non-trained baseline.
- **Krishna et al. (2023), Bao et al. (2024)** — detector robustness benchmarks. None report hardness-stratified robustness as a standard. Our central methodological argument is that they should.

---

## Citation

```bibtex
@article{rahman2026hardness,
  title   = {Hardness-Aware Robustness of LLM-Text Detection Under Iterative Paraphrasing},
  author  = {Rahman, Mohammad Arifur},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT.
