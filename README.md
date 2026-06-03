# Hardness-Aware Robustness of LLM Text Detection Under Iterative Paraphrasing

> **Mohammad Arifur Rahman** · rahman.arif.cse@gmail.com · Anderson University
> *Hardness-Aware Robustness of LLM-Text Detection Under Iterative Paraphrasing: A Mechanistic Analysis*

A fully local, end-to-end pipeline that studies **how detector robustness collapses under iterative paraphrasing**, and shows the collapse is **highly non-uniform across samples**. The headline contribution is methodological: aggregate metrics hide catastrophic local failure, and this failure persists even when *hardness* is defined by signals independent of the evaluated detector.

Paper draft: [`paper/main.tex`](paper/main.tex)

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

**Area Under Robustness Curve (F1-based, multi-seed):**

| Detector × Track | AURC |
|---|---|
| word TF-IDF × standard      | 0.821 ± 0.011 |
| word TF-IDF × simplified    | 0.756 ± 0.011 |
| char TF-IDF × standard      | 0.874 ± 0.009 |
| **char TF-IDF × simplified** | **0.649 ± 0.009** ← worst overall |
| RoBERTa-base × standard     | 0.849 [0.802, 0.893] |
| RoBERTa-base × simplified   | 0.908 [0.867, 0.950] ← most robust overall |

**Hard-bucket F1 collapse (Flesch-Kincaid hardness, simplified track):**

| Detector | P0 F1 | P2-sim F1 | Drop |
|---|---|---|---|
| **char TF-IDF** | 0.923 [0.909, 0.930] | 0.336 [0.308, 0.370] | **−0.587** ← catastrophic |
| word TF-IDF    | 0.882 [0.857, 0.905] | 0.681 [0.626, 0.714] | −0.201 |
| RoBERTa-base   | 1.000 [1.000, 1.000] | 0.920 [0.865, 0.995] | −0.080 ← robust |

> The same Hard samples (defined by text complexity alone — no detector involved) cause catastrophic collapse in char TF-IDF but barely affect RoBERTa. This detector-family asymmetry is invisible to aggregate metrics.

**Hard-bucket collapse across paraphrasers (char TF-IDF, P2):**

| Paraphraser | Hard F1 (char) | Hard F1 (word) |
|---|---|---|
| P0 (baseline)                 | 0.923 | 0.882 |
| Llama-3.1-8B (simplified)     | 0.336 | 0.681 |
| Mistral-7B (simplified)       | 0.647 | 0.737 |
| Mistral-7B (standard)         | 0.826 | 0.773 |
| NLLB-200 (en→fr→en)          | 0.483 | 0.703 |

> The collapse direction is consistent across all three paraphraser families. NLLB (no LLM, pure MT) alone drops char TF-IDF Hard F1 by 0.44, ruling out an LLM-family confound.

---

## Why this design

Five reviewer-level concerns drove the methodology:

1. **Hardness must not be tautological.** Margin-from-the-evaluated-detector is circular. We add **cross-detector margin** (Hard for word detector defined by char detector's margin) and **text-only** hardness (Flesch-Kincaid, word count, TTR). All show the collapse.

2. **n=100 test set is small.** We add **multi-seed training** (k=20 stratified train/val splits) and **paired bootstrap on differences** (2000 resamples), reporting both within-seed test-bootstrap CIs and across-seed training-variance CIs.

3. **A single significance test per stage is not enough.** Every Δ is reported with a paired-bootstrap 95% CI; we also report **AURC** (area under the degradation curve) as a scalar summary per (detector, track).

4. **The "paraphrase fallback" risk.** The paraphrase scripts silently fall back to the original text on LLM failure. We verified that **0/100 rows are unchanged in any paraphrase split**, and quantify lexical change with **token Jaccard**: standard track has 21% near-copies (Jaccard > 0.85) at P1; simplified track has 0% (median Jaccard = 0.45 at P1, 0.41 at P2).

5. **Test class composition is reported transparently.** The 80/10/10 random split (seed=42, no stratification) produced **38 human / 62 LLM** in the test set. All metrics are reported with this in mind; multi-seed stratification is applied to train/val. See [figures/fig07_class_imbalance_warning.png](figures/fig07_class_imbalance_warning.png).

---

## Key scientific findings

### F1 — Iterative paraphrasing is mostly single-shot.
Paired-bootstrap on Δ accuracy / F1 shows **P0 → P1 is significant** for both tracks and both detectors, but **P1 → P2 is NOT significant** on accuracy/F1 for either detector. Most of the damage happens at the first paraphrase round.

### F2 — Char n-gram detector is more robust to lexical paraphrasing, **less** robust to simplification.
Char detector AURC: 0.874 (standard) vs 0.649 (simplified) — a 0.225 gap.
Word detector AURC: 0.821 (standard) vs 0.756 (simplified) — only 0.065 gap.
Char n-grams capture morphological style → survive synonym swap → collapse when simplification changes vocabulary form.

### F3 — Detector-margin hardness and text-complexity hardness measure **different things**.
Kendall's τ between margin (any detector) and {Flesch-Kincaid, word count, TTR} is statistically zero (|τ| < 0.13, p > 0.05). But Word and Char detectors agree strongly on margin-hardness (τ = +0.58, p < 1e-17).

### F4 — The collapse is largest under detector-hardness, but persists under text-hardness.
Hard-bucket F1 drops under Flesch-Kincaid: char −0.587, word −0.201. Both are well outside bootstrap CIs and reproduce across all three text-only hardness definitions (readability, TTR, length).

### F5 — Calibration degrades significantly under paraphrasing.
ECE for word detector: 0.106 (P0) → 0.207 (P2 standard) → 0.199 (P2 simplified).
Paired-bootstrap Δ ECE for P0 → P2 standard: −0.122 [−0.168, −0.042], significant.

### F6 — Collapse profile is detector-family-specific.
Under simplified paraphrasing, Hard-bucket F1:
- Char TF-IDF: 0.923 → 0.336 (Δ = −0.587)
- RoBERTa-base: 1.000 → 0.920 (Δ = −0.080)

Char n-grams encode morphological surface patterns destroyed by vocabulary-form simplification. RoBERTa's contextual representations are invariant to the same perturbation.

### F7 — Semantic preservation is asymmetric across classes.
SBERT cosine similarity: LLM rows are preserved better than human rows at P2 simplified (Δ ≈ +0.05–0.08, Mann-Whitney p < 0.0001). All four paraphrase splits pass the semantic gate (median cosine ≥ 0.80). Evasion in the Hard bucket is not driven by semantic drift.

### F8 — Calibration does not transfer across paraphrase stages.
Temperature scaling (T_opt ≈ 0.16–0.18 for TF-IDF) corrects overconfidence at P0 but worsens ECE at P2-simplified by +0.12–0.13. Val-set-fitted calibration breaks under distribution shift. RoBERTa is stably miscalibrated (~0.37–0.38 ECE) across all stages — poorly calibrated everywhere but not degrading.

### F9 — All Hard-bucket flips are LLM-as-Human (pure recall failure).
Defining a *flip* as P0-correct → P2-wrong:
- char TF-IDF: **16 of 22 LLM Hard examples flip (73%)**; 0 of 12 human Hard examples flip.
- word TF-IDF: **8 of 22 LLM Hard examples flip (36%)**; 0 human flips.

The detector becomes *permissive*, not confused. This is a recall failure only, not a precision failure.

### F10 — Feature survival: 13% (char) / 26% (word).
Only 13.1% of char TF-IDF's top-20 features (by |β|) survive from P0 into P2-simplified in flip cases. Word TF-IDF loses 73.8%. Char TF-IDF's signal is 7× more surface-fragile than word TF-IDF's.

### F11 — SHAP feature set rotates almost entirely at P2.
P0 top char features: `' var'`, `'ludin'`, `'cludi'` — fragments of academic words (*variables, including*).
P2 top char features: `' comp'`, `' it'`, `'can'` — generic everyday n-grams.
The detector exploits a formal-register gap that simplified paraphrasing erases.

### F12 — Vocabulary homogenisation is the mechanism.
Mean word length (Hard bucket): Human P0=5.21 → P2-sim=4.76; LLM P0=5.36 → P2-sim=4.71.
After one round of simplified paraphrasing both groups converge to the same word length (~4.7 chars), eliminating the character-level formal-register gap the detector relied on.

---

## Dataset & pipeline

- **P0:** 500 Wikipedia paragraphs (human) + 500 Llama-3.1-8B paragraphs (LLM) on matched topics, 100–200 words each.
- **Train/Val/Test:** 800 / 100 / 100 with seed=42.
- **Paraphrase tracks (Llama-3.1-8B):**
  - **Standard:** Preserve meaning. Median P0↔P1 Jaccard = 0.76.
  - **Simplified:** Non-expert style, simple vocabulary. Median P0↔P1 Jaccard = 0.45.
- **Paraphraser diversity tracks:**
  - **Mistral-7B** (standard + simplified): Different LLM family.
  - **NLLB-200** (en→fr→en back-translation): No LLM, pure MT system.
- **Iteration:** Two rounds per track (P1, P2).

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
│   ├── run_semantic_preservation.py           # SBERT semantic gate (P7)
│   ├── run_calibration_depth.py              # temperature scaling + per-bucket ECE (P5)
│   ├── evaluate_nllb_track.py                # NLLB back-translation evaluation (P6)
│   ├── evaluate_mistral_track.py             # Mistral-7B evaluation (P6)
│   ├── run_mechanistic_analysis.py           # SHAP attribution + flip analysis (P8)
│   ├── integrate_neural_predictions.py       # merge RoBERTa Colab results
│   ├── make_figures.py                        # figures 01–07
│   └── make_multiseed_figures.py              # figures 08–10
├── paraphrase_test_nllb.py                    # NLLB back-translation paraphrase generator
├── paraphrase_test_mistral.py                 # Mistral-7B paraphrase generator (Ollama)
└── diagnose_paraphrase_quality.py             # paraphrase-contamination and Jaccard diagnostic

data/
├── p0/p0.jsonl                                # 1000 rows
├── p1/{p1_test, p1_test_simplified,
│       p1_test_nllb,
│       p1_test_mistral, p1_test_mistral_simplified}.jsonl
├── p2/{p2_test, p2_test_simplified,
│       p2_test_nllb,
│       p2_test_mistral, p2_test_mistral_simplified}.jsonl
└── splits/{train,val,test}_ids.txt            # fixed reference split

results/eval/
├── multiseed_summary_{flat,buckets,aurc}.csv  # 20-seed results
├── roberta_summary_{flat,buckets,aurc}.csv    # RoBERTa 3-seed results
├── calibration_{temperature,buckets,summary}.csv  # P5 calibration
├── nllb_{metrics_flat,hardness_buckets,aurc}.csv  # P6 NLLB
├── mistral_{metrics_flat,hardness_buckets,aurc}.csv  # P6 Mistral
├── paraphraser_comparison_full.csv            # 3-paraphraser Hard-bucket comparison
├── mechanistic_{flip_cases,shap_summary,feature_drift,vocab_richness}.csv  # P8
└── semantic_{preservation,summary}.csv        # P7 semantic gate

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
├── fig12_three_detector_robustness.png        # word + char + RoBERTa overlay
├── fig13_universal_hardness_collapse.png      # Hard-bucket collapse on all 3 detectors
├── fig14_nllb_robustness.png                  # NLLB track trajectory
├── fig15_paraphraser_comparison.png           # Llama vs NLLB Hard-bucket F1
├── fig16_mistral_robustness.png               # Mistral standard + simplified tracks
├── fig17_three_paraphraser_comparison.png     # 3-paraphraser Hard-bucket comparison
├── fig18_temperature_scaling.png              # reliability curves before/after T-scaling
├── fig19_bucket_ece.png                       # per-bucket ECE (Easy/Medium/Hard)
├── fig20_shap_top_features.png                # top SHAP features P0 vs P2-simplified
├── fig21_feature_survival.png                 # feature survival rate in flip cases
└── fig22_vocab_drift.png                      # vocabulary richness drift (TTR, word length)

colab/
├── roberta_detector_finetune.ipynb            # self-contained Colab notebook
├── COLAB_INSTRUCTIONS.md                      # step-by-step guide
└── bundle/                                    # 8 files to upload to Colab in one go

paper/
└── main.tex                                   # LaTeX paper draft (submission-ready)
```

---

## Reproducing the full pipeline

```powershell
# 0. Setup
python -m venv .venv312
.venv312\Scripts\Activate.ps1
pip install -U pip && pip install -r requirements.txt

# 1. (if data not already present) regenerate from scratch
python src/collect_human_wikipedia.py
python src/generate_llm_ollama.py
python src/build_dataset.py && python src/repair_ids.py && python src/make_splits.py
python src/paraphrase_test_only.py
python src/generate_paraphrase_test_simplified.py

# 2. Diagnostic — verify no contamination, log Jaccard
$env:PYTHONIOENCODING="utf-8"
python src/diagnose_paraphrase_quality.py

# 3. Core eval: paired-bootstrap + AURC + 6-way hardness (single seed)
python src/experiments/run_evaluation.py

# 4. Multi-seed (20 resamples; ~30s on CPU)
python src/experiments/run_multiseed.py
python src/experiments/make_figures.py
python src/experiments/make_multiseed_figures.py

# 5. Semantic preservation gate (SBERT; ~2 min first run)
python src/experiments/run_semantic_preservation.py

# 6. Neural detector — RoBERTa-base (3 seeds via Colab T4 GPU)
#    See colab/COLAB_INSTRUCTIONS.md, then:
python src/experiments/integrate_neural_predictions.py

# 7. Paraphraser diversity — NLLB back-translation (CPU, ~10 min)
python src/paraphrase_test_nllb.py
python src/experiments/evaluate_nllb_track.py

# 8. Paraphraser diversity — Mistral-7B (requires Ollama: ollama pull mistral)
python src/paraphrase_test_mistral.py
python src/experiments/evaluate_mistral_track.py

# 9. Calibration depth — temperature scaling + per-bucket ECE
python src/experiments/run_calibration_depth.py

# 10. Mechanistic analysis — SHAP attribution + flip cases + vocab drift
python src/experiments/run_mechanistic_analysis.py
```

All outputs are deterministic given the seed.

---

## Known limitations and follow-ups

| Limitation | Status |
|---|---|
| Single generator+paraphraser (Llama-3.1-8B) — potential confound | **DONE (P6).** Reproduced under Mistral-7B (different LLM family) and NLLB-200 (no LLM). Collapse direction is consistent across all three paraphrasers. See fig14–fig17. |
| Semantic gate needed — LLM text may be paraphrased less aggressively | **DONE (P7).** SBERT cosine gate: all splits pass (median ≥ 0.80). Asymmetry confirmed (LLM preserved better than human, Δ ≈ +0.05, p < 0.0001) but does not explain the Hard-bucket collapse. |
| Neural detector — is collapse detector-family-specific? | **DONE (P2).** RoBERTa-base (3 seeds, Colab T4). Hard-bucket F1: char TF-IDF 0.923→0.336 (Δ=−0.587), RoBERTa 1.000→0.920 (Δ=−0.080). Collapse is real but magnitude is strongly family-dependent. See fig12, fig13. |
| Calibration not corrected | **DONE (P5).** Temperature scaling (T_opt ≈ 0.16–0.18) fitted on val set. Corrects P0 ECE but worsens P2-simplified ECE by +0.12–0.13. Calibration does not transfer across distribution shift. RoBERTa stably miscalibrated everywhere. See fig18, fig19. |
| No mechanistic explanation of *why* Hard examples flip | **DONE (P8).** SHAP attribution shows 87% of char TF-IDF's top features disappear in flip cases. All flips are LLM-as-Human (pure recall failure). Vocabulary homogenisation (mean word length LLM: 5.36→4.71) erases the formal-register gap. See fig20–fig22. |
| Test n=100, fixed | Mitigated by 20-seed multi-seed framework. Full mitigation requires larger paraphrase pool. |
| Single domain (Wikipedia) | OPEN. News (XSum) or essays (Brown) sanity check at n=200 is the cheapest external-validity move. |
| Cross-generator evaluation | OPEN. Train on Llama, test on GPT-4 outputs — not yet done. |
| No external-LLM perplexity hardness | OPEN. GPT-2 PPL as fourth non-circular hardness signal. |

---

## Position vs prior work

- **Krishna et al. (2023), DIPPER** — adversarial paraphrasing for detector evasion. We use off-the-shelf LLMs under natural prompts and stratify by hardness rather than adversarially optimising the paraphraser.
- **Mitchell et al. (2023), DetectGPT** and **Fast-DetectGPT (2024)** — zero-shot detection via log-probability curvature. Complementary; we evaluate fine-tuned and shallow detectors.
- **Krishna et al. (2023), Bao et al. (2024)** — detector robustness benchmarks. None report hardness-stratified robustness. Our central argument is that they should.

---

## Citation

```bibtex
@article{rahman2026hardness,
  title   = {Hardness-Aware Robustness of LLM-Text Detection Under Iterative Paraphrasing: A Mechanistic Analysis},
  author  = {Rahman, Mohammad Arifur},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## License

MIT.
