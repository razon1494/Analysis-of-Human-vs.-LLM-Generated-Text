# LLM Text Detection Robustness Under Iterative Paraphrasing (Local, No API Keys)

This project builds an end-to-end pipeline to study how well a classical detector can distinguish **human-written** vs **LLM-generated** text under **iterative paraphrasing**. The main result demonstrates *signature erosion*: a detector trained on original text degrades as paraphrasing increases.

---

## Highlights

- **Balanced dataset:** 500 human + 500 LLM samples (100–200 words)
- **Baseline detector:** TF-IDF (1–2 grams) + Logistic Regression
- **Robustness test:** Evaluate on the **same fixed test set** under:
  - **P0_test:** original
  - **P1_test:** paraphrased once
  - **P2_test:** paraphrased twice
- **Key result (test set):** Accuracy drops **0.86 → 0.77 → 0.75** under iterative paraphrasing

---

## Results

### Baseline (P0 test)
- **Accuracy:** 0.8600  
- **F1:** 0.8833  

### Robustness Under Iterative Paraphrasing (Test Only)

| Condition | n | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| P0_test | 100 | 0.8600 | 0.9138 | 0.8548 | 0.8833 |
| P1_test | 100 | 0.7700 | 0.7826 | 0.8710 | 0.8244 |
| P2_test | 100 | 0.7500 | 0.7534 | 0.8871 | 0.8148 |

Saved:
- `results/robustness_test.json`
- `results/robustness_test.csv`

Plots:
- `figures/accuracy_vs_paraphrase_test.png`
- `figures/f1_vs_paraphrase_test.png`

---

## Linguistic Drift (Test Only)

Paraphrasing causes measurable shifts in shallow style signals:

| split | n | words | sents | words_per_sent | ttr | punct_rate | uniq_word_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0_test | 100 | 147.13 | 6.68 | 22.90 | 0.656 | 0.01836 | 0.656 |
| P1_test | 100 | 131.04 | 5.15 | 26.14 | 0.720 | 0.01779 | 0.720 |
| P2_test | 100 | 129.21 | 4.74 | 27.82 | 0.732 | 0.01697 | 0.732 |

Interpretation:
- **Lexical diversity increases** (TTR rises): 0.656 → 0.720 → 0.732  
- **Fewer but longer sentences**: words/sentence 22.9 → 26.1 → 27.8  
- **Punctuation rate slightly decreases**  

Saved:
- `results/feature_drift_test.csv`

---

## Why This Matters

Many detectors pick up brittle surface patterns. Under paraphrasing (a realistic distribution shift), those patterns drift and detection performance drops. This project quantifies that erosion and provides a reproducible pipeline to study robustness.

---

## Tech Stack

- Python 3.11+
- scikit-learn (TF-IDF + Logistic Regression)
- Ollama (local LLM generation + paraphrasing; no API keys)
- matplotlib (plots)

---
## Project Structure

LLM_Text_Detection/
├─ src/
│ ├─ collect_human_wikipedia.py # collect human-written paragraphs (Wikipedia)
│ ├─ generate_llm_ollama.py # generate LLM paragraphs locally (Ollama)
│ ├─ build_dataset.py # filter + per-class dedup + build P0 dataset
│ ├─ repair_ids.py # fix duplicate IDs (if needed)
│ ├─ make_splits.py # fixed train/val/test IDs
│ ├─ train_detector.py # baseline TF-IDF + Logistic Regression
│ ├─ paraphrase_test_only.py # create P1_test and P2_test (100 rows each)
│ ├─ evaluate_robustness.py # evaluate P0_test / P1_test / P2_test
│ ├─ feature_drift.py # compute shallow linguistic drift features
│ └─ plot_robustness.py # plot accuracy and F1 under paraphrasing
├─ data/
│ ├─ raw_human/ # human.jsonl
│ ├─ raw_llm/ # llm.jsonl
│ ├─ processed/ # all.jsonl (optional combined view)
│ ├─ p0/ # p0.jsonl (balanced 1000 rows)
│ ├─ p1/ # p1_test.jsonl
│ ├─ p2/ # p2_test.jsonl
│ └─ splits/ # train_ids.txt, val_ids.txt, test_ids.txt
├─ results/
│ ├─ vectorizer.joblib # saved TF-IDF vectorizer
│ ├─ model.joblib # saved logistic regression model
│ ├─ metrics_p0.json # baseline metrics on P0
│ ├─ robustness_test.json # robustness metrics on P0/P1/P2 (test-only)
│ ├─ robustness_test.csv # robustness metrics (CSV)
│ └─ feature_drift_test.csv # drift features across P0/P1/P2 (test-only)
├─ figures/
│ ├─ accuracy_vs_paraphrase_test.png
│ └─ f1_vs_paraphrase_test.png
├─ requirements.txt
└─ README.md


---

## Setup

### 1) Create a virtual environment
**Windows (PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

2) Install & verify Ollama (local)

Pull the model used in this project:

ollama pull llama3.1:8b


Verify the Ollama API is available:

curl http://localhost:11434/api/tags

Reproduce the Pipeline (End-to-End)
Step A — Collect human text (Wikipedia)

Collects 500 human-written paragraphs (100–200 words) and saves them to JSONL.

python src/collect_human_wikipedia.py


Output:

data/raw_human/human.jsonl

Step B — Generate LLM text (Ollama)

Generates 500 LLM paragraphs locally using Ollama (matching the same length constraints).

python src/generate_llm_ollama.py


Output:

data/raw_llm/llm.jsonl

Step C — Build balanced dataset (P0)

Filters 100–200 words and deduplicates within each class (human vs llm), then builds the P0 dataset.

python src/build_dataset.py


Outputs:

data/p0/p0.jsonl (1000 rows total: 500 human + 500 llm)

data/processed/all.jsonl (optional combined output)

If you ever see duplicate-ID split overlap issues, run this once:

python src/repair_ids.py


Then rename p0_fixed.jsonl → p0.jsonl and regenerate splits.

Step D — Create fixed train/val/test splits (by ID)

Creates stable split lists to avoid leakage.

python src/make_splits.py


Outputs:

data/splits/train_ids.txt (800)

data/splits/val_ids.txt (100)

data/splits/test_ids.txt (100)

Step E — Train baseline detector (TF-IDF + Logistic Regression)

Trains on P0 train split, evaluates on val/test, and saves the model.

python src/train_detector.py


Outputs:

results/vectorizer.joblib

results/model.joblib

results/metrics_p0.json

Step F — Create paraphrased test sets (P1_test and P2_test)

Paraphrases only the test set (100 samples) to quickly measure robustness.

P1_test: paraphrased once

P2_test: paraphrased twice (iterative paraphrasing)

python src/paraphrase_test_only.py


Outputs:

data/p1/p1_test.jsonl

data/p2/p2_test.jsonl

Step G — Evaluate robustness on P0_test vs P1_test vs P2_test

Runs the saved P0 detector on all three test conditions.

python src/evaluate_robustness.py


Outputs:

results/robustness_test.json

results/robustness_test.csv

Step H — Linguistic drift features + plots

Computes shallow drift features and generates the robustness plots.

python src/feature_drift.py
python src/plot_robustness.py


Outputs:

results/feature_drift_test.csv

figures/accuracy_vs_paraphrase_test.png

figures/f1_vs_paraphrase_test.png

Notes / Limitations

This project focuses on robustness under paraphrasing, not on building the best detector possible.

Paraphrases are generated locally using Ollama; different local models may produce different degradation magnitudes.

Easy extensions:

paraphrase full dataset (P1/P2 for train+test)

add a non-LLM paraphrase method (e.g., back-translation)

compare against neural detectors

