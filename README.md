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

## Project Structure

```text
LLM_Text_Detection/
├── src/
│   ├── collect_human_wikipedia.py      # collect human-written paragraphs (Wikipedia)
│   ├── generate_llm_ollama.py          # generate LLM paragraphs locally (Ollama)
│   ├── build_dataset.py                # filter + per-class dedup + build P0 dataset
│   ├── repair_ids.py                   # fix duplicate IDs (if needed)
│   ├── make_splits.py                  # fixed train/val/test IDs
│   ├── train_detector.py               # baseline TF-IDF + Logistic Regression
│   ├── paraphrase_test_only.py         # create P1_test and P2_test (100 rows each)
│   ├── evaluate_robustness.py          # evaluate P0_test / P1_test / P2_test
│   ├── feature_drift.py                # compute shallow linguistic drift features
│   └── plot_robustness.py              # plot accuracy and F1 under paraphrasing
├── data/
│   ├── raw_human/
│   │   └── human.jsonl
│   ├── raw_llm/
│   │   └── llm.jsonl
│   ├── processed/
│   │   └── all.jsonl                   # optional combined view
│   ├── p0/
│   │   └── p0.jsonl                    # balanced 1000 rows
│   ├── p1/
│   │   └── p1_test.jsonl
│   ├── p2/
│   │   └── p2_test.jsonl
│   └── splits/
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
├── results/
│   ├── vectorizer.joblib               # saved TF-IDF vectorizer
│   ├── model.joblib                    # saved logistic regression model
│   ├── metrics_p0.json                 # baseline metrics on P0
│   ├── robustness_test.json            # robustness metrics on P0/P1/P2 (test-only)
│   ├── robustness_test.csv             # robustness metrics (CSV)
│   └── feature_drift_test.csv          # drift features across P0/P1/P2 (test-only)
├── figures/
│   ├── accuracy_vs_paraphrase_test.png
│   └── f1_vs_paraphrase_test.png
├── requirements.txt
└── README.md
