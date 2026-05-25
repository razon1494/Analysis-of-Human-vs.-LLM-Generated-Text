#!/usr/bin/env bash
# =============================================================================
# run_all.sh  —  Full pipeline reproduction script
# =============================================================================
# Usage:
#   bash run_all.sh              # run everything end-to-end
#   bash run_all.sh --from-step 5   # resume from step N (skip data collection)
#
# Steps:
#   1  collect_human_wikipedia.py
#   2  generate_llm_ollama.py
#   3  build_dataset.py
#   4  repair_ids.py
#   5  make_splits.py
#   6  train_detector.py
#   7  train_char_ngram_detector.py     [NEW]
#   8  paraphrase_test_only.py
#   9  generate_paraphrase_test_simplified.py
#   10 evaluate_robustness.py
#   11 evaluate_robustness_test_dualtrack.py
#   12 feature_drift.py
#   13 feature_drift_extended.py       [NEW]
#   14 evaluate_hardness_buckets.py
#   15 evaluate_auroc.py               [NEW]
#   16 evaluate_bootstrap_ci.py        [NEW]
#   17 analyze_top_features.py         [NEW]
#   18 error_analysis.py               [NEW]
#   19 plot_robustness.py
#   20 plot_robustness_test_dualtrack.py
#   21 plot_hardness_buckets.py
# =============================================================================

set -euo pipefail

PYTHON=${PYTHON:-python}
SRC=src
FROM_STEP=1

# Parse --from-step argument
while [[ $# -gt 0 ]]; do
  case $1 in
    --from-step) FROM_STEP="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

run_step() {
  local step=$1
  local script=$2
  local desc=$3

  if [[ $step -lt $FROM_STEP ]]; then
    echo "  [SKIP] Step $step: $desc"
    return
  fi

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Step $step / 21: $desc"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  $PYTHON $SRC/$script
}

echo "============================================================"
echo "  LLM Text Detection Robustness — Full Pipeline"
echo "  Starting from step $FROM_STEP"
echo "============================================================"

# ── Data collection (requires Ollama + internet) ──────────────────────────────
run_step  1 collect_human_wikipedia.py     "Collect Wikipedia paragraphs"
run_step  2 generate_llm_ollama.py         "Generate LLM paragraphs (Ollama)"

# ── Dataset preparation ───────────────────────────────────────────────────────
run_step  3 build_dataset.py               "Build balanced P0 dataset"
run_step  4 repair_ids.py                  "Repair duplicate IDs (if any)"
run_step  5 make_splits.py                 "Create train/val/test splits"

# ── Training ──────────────────────────────────────────────────────────────────
run_step  6 train_detector.py              "Train word n-gram detector"
run_step  7 train_char_ngram_detector.py   "Train char n-gram detector [NEW]"

# ── Paraphrase generation ─────────────────────────────────────────────────────
run_step  8 paraphrase_test_only.py                   "Generate P1/P2 standard paraphrases"
run_step  9 generate_paraphrase_test_simplified.py    "Generate P1/P2 simplified paraphrases"

# ── Evaluation ────────────────────────────────────────────────────────────────
run_step 10 evaluate_robustness.py                    "Evaluate robustness (standard)"
run_step 11 evaluate_robustness_test_dualtrack.py     "Evaluate robustness (dual-track)"
run_step 12 feature_drift.py                          "Compute linguistic drift (standard)"
run_step 13 feature_drift_extended.py                 "Compute linguistic drift (extended) [NEW]"
run_step 14 evaluate_hardness_buckets.py              "Evaluate by hardness bucket"
run_step 15 evaluate_auroc.py                         "Compute AUROC + ROC curves [NEW]"
run_step 16 evaluate_bootstrap_ci.py                  "Bootstrap CIs + McNemar tests [NEW]"
run_step 17 analyze_top_features.py                   "Analyze top discriminative features [NEW]"
run_step 18 error_analysis.py                         "Error analysis + confusion matrices [NEW]"

# ── Plotting ──────────────────────────────────────────────────────────────────
run_step 19 plot_robustness.py                        "Plot robustness curves"
run_step 20 plot_robustness_test_dualtrack.py         "Plot dual-track robustness"
run_step 21 plot_hardness_buckets.py                  "Plot hardness bucket F1"

echo ""
echo "============================================================"
echo "  Pipeline complete!"
echo "  Results in:  results/"
echo "  Figures in:  figures/"
echo "============================================================"
