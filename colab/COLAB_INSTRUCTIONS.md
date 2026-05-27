# Running the RoBERTa detector in Colab

The local Windows pipeline runs CPU-only torch (because pip on Windows defaults to CPU wheels). RoBERTa fine-tuning needs a GPU; the easiest route is Colab's free T4. Total time end-to-end: **~15 minutes** (5 min setup, 7 min training on 3 seeds, 3 min eval + download).

---

## Files

- **Notebook:** `colab/roberta_detector_finetune.ipynb` — self-contained; opens directly in Colab.
- **Data bundle:** `colab/bundle/` — 8 files (5 JSONL + 3 split ID files) ready to upload in one go.

---

## Step-by-step

### 1. Open the notebook in Colab

Visit https://colab.research.google.com → File → Upload notebook → choose `colab/roberta_detector_finetune.ipynb`.

### 2. Switch to GPU

Runtime → Change runtime type → Hardware accelerator: **T4 GPU** (the free tier). Save.

Verify in cell 1 output that it prints `CUDA: True` and shows a GPU name. If not, the wrong runtime is selected.

### 3. Run cell 1 (install)

About 60 seconds. Installs `transformers`, `datasets`, `accelerate`, `evaluate`, `scikit-learn`. Re-running idempotent.

### 4. Run cell 2 (file upload)

A file picker pops up.

**Upload all 8 files from `colab/bundle/` at once** — hold Ctrl (Windows) or Cmd (Mac) and click each file, then "Open." The files land in the Colab working directory. The cell prints the file names back to confirm.

If you forget a file the next cell will fail with `FileNotFoundError`; re-run cell 2 to add the missing file.

### 5. Run cells 3 → 5 (data load, tokenize, training)

- Cell 3 loads JSONL into HuggingFace `Dataset` objects.
- Cell 4 tokenizes with the RoBERTa tokenizer.
- Cell 5 fine-tunes **RoBERTa-base across 3 seeds** (~3 minutes per seed on T4). Watch the per-epoch validation accuracy — expect it to plateau at ~0.96-0.99 on the val set.

If Colab complains about RAM or disconnect: Runtime → Restart, then re-run from cell 1. The notebook is stateless except for the uploaded files.

### 6. Run cell 6 (summary)

Prints across-seed mean ± std for accuracy, F1, AUROC on each paraphrase split. Sanity check: P0_test accuracy should be ≥ 0.95; simplified P2 should be lower (the whole point of the paper).

### 7. Run cell 7 (download)

Triggers a browser download of `roberta_predictions.csv` (~30KB; one row per prediction across all seeds and splits).

### 8. Move the CSV into your local repo

Drop the downloaded file into:

```
Analysis-of-Human-vs.-LLM-Generated-Text\results\eval\roberta_predictions.csv
```

### 9. Run the integration script locally

```powershell
cd "C:\Users\razon\OneDrive\Desktop\Arjun Mukherjee\Human vs LLM\Analysis-of-Human-vs.-LLM-Generated-Text"
$env:PYTHONIOENCODING="utf-8"
.venv312\Scripts\python.exe src\experiments\integrate_neural_predictions.py
```

This will:
- Compute across-seed means + 95% CIs for every metric on every split (writes `results/eval/roberta_summary_flat.csv`).
- Compute AURC across seeds (writes `results/eval/roberta_summary_aurc.csv`).
- Compute per-bucket metrics under all 6 hardness definitions (writes `results/eval/roberta_summary_buckets.csv`).
- Combine with the word + char TF-IDF results into `results/eval/combined_detector_summary.csv`.
- Generate `figures/fig12_three_detector_robustness.png` (3-detector overlay across stages).
- Generate `figures/fig13_universal_hardness_collapse.png` (Hard-bucket collapse on all 3 detectors under non-circular hardness — **the universality plot**).

The headline question this answers: **does the hardness collapse persist when the detector is a neural model rather than a shallow bag-of-features?** If yes, the paper's universality claim is established.

---

## Optional: also download the fine-tuned checkpoint

Uncomment the final cell to download a 500MB zip of the seed-42 RoBERTa weights. Useful only if you want to evaluate it on additional paraphrase tracks later without retraining.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Cell 1 prints `CUDA: False` | Runtime → Change runtime type → GPU. |
| Cell 2 missing files | Re-run cell 2 and add the missed file. |
| Out of memory in cell 5 | Reduce `BATCH = 16` to `8`. Should not happen on T4 with these inputs. |
| Per-epoch acc < 0.85 on val | Confirm `train_ids.txt` and `p0.jsonl` were uploaded — if val_ids weren't loaded properly the val set may be wrong. |
| Cell 7 doesn't trigger download | The file `roberta_predictions.csv` is in the Colab file browser (left sidebar). Right-click → Download. |
| Integration script: `ERROR: results/eval/roberta_predictions.csv not found` | Move the downloaded CSV into `results/eval/`. |

---

## What this gives you scientifically

The single most important question — *does the hardness collapse persist under a neural detector?* — is answered by fig13. With three detectors agreeing on the collapse under non-circular hardness, the paper transitions from:

> "TF-IDF detectors are fragile under paraphrasing"  (descriptive, low novelty)

to:

> "Detection robustness fails *catastrophically and selectively* on a hardness-stratified subset, across detector families, even when hardness is defined independently of the evaluated detector." (mechanistic, methodological, publishable)
