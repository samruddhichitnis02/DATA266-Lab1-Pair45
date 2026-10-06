# DATA266 Lab 1 — Pair 45

This repository contains the joint work completed by Pair 45 for DATA266 Lab 1.

## Team Members

- Nikhil Kanaparthi
- Samruddhi Chitnis

Both members contributed to implementation, experimentation, evaluation, documentation, and report preparation across the three tasks.

## Repository Structure

```text
DATA266-Lab1-Pair45/
├── task1_llm/
│   ├── member_folder_1/
│   └── member_folder_2/
├── task2_sentiment/
│   ├── member_folder_1/
│   └── member_folder_2/
└── task3_GAN/
    ├── member_folder_1/
    └── member_folder_2/
```

Each task folder contains the work and results contributed by the pair members.

## Task Summary

| Task | Description | Folder |
|---|---|---|
| Task 1 | Character-level language modeling | `task1_llm/` |
| Task 2 | Yelp Polarity sentiment classification | `task2_sentiment/` |
| Task 3 | CycleGAN image style transfer | `task3_GAN/` |

## Pair Contributions

| Work Area | Pair Contribution |
|---|---|
| Task 1 | Language-model implementation, training, text generation, and evaluation |
| Task 2 | Sentiment-model comparison, metrics, error analysis, and manual review |
| Task 3 | CycleGAN training, image translation, quantitative evaluation, human audit, and failure analysis |
| Documentation | README, manifests, logs, results, and final report |
| Reproducibility | Configuration files, smoke tests, checkpoints, and experiment records |

---

## Task 1 — Character-Level Language Modeling

Task 1 trains and evaluates a character-level language model.

The evaluation includes:

- Training and validation cross-entropy loss
- Perplexity
- Bits per character
- Next-character accuracy
- Generalization gap
- Greedy and sampled generation
- Distinct-n diversity metrics
- Repeated n-gram rate
- Gradient norms
- Numerical stability checks
- Training and generation speed

### Task 1 Smoke Test

Run this command from the repository root:

```bash
for folder in task1_llm/*; do
  if [ -f "$folder/src/train_task1.py" ]; then
    echo "Running Task 1 smoke test in $folder"
    (cd "$folder" && python src/train_task1.py --smoke-test)
  fi
done
```

---

## Task 2 — Yelp Polarity Sentiment Classification

Task 2 uses the Yelp Polarity dataset for binary sentiment classification.

The following models are compared:

1. Mean-embedding baseline
2. Convolutional neural network
3. Bidirectional GRU classifier

The evaluation includes:

- Accuracy
- Macro F1 score
- ROC-AUC
- PR-AUC
- Matthews correlation coefficient
- Brier score
- Expected calibration error
- Confusion matrices
- Training curves
- Error analysis
- Manual error review

### Task 2 Smoke Test

Run this command from the repository root:

```bash
for folder in task2_sentiment/*; do
  if [ -f "$folder/src/train_task2.py" ]; then
    echo "Running Task 2 smoke test in $folder"
    (cd "$folder" && python src/train_task2.py --smoke-test)
  fi
done
```

Important artifacts include:

```text
metrics_report.csv
metrics_report.json
error_review.csv
history.json
metrics/
outputs/
raw_logs/
results.md
```

---

## Task 3 — CycleGAN Image Style Transfer

Task 3 trains a CycleGAN model for unpaired image-to-image translation.

The experiment includes:

- Training through 100 epochs
- Periodic checkpoints
- Image translation previews
- Training-loss history
- FID evaluation
- MiFID evaluation
- KID evaluation
- Precision and recall
- LPIPS perceptual similarity
- Cycle-consistency error
- Human audit of translated samples
- Failure analysis
- Kaggle submission preparation

### Task 3 Smoke Test

Run this command from the repository root:

```bash
for folder in task3_GAN/*; do
  if [ -f "$folder/src/train_cyclegan.py" ]; then
    echo "Running Task 3 smoke test in $folder"
    (cd "$folder" && python src/train_cyclegan.py --smoke-test)
  fi
done
```

### Task 3 Evaluation

```bash
for folder in task3_GAN/*; do
  if [ -f "$folder/src/evaluate_task3.py" ]; then
    echo "Evaluating Task 3 in $folder"
    (cd "$folder" && python src/evaluate_task3.py)
  fi
done
```

Important artifacts include:

```text
full_metrics_report.csv
metrics/full_metrics_report.json
human_audit.csv
submission.csv
results.md
failure_analysis.md
history.json
config.json
run_manifest.json
backup_manifest.json
outputs/
raw_logs/
src/
```

---

## Reproducibility

Each task contains its own configuration, training history, logs, evaluation metrics, and output files.

The run manifests document:

- Random seed
- Device used
- Dataset information
- Training configuration
- Model settings
- Output locations
- Evaluation artifacts

The smoke-test commands above allow the task pipelines to be checked from the repository root without manually entering one member’s folder.

---

## Large Files

Large model checkpoints and generated image folders are excluded from the regular GitHub commit because GitHub has a 100 MB file-size limit.

The complete checkpoints and generated outputs are preserved in the experiment backup archives. The repository contains the related manifests, logs, metrics, human-audit results, and representative output images.

---

## Authors

**DATA266 Lab 1 — Pair 45**

- Nikhil Kanaparthi
- Samruddhi Chitnis
