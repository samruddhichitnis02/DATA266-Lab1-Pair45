# DATA266 Lab 1 — Pair 45

This repository contains the completed experiments for DATA266 Lab 1. Each task is organized in its own folder with source code, configurations, metrics, logs, evaluation results, and selected outputs.

## Repository Structure

```text
DATA266-Lab1-Pair45/
├── task1_llm/
│   └── Nikhil_Kanaparthi/
├── task2_sentiment/
│   └── Nikhil_Kanaparthi/
└── task3_GAN/
    └── Nikhil_Kanaparthi/
```

## Task Summary

| Task | Description | Main Folder |
|---|---|---|
| Task 1 | Character-level language modeling | `task1_llm/Nikhil_Kanaparthi/` |
| Task 2 | Yelp Polarity sentiment classification | `task2_sentiment/Nikhil_Kanaparthi/` |
| Task 3 | CycleGAN image style transfer | `task3_GAN/Nikhil_Kanaparthi/` |

---

## Task 1 — Character-Level Language Modeling

Task 1 trains a character-level language model and evaluates its ability to predict and generate text.

The experiment reports:

- Training and validation cross-entropy loss
- Perplexity
- Bits per character
- Next-character accuracy
- Generalization gap
- Greedy decoding results
- Sampling-based generation results
- Distinct-n diversity metrics
- Repeated n-gram rate
- Gradient norms
- Numerical stability checks
- Training and generation speed

### Task 1 Smoke Test

```bash
cd task1_llm/Nikhil_Kanaparthi
python src/train_task1.py --smoke-test
```

Important Task 1 files include:

```text
config.json
history.json
metrics/
outputs/
raw_logs/
results.md
run_manifest.json
src/
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

```bash
cd task2_sentiment/Nikhil_Kanaparthi
python src/train_task2.py --smoke-test
```

Full-run validation accuracy:

| Model | Validation Accuracy |
|---|---:|
| Mean-embedding baseline | 0.9337 |
| CNN classifier | 0.9470 |
| Bidirectional GRU | 0.9496 |

Important Task 2 files include:

```text
metrics_report.csv
metrics_report.json
error_review.csv
results.md
history.json
metrics/
outputs/
raw_logs/
```

---

## Task 3 — CycleGAN Image Style Transfer

Task 3 trains a CycleGAN model for unpaired image-to-image translation.

The experiment includes:

- Training through 100 epochs
- Periodic model checkpoints
- Image translation previews
- Training loss history
- FID evaluation
- MiFID evaluation
- KID evaluation
- Precision and recall
- LPIPS perceptual similarity
- Cycle-consistency error
- Human audit of 30 translated samples
- Failure analysis
- Kaggle submission preparation

### Task 3 Smoke Test

```bash
cd task3_GAN/Nikhil_Kanaparthi
python src/train_cyclegan.py --smoke-test
```

### Task 3 Evaluation

```bash
cd task3_GAN/Nikhil_Kanaparthi
python src/evaluate_task3.py
```

Important Task 3 files include:

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

### Task 3 Final Metrics

| Metric | A → B | B → A |
|---|---:|---:|
| FID | 104.0653 | 98.3351 |
| MiFID | 0.4194 | 0.4066 |
| KID | 0.022950 | 0.006728 |
| Precision | 0.3567 | 0.1833 |
| Recall | 0.1867 | 0.3167 |
| Content cosine similarity | 0.9749 | 0.9664 |

Additional results:

| Metric | Result |
|---|---:|
| Cycle-consistency L1 mean | 0.0833 |
| Mean LPIPS | 0.2567 |
| Human audit rows | 30 |
| Mean content score, Rater A | 4.60 |
| Mean content score, Rater B | 4.60 |
| Mean artifact score, Rater A | 3.77 |
| Mean artifact score, Rater B | 3.70 |
| Content agreement | 100% |
| Artifact agreement | 93.33% |

The file `submission.csv` is prepared for submission to the class Kaggle competition.

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

---

## Large Files

Large model checkpoints and generated image folders are excluded from the regular GitHub commit because GitHub has a 100 MB file-size limit.

The final Task 3 checkpoint and complete generated results are preserved in the experiment backup archive. The GitHub repository contains the corresponding manifests, logs, metrics, human-audit results, and representative output images.

---

## Authors

DATA266 Lab 1 — Pair 45
