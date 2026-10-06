# DATA266 Lab 1 — Pair 45

This repository contains the joint work completed by Pair 45 for DATA266 Lab 1.

**Team members:**

- Nikhil Kanaparthi
- Samruddhi Chitnis

The repository includes the source code, configurations, training logs, metrics, evaluation results, error analyses, and selected outputs for all three tasks.

## Repository Structure

```text
DATA266-Lab1-Pair45/
├── task1_llm/
├── task2_sentiment/
└── task3_GAN/
```

Each task folder contains the relevant implementation and experiment artifacts.

## Task Summary

| Task | Description | Folder |
|---|---|---|
| Task 1 | Character-level language modeling | `task1_llm/` |
| Task 2 | Yelp Polarity sentiment classification | `task2_sentiment/` |
| Task 3 | CycleGAN image style transfer | `task3_GAN/` |

---

## Task 1 — Character-Level Language Modeling

Task 1 trains and evaluates a character-level language model.

The evaluation includes:

- Training and validation cross-entropy loss
- Perplexity
- Bits per character
- Next-character accuracy
- Generalization gap
- Greedy text generation
- Sampling-based text generation
- Distinct-n diversity metrics
- Repeated n-gram rate
- Gradient norm monitoring
- Numerical stability checks
- Training and generation speed

### Smoke Test

```bash
cd task1_llm/Nikhil_Kanaparthi
python src/train_task1.py --smoke-test
```

Important artifacts include:

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

### Smoke Test

```bash
cd task2_sentiment/Nikhil_Kanaparthi
python src/train_task2.py --smoke-test
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

### Smoke Test

```bash
cd task3_GAN/Nikhil_Kanaparthi
python src/train_cyclegan.py --smoke-test
```

### Evaluation

```bash
cd task3_GAN/Nikhil_Kanaparthi
python src/evaluate_task3.py
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

The included smoke-test commands provide a quick way to verify that the main training pipelines are configured correctly.

---

## Large Files

Large model checkpoints and generated image folders are excluded from the regular GitHub commit because GitHub has a 100 MB file-size limit.

The complete checkpoints and generated outputs are preserved in the experiment backup archives. This repository contains the corresponding manifests, logs, metrics, human-audit results, and representative output images.

---

## Collaboration

The repository is organized by task so that both team members can contribute independently while maintaining a shared project structure.

Major updates are recorded through separate Git commits for:

- Source code and configuration
- Training results and metrics
- Evaluation and error analysis
- Documentation and report updates

---

## Authors

**DATA266 Lab 1 — Pair 45**

- Nikhil Kanaparthi
- Samruddhi Chitnis
