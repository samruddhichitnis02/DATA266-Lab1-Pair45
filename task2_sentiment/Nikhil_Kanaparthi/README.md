# DATA266 Lab 1 — Task 2: Yelp Polarity Sentiment Classification

This folder contains my three-model sentiment-classification experiment. All
token embeddings are initialized and trained in this project; no pretrained
embedding or pretrained language model is used.

## Models

1. **Baseline:** mean-pooled learned word embeddings followed by a linear
   classifier.
2. **Experimental CNN:** learned embeddings followed by convolution kernels of
   widths 3, 5, and 7, global max pooling, and a classifier.
3. **Experimental BiGRU:** learned embeddings followed by a bidirectional GRU
   and a classifier.

The three models use the same cleaned data, vocabulary, train/validation split,
test set, decision threshold, and evaluation code. Their architectures and
hyperparameters are recorded in `config.json`. The final lineup must be
checked against teammate models before the team report is submitted.

## Reproduce the experiment

From this member folder:

```bash
python -m pip install -r requirements.txt
python src/train_task2.py --smoke-test
mkdir -p raw_logs
python src/train_task2.py --epochs 5 2>&1 | tee raw_logs/task2_full_run.log
```

The first run downloads the Yelp Polarity CSV release corresponding to the
two-class Yelp Polarity dataset listed at the Hugging Face dataset page. The
script does not commit the raw dataset or processed arrays.

## Generated evidence

- `metrics_report.csv` and `metrics_report.json`: all required metrics for all
  three models.
- `outputs/`: preprocessing plots, training curves, confusion matrices, and
  per-model histories.
- `checkpoints/`: one final checkpoint per model.
- `error_review.csv`: exactly 20 candidate errors for manual review. I must
  fill in an observation and one testable fix for each row after reading the
  review text.
- `raw_logs/`: unedited console output from the training run.
- `run_manifest.json`: package versions, hardware, seeds, and checkpoint map.
- `results.md`: generated comparison and interpretation after a run.
