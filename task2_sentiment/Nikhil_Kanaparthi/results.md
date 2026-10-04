# Task 2 Results — Nikhil Kanaparthi

## Dataset and preprocessing

Yelp Polarity was loaded from the public Yelp Polarity CSV release corresponding to the Hugging Face `fancyzhx/yelp_polarity` dataset. Text was lowercased, punctuation and special characters were removed, text was tokenized, common stopwords were removed, and negation words were retained. No pretrained embedding or language model was used. No stemming or lemmatization was applied because preserving word forms supports the error analysis.

The run used 503975 training examples, 55997 validation examples, and 38000 test examples. The vocabulary contained 30000 tokens and the sequence limit was 200 tokens. Device: `mps`.

Training class counts: {'0': 251980, '1': 251995}. Review-length statistics after preprocessing: {'min': 1, 'median': 60.0, 'mean': 81.70540602212411, 'p90': 172.0, 'p99': 371.0, 'max': 987}.

## Model comparison

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_mean | 0.9362 | 0.9362 | 0.9816 | 0.9813 | 0.8723 | 0.0480 | 0.0062 | 3,840,258 | 203.4 |
| experimental_cnn | 0.9491 | 0.9491 | 0.9892 | 0.9895 | 0.8983 | 0.0382 | 0.0134 | 4,086,914 | 640.9 |
| experimental_bigru | 0.9521 | 0.9521 | 0.9897 | 0.9897 | 0.9042 | 0.0378 | 0.0221 | 4,038,658 | 1965.6 |

The baseline uses a mean of learned token embeddings. The CNN tests local n-gram features with kernel widths 3, 5, and 7. The bidirectional GRU tests order-sensitive sequential features. The same preprocessing, split, test set, threshold, and bootstrap procedure are used for all three models.

## Statistical comparisons

McNemar tests compare each experimental model's paired predictions with the baseline. The exact p-values and discordant-pair counts are in `metrics_report.json` and `metrics_report.csv`.

## Outputs

- `outputs/eda_length_distribution.png` and `outputs/class_distribution.png`: preprocessing analysis.
- `outputs/training_curves.png`: training/validation loss and accuracy.
- `outputs/confusion_matrices.png`: confusion matrices for all models.
- `metrics_report.csv`: flattened rubric metrics for every model.
- `error_review.csv`: 20 candidates for manual review; observations and testable fixes must be completed after reading each review.
- `raw_logs/`: unedited console logs from each run.

## Limitations and future work

The models use a fixed vocabulary and truncate long reviews at the sequence limit. The preprocessing removes punctuation and many function words, which may discard some stylistic sentiment cues. Future work could compare learned subword features, calibrated thresholds, attention-free temporal pooling, and additional linguistic slices.
