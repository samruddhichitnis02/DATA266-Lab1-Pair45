# DATA266 Lab 1 — Task 2: Yelp Polarity Sentiment Classification

**Member:** Samruddhi Chitnis  
**Dataset:** Yelp Polarity (`fancyzhx/yelp_polarity`)  
**Task:** Binary sentiment classification using neural models trained from scratch

## Project overview

This project evaluates three neural sentiment-classification models on the Yelp Polarity dataset. The assignment prohibits pretrained embeddings and pretrained language models, so every model learns its token embeddings from scratch.

The three models are:

1. **MeanMaxMLP baseline** — mean and max pooling over learned embeddings followed by a multilayer perceptron.
2. **TinyTransformer** — learned embeddings, positional embeddings, and a small Transformer encoder.
3. **BiLSTM with attention** — learned embeddings, a bidirectional LSTM, and attention pooling.

The BiLSTM with attention achieved the best performance, with 95.68% test accuracy, 0.9568 macro-F1, 0.9137 MCC, 0.9920 ROC-AUC, and 0.9922 PR-AUC.

## Repository contents

The final member folder should have the following structure:

```text
Samruddhi/
├── README.md
├── results.md
├── yelp_polarity_task2.ipynb
├── config.json
├── requirements.txt
├── run_manifest.json
├── sysinfo.json
├── vocabulary.json
├── training_log.txt
├── error_review.md
├── error_audit_Samruddhi_exp2_bilstm_attn.csv
├── failure_analysis.md
├── metrics_report.csv
├── metrics_report.json
├── test_probs.npz
├── class_and_length_distribution.png
├── training_curves.png
├── confusion_matrices.png
├── baseline_meanmax_mlp_history.csv
├── exp1_transformer_history.csv
├── exp2_bilstm_attn_history.csv
└── checkpoints/
    ├── baseline_meanmax_mlp_weights.pt
    ├── exp1_transformer_weights.pt
    └── exp2_bilstm_attn_weights.pt
```

Large intermediate files such as `encoded_data.pt`, `test_df.pkl`, and raw Yelp data are not required in the repository. The raw dataset is downloaded by the notebook and is intentionally not committed.

## Dataset and preprocessing

The notebook loads the Yelp Polarity dataset from Hugging Face. The original training split contains 560,000 examples and the official test split contains 38,000 examples. After cleaning, the training data are divided into training and validation sets.

The preprocessing pipeline performs the following operations:

- checks null, empty, duplicate, and malformed entries;
- lowercases the text;
- expands common contractions so that negation is retained;
- removes punctuation and special characters;
- tokenizes the cleaned text;
- removes common stopwords while retaining important negation and contrast words;
- builds the vocabulary from the training split only;
- maps tokens to integer IDs;
- pads or truncates reviews to a maximum sequence length of 200 tokens.

The vocabulary contains 30,002 entries, including special tokens. No pretrained embeddings, pretrained language models, GloVe vectors, Word2Vec vectors, FastText vectors, BERT weights, or other external representations are used.

## Model architectures

### MeanMaxMLP baseline

The baseline learns 128-dimensional token embeddings. It computes both mean pooling and max pooling over the sequence, concatenates the two representations, and sends the result through a 256-unit MLP with dropout.

This model is an order-insensitive reference point. It measures how well sentiment can be predicted from the presence and strength of individual token features without explicitly modeling word order.

### TinyTransformer

The Transformer uses 128-dimensional learned embeddings, learned positional embeddings, two encoder layers, four attention heads, a 256-unit feed-forward layer, and dropout of 0.20. Masked mean pooling produces the review representation used by the classifier.

This model tests whether self-attention can capture long-range relationships, negation, and contrast without using pretraining.

### BiLSTM with attention

The BiLSTM uses 160-dimensional learned embeddings, one bidirectional LSTM layer with hidden size 128, dropout of 0.40, and additive attention pooling.

This model reads each review in both directions and allows the attention mechanism to focus on the most informative parts of the review. It achieved the best performance in this experiment.

## Training configuration

All models use:

- random seed: `42`;
- binary cross-entropy with logits;
- AdamW optimization with weight decay `0.01`;
- gradient clipping at `1.0`;
- learning-rate reduction on validation plateau;
- early stopping based on validation accuracy;
- restoration of the best validation checkpoint;
- sigmoid probabilities with a decision threshold of `0.5`.

| Model | Learning rate | Batch size | Epoch limit |
|---|---:|---:|---:|
| MeanMaxMLP | 0.0015 | 256 | 8 |
| TinyTransformer | 0.0005 | 128 | 6 |
| BiLSTM with attention | 0.0010 | 128 | 5 |

## Hardware

The models were trained using:

- GPU: NVIDIA GeForce RTX 3060 Laptop GPU;
- device: CUDA;
- CPU: Intel64 processor with 20 logical threads;
- RAM: approximately 16.8 GB;
- operating system: Windows.

The exact recorded hardware and package information is stored in `sysinfo.json` and `run_manifest.json`.

## Evaluation

The evaluation reports the following metrics for every model:

- accuracy;
- precision, recall, and F1-score using macro, micro, and weighted averaging;
- confusion matrix;
- ROC-AUC and PR-AUC;
- Matthews correlation coefficient;
- Brier score;
- expected calibration error;
- 95% bootstrap confidence intervals for accuracy, macro-F1, and MCC;
- paired McNemar tests;
- macro-F1 and error rate across review slices;
- parameter count;
- training time;
- examples processed per second;
- peak GPU and CPU memory.

The main evaluation files are `metrics_report.csv` and `metrics_report.json`.

## Main results

| Model | Accuracy | Macro-F1 | MCC | ROC-AUC | PR-AUC | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| MeanMaxMLP | 0.9358 | 0.9358 | 0.8718 | 0.9831 | 0.9835 | 0.0481 | 0.0094 |
| TinyTransformer | 0.9379 | 0.9379 | 0.8760 | 0.9852 | 0.9858 | 0.0463 | 0.0171 |
| BiLSTM with attention | **0.9568** | **0.9568** | **0.9137** | **0.9920** | **0.9922** | **0.0332** | 0.0164 |

The BiLSTM with attention is the strongest model. It achieves the highest predictive performance and the lowest error rate on every evaluated data slice. The baseline is much faster and uses less memory, while the Transformer provides only a modest improvement over the baseline under the selected training budget.

## Error analysis

The manual error review contains exactly 20 errors from the BiLSTM with attention:

- 5 confident false positives;
- 5 confident false negatives;
- 5 near-threshold errors;
- 5 contrast-word slice failures.

The main observed error types include mixed sentiment, sarcasm, irony, truncation, comparative language, implicit negative sentiment, numeric ratings, slang, misspellings, elongated words, and possible label noise.

The complete examples and proposed testable fixes are in:

- `error_review.md`;
- `error_audit_Samruddhi_exp2_bilstm_attn.csv`;
- `failure_analysis.md`.

## Reproducing the experiment

Create and activate a clean Python environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter lab
```

Open and run:

```text
yelp_polarity_task2.ipynb
```

Run the notebook from top to bottom. The first execution downloads the Yelp Polarity dataset and the required NLTK resources. GPU execution is recommended because training all three models is computationally expensive.

For a command-line notebook execution after the environment is configured:

```bash
jupyter nbconvert --to notebook --execute yelp_polarity_task2.ipynb --output executed_task2.ipynb
```

## Reproducibility notes

- The test set is evaluated in its official order.
- The vocabulary is built using the training split only.
- The random seed is recorded in `config.json` and `run_manifest.json`.
- Raw training output is preserved in `training_log.txt`.
- Model checkpoints are stored in the `checkpoints/` directory.
- The full metric outputs and prediction probabilities are retained for traceability.
- Personal file paths and credentials must not be committed.

## Important notebook cleanup before submission

Before committing the notebook, verify that:

1. the detailed comparison section is present;
2. no `<FILL IN>` placeholders remain;
3. no personal Windows paths appear in cell outputs;
4. the export cell does not overwrite the detailed `README.md` or `results.md`;
5. only final, canonical filenames are included;
6. no raw dataset, credentials, or API keys are included.

