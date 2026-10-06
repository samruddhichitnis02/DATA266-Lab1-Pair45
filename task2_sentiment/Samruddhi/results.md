# Task 2 Results — Yelp Polarity Sentiment Classification

**Member:** Samruddhi Chitnis  
**Dataset:** Yelp Polarity (`fancyzhx/yelp_polarity`)  
**Task:** Binary sentiment classification without pretrained embeddings or pretrained language models

## 1. Dataset and preprocessing

The Yelp Polarity dataset contains binary review labels: negative (`0`) and positive (`1`). The original training split contained 560,000 reviews. After preprocessing, 32 training rows became empty and were removed. The remaining data were split into 504,000 training examples and 56,000 validation examples. The test set contained 38,000 examples.

The preprocessing pipeline was:

1. Audit missing, empty, and malformed entries.
2. Lowercase the review text.
3. Expand contractions where possible.
4. Remove punctuation and special characters while preserving the words needed for sentiment interpretation.
5. Tokenize the cleaned text.
6. Remove common stopwords while retaining negation information.
7. Build the vocabulary from the training split only to avoid vocabulary leakage.
8. Map tokens to integer IDs and use `<pad>` and `<unk>` tokens.
9. Pad or truncate reviews to a maximum length of 200 tokens.

The final vocabulary contains 30,002 entries, including special tokens. Every model learns its own embedding matrix from scratch during training. No GloVe, Word2Vec, FastText, BERT, or other pretrained representation was used.

The class distribution is effectively balanced. The class and review-length analysis is saved in `class_and_length_distribution.png`.

## 2. Hardware and shared training setup

All three models were trained using CUDA on an NVIDIA GeForce RTX 3060 Laptop GPU with 6.4 GB available GPU memory. The system had an Intel64 processor with 20 threads and 16.8 GB RAM. The random seed was 42.

The shared training setup used binary cross-entropy with logits, AdamW optimization, weight decay, gradient clipping at 1.0, validation monitoring, learning-rate reduction on plateau, early stopping based on validation accuracy, and restoration of the best validation checkpoint. Probabilities were computed with the sigmoid function and converted to class predictions using a 0.5 threshold.

## 3. Model architectures and hyperparameters

### Baseline: MeanMaxMLP

The baseline learns a 128-dimensional embedding for each token. It summarizes a review using both mean pooling and max pooling over its token embeddings. The pooled representation is passed through a 256-unit multilayer perceptron with dropout.

This model provides a simple order-insensitive reference point. It can identify important words and general sentiment vocabulary, but it does not explicitly model word order or longer interactions between words.

### Experiment 1: Tiny Transformer

The first experimental model uses learned 128-dimensional embeddings, four attention heads, two Transformer encoder layers, a 256-unit feed-forward layer, dropout of 0.20, and a maximum sequence length of 200.

The Transformer was tested because self-attention can connect words that are far apart in a review and can model interactions such as negation or contrast. Its limitations are higher computational cost and sensitivity to sequence length.

### Experiment 2: BiLSTM with attention

The second experimental model uses 160-dimensional learned embeddings, a bidirectional LSTM with hidden size 128, one recurrent layer, and dropout of 0.40. Attention pooling is used to combine the token-level recurrent representations into a review representation.

The BiLSTM was tested because it can model sequential context in both directions. The attention pooling allows the model to emphasize informative parts of the review instead of treating all tokens equally. This combination was especially useful for long reviews, mixed sentiment, and reviews where the final opinion depends on earlier context.

| Model | Embedding | Main architecture | Dropout | Learning rate | Batch size | Epochs |
|---|---:|---|---:|---:|---:|---:|
| `baseline_meanmax_mlp` | 128 | Mean + max pooling, MLP hidden size 256 | 0.35 | 0.0015 | 256 | 8 |
| `exp1_transformer` | 128 | 2-layer Transformer, 4 heads, feed-forward size 256 | 0.20 | 0.0005 | 128 | 6 |
| `exp2_bilstm_attn` | 160 | 1-layer bidirectional LSTM, hidden size 128, attention pooling | 0.40 | 0.0010 | 128 | 5 |

## 4. Results for my three models

The complete metric report is stored in `metrics_report.csv` and `metrics_report.json`. The table below contains the main comparison metrics.

| Model | Accuracy | Precision macro | Recall macro | F1 macro | F1 weighted | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Time (s) | Examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline MeanMaxMLP | 0.9358 | 0.9360 | 0.9358 | 0.9358 | 0.9358 | 0.9831 | 0.9835 | 0.8718 | 0.0481 | 0.0094 | 3,906,305 | 294.63 | 13,684.81 |
| Tiny Transformer | 0.9379 | 0.9381 | 0.9379 | 0.9379 | 0.9379 | 0.9852 | 0.9858 | 0.8760 | 0.0463 | 0.0171 | 4,130,945 | 1,294.78 | 2,335.53 |
| BiLSTM + attention | **0.9568** | **0.9569** | **0.9568** | **0.9568** | **0.9568** | **0.9920** | **0.9922** | **0.9137** | **0.0332** | 0.0164 | 5,097,794 | 3,353.64 | 751.42 |

The 95% bootstrap confidence intervals were:

| Model | Accuracy | Macro-F1 | MCC |
|---|---|---|---|
| Baseline MeanMaxMLP | [0.9335, 0.9383] | [0.9335, 0.9383] | [0.8672, 0.8766] |
| Tiny Transformer | [0.9356, 0.9403] | [0.9356, 0.9403] | [0.8713, 0.8807] |
| BiLSTM + attention | [0.9547, 0.9587] | [0.9547, 0.9587] | [0.9096, 0.9176] |

The BiLSTM with attention is the best model across the main predictive metrics. It improves accuracy over the baseline by approximately 2.10 percentage points and over the Transformer by approximately 1.88 percentage points. It also has the lowest Brier score, indicating the strongest probability quality among the three models. The baseline is substantially faster, while the BiLSTM provides the best accuracy and robustness at a higher computational cost.

## 5. Confusion matrices and statistical comparison

The confusion matrices are saved in `confusion_matrices.png`. In the order true-negative, false-positive, false-negative, true-positive, the matrices were:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| Baseline MeanMaxMLP | 17,936 | 1,064 | 1,374 | 17,626 |
| Tiny Transformer | 17,987 | 1,013 | 1,345 | 17,655 |
| BiLSTM + attention | 18,006 | 994 | 648 | 18,352 |

McNemar tests were performed on paired predictions from the same test examples:

| Comparison | Baseline-only correct | Experimental-only correct | p-value |
|---|---:|---:|---:|
| Baseline vs. Tiny Transformer | 503 | 583 | 0.0165 |
| Baseline vs. BiLSTM + attention | 610 | 1,406 | 3.77 × 10⁻⁷⁰ |

Both experimental models improve over the baseline significantly at the 0.05 level. The improvement from the BiLSTM is especially strong: it corrects far more baseline errors than the baseline corrects in cases where the BiLSTM is wrong.

## 6. Robustness by data slice

The slice metrics below measure macro-F1 and error rate on groups of reviews with different characteristics.

| Slice | Baseline F1 / error | Transformer F1 / error | BiLSTM F1 / error |
|---|---:|---:|---:|
| Short reviews | 0.9340 / 0.0635 | 0.9345 / 0.0632 | **0.9514 / 0.0465** |
| Long reviews | 0.9295 / 0.0667 | 0.9334 / 0.0633 | **0.9513 / 0.0465** |
| Reviews with negation | 0.9279 / 0.0695 | 0.9309 / 0.0666 | **0.9551 / 0.0436** |
| Reviews with contrast words | 0.9251 / 0.0740 | 0.9279 / 0.0712 | **0.9522 / 0.0474** |
| Exclamation-heavy reviews | 0.9507 / 0.0481 | 0.9534 / 0.0454 | **0.9710 / 0.0281** |

Contrast-word reviews are the hardest slice for all three models. This is consistent with the fact that these reviews often contain both positive and negative statements, with the final judgment depending on context. The BiLSTM has the lowest error rate on every reported slice.

## 7. Manual error review

I manually reviewed 20 errors from the BiLSTM with attention:

- 5 confident false positives
- 5 confident false negatives
- 5 near-threshold errors
- 5 contrast-word slice failures

The complete review, including the review text, error type, and proposed testable fix, is stored in `error_review.md` and `error_audit_Samruddhi_exp2_bilstm_attn.csv`.

The main observed failure patterns were:

- mixed sentiment, where praise and complaints occur in the same review;
- sarcasm, irony, and emoticons removed during preprocessing;
- long reviews where the decisive opinion appears after the truncation boundary;
- comparative or lukewarm language such as “fine enough” or “still prefer”;
- numeric ratings and implicit negative sentiment;
- slang, misspellings, elongated words, and topic-sentiment confounds;
- possible label noise or stale ratings.

The proposed fixes are testable rather than purely speculative. They include head-and-tail truncation, preserving emoticons, adding rating-pattern tokens, subword tokenization, adding last-state pooling, using a larger maximum length, and conducting blind relabeling to estimate a possible label-noise ceiling.

## 8. Team comparison with Nikhil Kanaparthi

Nikhil used a different model lineup: a mean-pooled baseline, a CNN with kernel widths 3, 5, and 7, and a bidirectional GRU. His models were also trained from scratch on the same Yelp Polarity task without pretrained embeddings.

| Member | Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Samruddhi | MeanMaxMLP baseline | 0.9358 | 0.9358 | 0.9831 | 0.9835 | 0.8718 | 0.0481 | 0.0094 |
| Samruddhi | Tiny Transformer | 0.9379 | 0.9379 | 0.9852 | 0.9858 | 0.8760 | 0.0463 | 0.0171 |
| Samruddhi | BiLSTM + attention | **0.9568** | **0.9568** | **0.9920** | **0.9922** | **0.9137** | **0.0332** | 0.0164 |
| Nikhil | Mean-pooling baseline | 0.9362 | 0.9362 | 0.9816 | 0.9813 | 0.8723 | 0.0480 | **0.0062** |
| Nikhil | CNN | 0.9491 | 0.9491 | 0.9892 | 0.9895 | 0.8983 | 0.0382 | 0.0134 |
| Nikhil | BiGRU | 0.9521 | 0.9521 | 0.9897 | 0.9897 | 0.9042 | 0.0378 | 0.0221 |

The best overall result is the Samruddhi BiLSTM with attention. It exceeds Nikhil's best BiGRU by approximately 0.47 percentage points in accuracy and macro-F1, and by approximately 0.96 percentage points in MCC. The likely reason is that the attention-pooled BiLSTM can combine bidirectional sequential context with a learned focus on informative tokens. Nikhil's CNN is a strong alternative because local n-gram features improve over the baseline at substantially lower cost than a recurrent model.

The comparison also shows that performance is not determined by parameter count alone. The BiLSTM has more parameters and is slower than the simpler models, but it provides the strongest predictive performance and the best slice robustness. The baselines are useful because they are faster and provide a reproducible reference point. The Transformer improves only modestly over the baseline in this experiment, suggesting that its additional computational cost was not fully converted into accuracy under the selected sequence length and training budget.

## 9. Limitations and future work

The reviews are padded or truncated at 200 tokens, so information near the end of long reviews may be lost. Removing punctuation, emoticons, and some stopwords can also remove useful sentiment cues. Keyword-defined slices overlap and do not fully represent linguistic phenomena such as sarcasm, discourse structure, or domain-specific language. The reported confidence intervals quantify sampling uncertainty but do not capture dataset shift or annotation noise.

Future experiments should test head-and-tail truncation, subword tokenization, preservation of emoticons and rating expressions, larger sequence lengths, probability calibration, and an ensemble of the BiLSTM and Transformer. A human relabeling study of highly confident errors would also help estimate the effect of noisy or stale Yelp ratings.


