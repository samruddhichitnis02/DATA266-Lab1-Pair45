# DATA266 Lab 1 — Task 2: Team Sentiment Classification

This folder contains the team's independent experiments for binary sentiment classification on the Yelp Polarity dataset. Each team member trained three models using learned embeddings and no pretrained embeddings or pretrained language models.

## Team ownership

### Samruddhi Chitnis

Samruddhi trained and evaluated:

1. `MeanMaxMLP` baseline using mean and max pooling;
2. `TinyTransformer` using self-attention and positional embeddings;
3. `BiLSTM with attention` using bidirectional sequential modeling and attention pooling.

Her models were trained with CUDA on an NVIDIA GeForce RTX 3060 Laptop GPU. Her complete code, configurations, metrics, figures, error review, logs, and checkpoints are in [`Samruddhi/`](Samruddhi/).

### Nikhil Kanaparthi

Nikhil trained and evaluated:

1. `Mean-pooling` baseline using learned word embeddings;
2. `CNN` using convolution kernels of widths 3, 5, and 7;
3. `BiGRU` using bidirectional recurrent modeling.

His models were trained using an Apple GPU through MPS. His complete code, configurations, metrics, figures, logs, and checkpoints are in [`Nikhil_Kanaparthi/`](Nikhil_Kanaparthi/).

## Shared task setup

Both members used the Yelp Polarity binary classification task. The reviews were cleaned, lowercased, tokenized, and converted into vocabulary indices. Each model learned its embeddings from scratch. The official 38,000-review test set was used for final evaluation.

The evaluation included accuracy, macro-F1, ROC-AUC, PR-AUC, MCC, Brier score, expected calibration error, confusion matrices, bootstrap confidence intervals, slice-level performance, parameter count, training time, throughput, and memory usage.

## Model comparison

| Member | Model | Architecture summary | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Samruddhi | MeanMaxMLP | Mean + max pooling followed by MLP | 0.9358 | 0.9358 | 0.9831 | 0.9835 | 0.8718 | 0.0481 | 0.0094 |
| Samruddhi | TinyTransformer | 2-layer, 4-head Transformer encoder | 0.9379 | 0.9379 | 0.9852 | 0.9858 | 0.8760 | 0.0463 | 0.0171 |
| Samruddhi | BiLSTM + attention | Bidirectional LSTM with attention pooling | **0.9568** | **0.9568** | **0.9920** | **0.9922** | **0.9137** | **0.0332** | 0.0164 |
| Nikhil | Mean-pooling baseline | Mean pooling followed by linear classifier | 0.9362 | 0.9362 | 0.9816 | 0.9813 | 0.8723 | 0.0480 | **0.0062** |
| Nikhil | CNN | Convolution kernels with widths 3, 5, and 7 | 0.9491 | 0.9491 | 0.9892 | 0.9895 | 0.8983 | 0.0382 | 0.0134 |
| Nikhil | BiGRU | Bidirectional GRU with learned embeddings | 0.9521 | 0.9521 | 0.9897 | 0.9897 | 0.9042 | 0.0378 | 0.0221 |

## Observations

### 1. Recurrent models performed best

The two strongest models were the recurrent models: Samruddhi's BiLSTM with attention and Nikhil's BiGRU. This suggests that modeling word order and sequence context was useful for Yelp reviews, where the final sentiment often depends on negation, contrast, and the relationship between multiple clauses.

### 2. Samruddhi's BiLSTM with attention achieved the best overall performance

The BiLSTM with attention was the best model in the team comparison. It achieved 95.68% accuracy and 0.9568 macro-F1, exceeding Nikhil's BiGRU by approximately 0.47 percentage points in both accuracy and macro-F1. It also achieved the highest ROC-AUC, PR-AUC, and MCC, and the lowest Brier score.

The attention mechanism likely helped the model focus on sentiment-bearing words while the bidirectional LSTM captured context from both directions. This was particularly helpful for long reviews, negation, and contrast-word reviews.

### 3. CNN was a strong alternative

Nikhil's CNN achieved 94.91% accuracy, substantially improving over the mean-pooling baseline. The convolution kernels captured local phrases and n-gram patterns efficiently. The CNN provided a strong accuracy-to-cost alternative to the recurrent models.

### 4. The two baselines performed similarly

The two baseline models achieved nearly identical accuracy: 93.58% for Samruddhi's MeanMaxMLP and 93.62% for Nikhil's mean-pooling baseline. This provides a useful sanity check that the two experiments began from comparable order-insensitive baselines.

### 5. Accuracy and calibration were different objectives

The highest-accuracy model did not have the lowest ECE. Nikhil's mean-pooling baseline had the lowest ECE at 0.0062, while Samruddhi's BiLSTM achieved the best Brier score at 0.0332. This shows that predictive accuracy and probability calibration should be analyzed separately.

### 6. Hardware differences limit resource comparisons

Training time and memory should not be compared as exact efficiency rankings because Samruddhi used an NVIDIA CUDA GPU and Nikhil used an Apple MPS device. The model-quality metrics are more directly comparable because both members evaluated the same type of task and test-set size.

## Error and robustness observations

Samruddhi's 20-example manual audit identified mixed sentiment, sarcasm, irony, truncation, comparative language, numeric ratings, implicit negative sentiment, slang, misspellings, elongated words, and possible label noise as common failure patterns.

The hardest Samruddhi slice was the contrast-word slice containing words such as `but`, `however`, and `although`. These reviews often contain both positive and negative statements, so the correct label may depend on the final clause or the overall balance of the review.

Nikhil's slice metrics also showed reduced performance on long reviews and reviews containing negation. This is consistent with the difficulty of preserving sentiment context when reviews are long or contain conflicting words.

## Final conclusion

The team comparison shows that architectural choices mattered substantially. Simple pooling baselines were fast and competitive, but models that represented sequence context performed better. The CNN improved over the baseline by learning local phrase patterns, while the BiGRU and BiLSTM captured bidirectional context.

The best overall model was Samruddhi's BiLSTM with attention, which achieved 95.68% accuracy, 0.9568 macro-F1, 0.9137 MCC, 0.9920 ROC-AUC, and 0.9922 PR-AUC. It also achieved the lowest error rate across the evaluated robustness slices in Samruddhi's experiments. The main trade-off was higher training cost compared with the simpler baseline.

For future work, the team would test head-and-tail truncation, subword tokenization, preservation of emoticons and rating expressions, improved calibration, multiple random seeds, and an ensemble combining recurrent and convolutional representations. These changes could improve performance on long, sarcastic, mixed-sentiment, and contrast-heavy reviews.

