## Task 1: Character-Level GPT on TinyStories

This project implements a character-level decoder-only GPT language model from scratch using the TinyStories dataset. The model learns to predict the next character from the characters that appear before it.


## Objective

The objective was to build, train, evaluate, and analyze a small GPT-style language model without using a prebuilt Transformer or attention layer.

For example:

```text
Input:  hell
Target: ello
```

## Dataset

The model was trained on TinyStories:

- Training set: 100,000 stories
- Validation set: 10,000 stories
- Tokenization: Character-level
- Context length: 512 characters
- Training examples: Random fixed-length character sequences
- Target: Input sequence shifted by one character

The input and target follow this pattern:

```text
Input:  characters 0–511
Target: characters 1–512
```

## Preprocessing

The preprocessing pipeline:

1. Loads the TinyStories training and validation files.
2. Removes empty examples.
3. Builds a character vocabulary using training data only.
4. Assigns each character an integer ID.
5. Adds an `<UNK>` token for unseen characters.
6. Encodes the text as PyTorch tensors.
7. Saves the processed data for model training.

Generated preprocessing files include:

```text
vocabulary.json
train_tokens.pt
val_tokens.pt
```

## Model Architecture

The model is a custom decoder-only GPT implemented in PyTorch.

| Component | Configuration |
|---|---:|
| Model type | Character-level decoder-only GPT |
| Context length | 512 characters |
| Transformer blocks | 6 |
| Embedding size | 384 |
| Attention heads | 6 |
| Dimension per head | 64 |
| Feed-forward network | 384 → 1536 → 384 |
| Activation | GELU |
| Attention | Causal multi-head self-attention |
| Normalization | Layer normalization |
| Connections | Residual connections |
| Embeddings | Learnable token and positional embeddings |
| Loss | Cross-entropy |
| Optimizer | AdamW |

The causal attention mask prevents a character from using information from future characters.

The model contains:

```text
Total parameters: 10,927,104
Trainable parameters: 10,927,104
```

## Main Source Files

| File | Purpose |
|---|---|
| `preprocessing.py` | Cleans text, builds the vocabulary, and creates token tensors |
| `dataset.py` | Creates input-target character sequences |
| `dataloaders.py` | Creates training and validation DataLoaders |
| `model.py` | Defines the GPT model, attention, Transformer blocks, loss, and generation |
| `model_training.ipynb` | Runs verbose baseline training and evaluation |
| `learning_rate_experiment.ipynb` | Runs the scheduled-learning-rate experiment |

## Baseline Training

The baseline model was trained for 21 epochs using AdamW with a fixed learning rate of `3e-4`.

Training was stopped after 21 epochs because the training and validation losses became nearly flat. This indicated that the model had reached a learning plateau, so continuing toward 1,000 epochs was unlikely to provide significant improvement.

### Baseline Results

| Metric | Result |
|---|---:|
| Training loss | 0.4805 |
| Validation loss | 0.5496 |
| Training perplexity | 1.6169 |
| Validation perplexity | 1.7326 |
| Validation bits-per-character | 0.7929 |
| Training top-1 accuracy | 84.52% |
| Validation top-1 accuracy | 82.62% |
| Generalization gap | 0.0691 |

## Learning-Rate Scheduling Experiment

Because learning-rate warm-up and scheduling were required, a separate experiment was performed using the best baseline checkpoint as the starting point.

The experiment did not restart from random weights. It continued training a separate copy of the existing model for 11 additional epochs.

Experiment configuration:

- Optimizer: AdamW
- Initial learning rate: `0.0003`
- Warm-up: 2 epochs
- Warm-up method: Linear increase
- Schedule after warm-up: Cosine decay
- Weight decay: `0.01`
- Gradient clipping: `1.0`
- Additional training: 11 epochs

### Experiment Results

| Metric | Baseline | Scheduled experiment |
|---|---:|---:|
| Training setup | 21 fixed-LR epochs | 11 additional scheduled epochs |
| Best validation loss | 0.5496 | 0.5280 |
| Final validation loss | 0.5496 | 0.5281 |
| Cumulative training epochs | 21 | 32 |

The scheduled experiment improved the best validation loss from `0.5496` to `0.5280`.

```text
Absolute improvement: 0.0216
Relative improvement: approximately 3.94%
Best experiment epoch: 10 of 11
```

### Experiment Performance

- Total experiment training time: approximately 42.52 minutes
- Training throughput: 387,965.63 tokens per second
- Peak GPU memory during training: 3.911 GB

## Text Generation

The model was tested using the prompt:

```text
Once upon a time
```

The baseline model generated readable story-like text with basic grammar, punctuation, names, and dialogue. It also showed repetition, unusual phrases, and an incomplete ending.

Example issues included:

- Repeating the name “Timmy”
- The unusual phrase “write with his jewels”
- An abrupt and incomplete ending

The generated sample was saved in:

```text
outputs/generated_sample_temperature_0_8.txt
```

Generation performance from the recorded evaluation:

- Generated characters: 100
- Generation time: 0.373 seconds
- Generation speed: 268.09 characters per second
- Peak generation memory: 0.281 GB

## Failure Analysis

The main failure cases were:

1. **Repetition:** The model repeated names and similar sentence patterns.
2. **Semantic inconsistency:** Some phrases were grammatically possible but did not make logical sense.
3. **Incomplete generation:** The generated story ended abruptly without a clear conclusion.

These findings are documented in:

```text
failure_analysis.md
```

## Project Outputs

Important outputs include:

```text
Results.md
metrics_report.csv
learning_rate_experiment_metrics.csv
failure_analysis.md
```

Important checkpoints include:

```text
checkpoints/best_model.pt
checkpoints/learning_rate_experiment/best_model_with_scheduler.pt
```

Important visualizations include:

```text
outputs/story_length_distribution.png
outputs/word_count_distribution.png
outputs/top_30_character_frequency.png
outputs/training_validation_loss_curves.png
outputs/learning_rate_experiment/scheduler_loss_curves.png
outputs/learning_rate_experiment/learning_rate_schedule.png
```

## Conclusion

Task 1 successfully implemented a character-level GPT from scratch, trained it on TinyStories, evaluated its language-modeling performance, generated sample text, analyzed failure cases, and measured training and generation performance.

The learning-rate scheduling experiment improved validation loss without changing the model architecture. The best scheduled-training checkpoint is therefore the preferred model checkpoint for Task 1.

Large raw datasets and generated binary files should remain excluded from GitHub according to the project `.gitignore` file. The repository should contain the source code, notebooks, reports, metrics, selected plots, and required checkpoints.
