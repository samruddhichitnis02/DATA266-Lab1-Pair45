# Task 1 — Character-Level GPT

## Objective

The objective was to build a character-level GPT model from scratch using the TinyStories dataset. The model learns to predict the next character based on the characters that came before it.

## Dataset and Preprocessing

- Dataset: TinyStories
- Tokenization: Character-level
- Training and validation data were converted into integer token IDs.
- Fixed-length sequences were used for autoregressive language modeling.
- Context length: 512 characters
- Random fixed-length sequences were used during training.
- Target sequences were shifted by one character.

## Model Architecture

- Model type: Decoder-only GPT
- Transformer blocks: 6
- Embedding size: 384
- Attention heads per block: 6
- Dimension of each attention head: 64
- Feed-forward network: 384 → 1536 → 384
- Activation function: GELU
- Attention type: Causal multi-head self-attention
- Normalization: Layer normalization
- Connections: Residual connections
- Embeddings: Learnable character and positional embeddings
- Loss function: Cross-entropy loss
- Optimizer: AdamW
- Total parameters: 10,927,104
- Trainable parameters: 10,927,104

## Training

The model was trained for 21 epochs. Training was stopped because the training and validation losses became nearly flat and showed very little improvement.

This indicated that the model had reached a learning plateau and that continuing toward 1,000 epochs would likely provide limited additional benefit.

## Evaluation Results

- Training cross-entropy loss: 0.4805
- Validation cross-entropy loss: 0.5496
- Training perplexity: 1.6169
- Validation perplexity: 1.7326
- Validation bits-per-character: 0.7929
- Generalization gap: 0.0691
- Training top-1 accuracy: 84.52%
- Validation top-1 accuracy: 82.62%

The difference between training and validation accuracy was only 1.90 percentage points. The small generalization gap also suggests that the model performed similarly on unseen validation data and did not show severe overfitting.

## Generated Text

The model was given the prompt:

> Once upon a time

The model generated readable story-like text with basic grammar, punctuation, names, and dialogue.

Example generated text:

> Once upon a time, there was a little boy named Timmy. He loved to write with his jewels. He would spread them on the paper every day. One day, Timmy wrote a letter to his friend, Timmy. Timmy asked, "What kind of person do you write letters?" Timmy replied, "I write how fast I can write letters and stories."

However, the generated text also contained repetition, unusual phrases, and an incomplete ending.

The generated sample was saved in:

```text
outputs/generated_sample_temperature_0_8.txt
```

## Learning-Rate Scheduling Experiment

The original model used a fixed learning rate for 21 epochs. A separate
experiment was then performed using the existing trained model as the starting
checkpoint. The model architecture, dataset, batch size, optimizer type, and
weight decay were kept the same.

The experiment used:

- Starting checkpoint: Best model from the original 21-epoch run
- Additional training: 11 epochs
- Optimizer: AdamW
- Initial learning rate: 0.0003
- Warm-up period: 2 epochs
- Learning-rate schedule: Linear warm-up followed by cosine decay
- Weight decay: 0.01
- Gradient clipping: 1.0

The scheduled model achieved its best validation loss during experiment epoch
10. The experiment did not restart training from random weights; it continued
from the previously trained model.

## Comparison with the Original Model

| Metric | Original model | Scheduled experiment |
|---|---:|---:|
| Training setup | 21 epochs, fixed learning rate | 11 additional epochs, warm-up and cosine decay |
| Best validation loss | 0.5496 | 0.5280 |
| Final validation loss | 0.5496 | 0.5281 |
| Total training epochs | 21 | 32 cumulative epochs |

The scheduled experiment reduced the best validation loss from 0.5496 to
0.5280. This is an improvement of approximately 0.0216, or 3.94%. The result
indicates that learning-rate warm-up and cosine decay helped the model improve
its performance on unseen validation data.

## Scheduled Experiment Performance

- Total experiment training time: 2,551.36 seconds, approximately 42.52 minutes
- Training throughput: 387,965.63 tokens per second
- Peak GPU memory during the experiment: 3.911 GB
- Best scheduled-experiment validation loss: 0.52797
- Best scheduled-experiment epoch: 10 of 11

The experiment metrics were saved in:

```text
outputs/learning_rate_experiment/learning_rate_experiment_metrics.csv
```

The experiment checkpoint was saved in:

```text
checkpoints/learning_rate_experiment/best_model_with_scheduler.pt
```

## Conclusion

The final selected model is the best checkpoint from the learning-rate
scheduling experiment because it achieved the lowest validation loss. The
experiment also satisfies the requirement to evaluate learning-rate warm-up
and scheduling. The original model and the scheduled model used the same GPT
architecture, so the improvement is attributed to the training schedule
rather than a change in model size or structure.
