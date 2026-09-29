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