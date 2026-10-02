# DATA266 Lab 1 — Task 1: GPT-Style LLM From Scratch

This project implements a small decoder-only GPT-style language model trained
on TinyStories at the character level. I implemented the attention mechanism,
causal mask, LayerNorm, feed-forward network, residual connections, token
embeddings, positional embeddings, and language-modeling head directly in
PyTorch. I did not use `nn.Transformer`, `nn.MultiheadAttention`, pretrained
embeddings, or a pretrained language model.

## My model

The model uses four decoder blocks with a model width of 256 and four attention
heads. Each block uses pre-LayerNorm, causal multi-head self-attention, a GELU
feed-forward network, and residual connections. The context length is 256
characters, and the input and output token embeddings are tied.

I trained the model with AdamW using a learning rate of 3e-4, weight decay of
0.1, gradient clipping at 1.0, 10% learning-rate warm-up, and cosine decay.
The training split contained 100,000 stories and the validation split
contained 10,000 stories. The complete configuration is recorded in
`config.json`.

## Reproducibility

The implementation can be reproduced with:

```bash
pip install -r requirements.txt
python src/train_task1.py --epochs 10 --train-stories 100000 --val-stories 10000
```

The final run was performed on Apple MPS. Checkpoints, raw logs, generated
samples, loss curves, and evaluation metrics are retained with the project so
that the reported results can be traced to the run that produced them.
