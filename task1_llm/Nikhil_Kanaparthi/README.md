# DATA266 Lab 1 — Task 1: GPT-Style LLM From Scratch

This folder contains one independent member implementation for Task 1. It uses
character-level TinyStories data and implements the Transformer components
manually. No `nn.Transformer`, `nn.MultiheadAttention`, or pretrained model is
used.

## Run in Google Colab or a GPU machine

```bash
pip install -r requirements.txt
python src/train_task1.py --epochs 10 --train-stories 100000 --val-stories 10000
```

For a quick smoke test:

```bash
python src/train_task1.py --smoke-test
```

The full run writes checkpoints, loss curves, generated samples, raw logs,
metrics, and timing information under this folder. Do not edit the raw log
after training.

## Main design

- Character-level encoding with `char_to_idx` and `idx_to_char`.
- 4 custom decoder-only Transformer blocks.
- 4 attention heads, model width 256, context length 256.
- Pre-LayerNorm, residual connections, causal self-attention, and a 4x GELU
  feed-forward network.
- AdamW with linear warm-up followed by cosine decay.
- Greedy and temperature-based sampling.

The exact values are stored in `config.json` after each run. The model is
intentionally different from a default tutorial configuration so it can be
compared against teammates' models.
