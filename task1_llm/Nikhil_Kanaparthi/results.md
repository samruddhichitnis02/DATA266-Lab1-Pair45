# Task 1 Results — Nikhil Kanaparthi

Run this file only after the training run. Replace each placeholder using the
generated `metrics_report.json`, `config.json`, loss plot, raw log, and text
outputs. Keep the raw log unchanged.

## Model and preprocessing

- Dataset: TinyStories, character level.
- Member-specific split: 100,000 training stories and 10,000 validation stories.
- Context length: 256 characters.
- Architecture: 4 decoder-only blocks, 256 hidden dimensions, 4 heads,
  pre-LayerNorm, causal self-attention, GELU feed-forward network, residual
  connections, tied input/output embeddings.
- Optimizer/schedule: AdamW, learning rate 3e-4, 10% warm-up, cosine decay.
- Hardware: record the exact GPU/CPU here.

## Metrics

Copy the final values from `metrics_report.json` into the team's comparison
table. Required metrics include train/validation cross-entropy, perplexity,
bits-per-character, generalization gap, top-1 accuracy, Distinct-1/2/3,
repeated 4-gram rate, gradient statistics, parameter count, throughput, peak
memory, and training time.

## Generation examples

See `outputs/greedy.txt` and `outputs/sampled.txt`. Paste representative
snippets into the report and reference the corresponding checkpoint.

## Three failure cases

Complete `failure_analysis.md` with three actual snippets from the generated
files. For each, label the failure type, explain what happened, and propose a
testable improvement.
