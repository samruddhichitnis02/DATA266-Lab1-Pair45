# Task 1 Results — Nikhil Kanaparthi

## Model and preprocessing

I trained a character-level decoder-only GPT model on TinyStories. I created
the character vocabulary with `char_to_idx` and `idx_to_char`, converted each
story into integer tokens, and sampled fixed-length next-character prediction
windows with a context length of 256. I used 100,000 stories for training and
10,000 stories for validation.

The model has four decoder blocks, 256 hidden dimensions, four attention heads,
pre-LayerNorm, causal self-attention, GELU feed-forward layers, residual
connections, and tied input/output embeddings. I trained it with AdamW at a
learning rate of 3e-4, 0.1 weight decay, 1.0 gradient clipping, 10% warm-up,
and cosine learning-rate decay.

## Final evaluation

The model was trained for 10 epochs on Apple MPS. It contains 3,254,016
trainable parameters and required 5,561.93 seconds, approximately 92.7
minutes, for training.

| Metric | Value |
|---|---:|
| Training cross-entropy loss | 0.8006 |
| Validation cross-entropy loss | 0.7448 |
| Perplexity | 2.1060 |
| Bits per character | 1.0745 |
| Generalization gap | -0.0558 |
| Top-1 next-character accuracy | 76.49% |
| Greedy Distinct-1 | 0.0918 |
| Greedy Distinct-2 | 0.4254 |
| Greedy Distinct-3 | 0.6274 |
| Greedy repeated 4-gram rate | 0.2907 |
| Sampled Distinct-1 | 0.1076 |
| Sampled Distinct-2 | 0.4730 |
| Sampled Distinct-3 | 0.7293 |
| Sampled repeated 4-gram rate | 0.1597 |
| Mean gradient norm | 0.6811 |
| Maximum gradient norm | 7.2606 |
| NaN count | 0 |
| Parameter count | 3,254,016 |
| Training throughput | 46,027.20 tokens/sec |
| Generation throughput | 313.06 tokens/sec |
| Total training time | 5,561.93 seconds |
| Peak memory | Not available through the PyTorch MPS peak-memory API |
| Hardware | Apple MPS |

The validation loss was slightly lower than the training loss. This is
consistent with dropout being active during training and disabled during
validation. The negative generalization gap does not indicate obvious
overfitting in this run. The zero NaN count and stable gradient statistics also
show that the optimization remained numerically stable.

## Generated text

The greedy sample produced a recognizable children's story about Lily, her
mother, and a park. However, it repeated the phrase “wanted to play with it.”
The temperature-based sample was more diverse and introduced dialogue, but it
also contained awkward grammar and repeated references to a button. The full
outputs are stored in `outputs/greedy.txt` and `outputs/sampled.txt`, and the
loss plot is stored in `outputs/loss_curves.png`.

## Limitations and next steps

The model learned local spelling and story patterns effectively, but its
long-range narrative consistency remains limited. A larger context length,
additional training data, more training steps, and repetition-aware sampling
would be reasonable next experiments. Since MPS does not expose the same peak
memory API as CUDA in this environment, peak memory is reported as unavailable
rather than estimated or fabricated.
