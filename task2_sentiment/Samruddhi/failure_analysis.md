# Robustness by data slice

| metric | baseline_meanmax_mlp | exp1_transformer | exp2_bilstm_attn |
|---|---|---|---|
| slice_macroF1 | short (<=q25 words) | 0.9340 | 0.9345 | 0.9514 |
| slice_err | short (<=q25 words) | 0.0635 | 0.0632 | 0.0465 |
| slice_macroF1 | long (>=q75 words) | 0.9295 | 0.9334 | 0.9513 |
| slice_err | long (>=q75 words) | 0.0667 | 0.0633 | 0.0465 |
| slice_macroF1 | has negation | 0.9279 | 0.9309 | 0.9551 |
| slice_err | has negation | 0.0695 | 0.0666 | 0.0436 |
| slice_macroF1 | has contrast (but/however/although) | 0.9251 | 0.9279 | 0.9522 |
| slice_err | has contrast (but/however/although) | 0.0740 | 0.0712 | 0.0474 |
| slice_macroF1 | exclamation-heavy (>=3 '!') | 0.9507 | 0.9534 | 0.9710 |
| slice_err | exclamation-heavy (>=3 '!') | 0.0481 | 0.0454 | 0.0281 |
