# Task 1 Source Code

`train_task1.py` contains my complete character-level language-modeling
implementation. The attention scores are calculated explicitly and masked
with a lower-triangular causal mask. The file also contains the data loading,
training loop, validation evaluation, text generation, loss-curve export, and
Task 1 metric calculations.

The script supports a short smoke test for debugging and a full 10-epoch run.
The full experiment configuration and generated evidence are stored in the
member directory rather than being hard-coded into the report.
