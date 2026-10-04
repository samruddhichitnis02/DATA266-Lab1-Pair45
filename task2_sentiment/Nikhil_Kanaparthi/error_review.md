# Task 2 Manual Error Review

The script creates `error_review.csv` using the experimental BiGRU's test
predictions. It contains exactly 20 rows: five rows in each required group.
The candidate-selection rule is recorded in the source code and the original
prediction data are retained in `metrics/all_predictions.json`.

| Group | Count | What to check |
|---|---:|---|
| `confident_false_positive` | 5 | True negative, predicted positive with high probability |
| `confident_false_negative` | 5 | True positive, predicted negative with high probability |
| `near_threshold_error` | 5 | Incorrect prediction with probability close to 0.50 |
| `slice_specific_failure` | 5 | Incorrect prediction from a long-review or negation slice |

For every row, read the complete review text and fill in:

1. `manual_observation`: what linguistic or contextual pattern caused the
   mistake;
2. `testable_fix`: one concrete change that could be tested in a new run.

Do not replace the review text or labels. The completed CSV is the evidence for
the manual error-review requirement.
