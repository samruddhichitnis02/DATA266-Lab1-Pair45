# Task 2 Error Analysis

Task 2's required failure analysis is a manual review of 20 test errors from
the experimental BiGRU model. The generated worksheet is `error_review.csv`.
It contains five confident false positives, five confident false negatives,
five near-threshold errors, and five slice-specific failures.

For each row, I will read the complete review text and record:

- the linguistic or contextual reason the prediction was difficult;
- one concrete, testable change that could reduce that error type.

The original predictions, labels, probabilities, and model name are preserved
in `metrics/all_predictions.json`. The candidate-selection process does not
replace the required human judgment: `manual_observation` and `testable_fix`
must be completed after inspecting the examples.
