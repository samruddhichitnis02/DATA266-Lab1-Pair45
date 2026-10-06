# Task 3 Failure and Stability Analysis

The training log and curves are the evidence for convergence and stability.
After training, I will describe at least three observed issues using actual
generated images or loss-curve evidence. For each issue I will record the
observation and one testable fix.

Suggested categories to inspect are:

1. mode collapse or repeated visual patterns;
2. weak style transfer while content is preserved;
3. content distortion, checkerboard/artifact patterns, or color shifts;
4. generator/discriminator imbalance or unstable cycle loss;
5. NaN/gradient spikes.

The final report must use the actual outputs from `pred_A2B/` and `pred_B2A/`,
not invented examples. The two-rater human audit is documented separately in
`human_audit.csv` and summarized with Cohen's kappa in
`full_metrics_report.csv`.
