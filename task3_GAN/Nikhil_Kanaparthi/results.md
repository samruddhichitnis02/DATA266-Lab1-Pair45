# Task 3 Results — Nikhil Kanaparthi

This file is completed after the Colab training, evaluation, human audit, and
Kaggle submission. The numerical results are generated into
`full_metrics_report.csv` and `submission.csv`; they must not be entered by
guessing.

## Architecture and training choices

- Domain A: Monet; Domain B: Photo.
- Generators: ResNet generators with two downsampling layers, residual blocks,
  and two upsampling layers.
- Discriminators: 70x70-style PatchGAN discriminators.
- Losses: least-squares adversarial loss, cycle-consistency L1 loss, and
  identity L1 loss.
- Optimizer: Adam with learning rate 0.0002 and betas (0.5, 0.999).
- Cycle-loss weight: 10; identity-loss weight: 5.

## Metrics and evidence

Copy the final table from `full_metrics_report.csv` and link the corresponding
loss curves, generated-image folders, checkpoint, raw log, and human-audit
worksheet.

## Kaggle

Record the public score, final/private score if available, and leaderboard rank
only after submitting the direct output of this trained CycleGAN.
