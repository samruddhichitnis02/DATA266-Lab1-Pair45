# DATA266 Lab 1 — Task 3: CycleGAN Style Transfer

This member folder contains an independently trained CycleGAN. Domain A is
Monet and Domain B is Photo. The model contains two generators and two
PatchGAN discriminators trained from scratch; no pretrained image model is
used to generate or modify submitted images.

## Required data layout

Keep the shared dataset outside this member folder when possible:

```text
task3_gan/data/
├── monet_jpg/
└── photo_jpg/
```

The supplied export contains 300 Monet images and 7,038 photo images. Do not
commit the raw images to GitHub.

## Colab commands

From the member folder:

```bash
pip install -q -r requirements.txt
python src/train_cyclegan.py --smoke-test --data-root ../data
python src/train_cyclegan.py --data-root ../data --epochs 10 --decay-start-epoch 50 --checkpoint-every 10 2>&1 | tee raw_logs/task3_epochs_01_10.log
python src/train_cyclegan.py --data-root ../data --epochs 100 --decay-start-epoch 50 --resume checkpoints/cyclegan_epoch_010.pt --checkpoint-every 10 2>&1 | tee raw_logs/task3_epochs_11_100.log
python src/make_human_audit.py --data-root ../data
python src/evaluate_task3.py --data-root ../data --n-eval 300
```

For Colab, set the runtime to GPU before running the commands. The evaluation
script downloads ImageNet Inception weights only for FID/KID-style evaluation;
those weights are not used by the CycleGAN generators or by the submitted
images.

## Required outputs

- `checkpoints/cyclegan_final.pt`: final model checkpoint.
- `outputs/pred_A2B/`: Monet-to-Photo translations.
- `outputs/pred_B2A/`: Photo-to-Monet translations.
- `outputs/training_curves.png`: generator, discriminator, cycle, identity,
  and gradient curves.
- `full_metrics_report.csv`: all local Task 3 metrics.
- `submission.csv`: instructor-compatible FID/MiFID summary.
- `human_audit.csv` and `outputs/human_audit_sheet.png`: 30 fixed samples for
  two independent raters.
- `run_manifest.json`, `config.json`, `history.json`, and `raw_logs/`: evidence
  and reproducibility information.



