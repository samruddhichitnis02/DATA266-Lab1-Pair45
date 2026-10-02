# Nikhil Kanaparthi — Task 1 Reproducibility Manifest

## Dataset

- Dataset: `roneneldan/TinyStories`
- Tokenization: Character level
- Training stories: 100,000
- Validation stories: 10,000
- Context length: 256
- Random seed: 2661

## Model

- Architecture: Decoder-only GPT-style Transformer
- Transformer blocks: 4
- Model dimension: 256
- Attention heads: 4
- Feed-forward activation: GELU
- Normalization: Pre-LayerNorm
- Dropout: 0.10
- Parameter count: 3,254,016
- Pretrained model: Not used
- Prebuilt attention module: Not used

## Training

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.1
- Warm-up fraction: 0.10
- Gradient clipping: 1.0
- Epochs: 10
- Temperature for sampling: 0.8

## Hardware

- Device: Apple MPS
- Peak memory: Not available through the PyTorch MPS peak-memory API

## Reproduction command

```bash
pip install -r requirements.txt
python src/train_task1.py --epochs 10 --train-stories 100000 --val-stories 10000