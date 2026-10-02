# DATA266 Lab 1 - Task 1: GPT-Style LLM From Scratch
This folder contains Nikhil Kanaparthi's independent Task 1 implementation. It uses character-level TinyStories data and manually implements the GPT-style Transformer components. No prebuilt Transformer or attention module is used.
## Run
```bash
pip install -r requirements.txt
python src/train_task1.py --smoke-test
python src/train_task1.py --epochs 10 --train-stories 100000 --val-stories 10000
```
The completed run saves checkpoints, raw logs, loss curves, generated samples, metrics, and timing information in this folder.
