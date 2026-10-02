# How to run

From this member folder:

```bash
pip install -r requirements.txt
python src/train_task1.py --smoke-test
python src/train_task1.py --epochs 10 --train-stories 100000 --val-stories 10000
```

The smoke test checks the end-to-end pipeline. Only the second command is the
required training run for the report.
