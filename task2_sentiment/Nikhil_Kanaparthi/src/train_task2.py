"""DATA266 Lab 1, Task 2: Yelp Polarity sentiment classification.

The experiment deliberately uses no pretrained language model and no
pretrained embedding. The vocabulary and all embedding weights are learned
from the Yelp training split. Three models are trained on the same split:

1. mean-pooled learned embeddings + linear classifier (baseline)
2. learned embeddings + multi-kernel 1-D CNN (experimental)
3. learned embeddings + bidirectional GRU (experimental)

The script writes preprocessing analysis, training curves, checkpoints,
metrics_report.csv, metrics_report.json, predictions, McNemar tests, data
slice metrics, and a 20-row manual error-review worksheet.

Run from this member folder:
    python src/train_task2.py --smoke-test
    python src/train_task2.py --epochs 5 2>&1 | tee raw_logs/task2_full_run.log
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset


# The member folder is the portable root of this experiment.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
METRICS_DIR = ROOT / "metrics"
CHECKPOINT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "raw_logs"
SEED = 266


NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "cannot",
    "isn't",
    "wasn't",
    "weren't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "couldn't",
    "shouldn't",
}

# A small built-in list keeps preprocessing reproducible without downloading
# an external corpus. Negation words are deliberately retained.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "here", "hers", "him", "his",
    "how", "i", "if", "in", "into", "it", "its", "itself", "me", "my", "of",
    "on", "or", "our", "ours", "she", "so", "that", "the", "their", "them",
    "there", "these", "they", "this", "those", "to", "was", "we", "were", "what",
    "when", "where", "which", "who", "why", "with", "you", "your", "yours",
}
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def seed_everything(seed: int) -> None:
    """Set all random seeds used by this experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, and finally CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before timing or saving."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def accelerator_memory_mb(device: torch.device) -> float | None:
    """Return the largest available accelerator memory estimate in MB."""
    values: List[float] = []
    if device.type == "cuda":
        values.append(float(torch.cuda.max_memory_allocated(device)) / (1024**2))
        values.append(float(torch.cuda.max_memory_reserved(device)) / (1024**2))
    elif device.type == "mps" and hasattr(torch, "mps"):
        for method_name in ("max_memory_allocated", "driver_allocated_memory", "current_allocated_memory"):
            method = getattr(torch.mps, method_name, None)
            if method is not None:
                try:
                    values.append(float(method()) / (1024**2))
                except Exception:
                    pass
    return max(values) if values else None


def reset_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        reset = getattr(torch.mps, "reset_peak_memory_stats", None)
        if reset is not None:
            try:
                reset()
            except Exception:
                pass


def normalize_label(value: int) -> int:
    """Support both the HF 0/1 labels and the original CSV 1/2 labels."""
    value = int(value)
    return value - 1 if value in (1, 2) else value


def download_yelp_csv() -> Tuple[Path, Path]:
    """Download Yelp Polarity once, using the standard public CSV release."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_csv = DATA_DIR / "train.csv"
    test_csv = DATA_DIR / "test.csv"
    if train_csv.exists() and test_csv.exists():
        return train_csv, test_csv

    archive = DATA_DIR / "yelp_review_polarity_csv.tgz"
    url = "https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz"
    print(f"Downloading Yelp Polarity data from {url}")
    urllib.request.urlretrieve(url, archive)
    with tarfile.open(archive, "r:gz") as tar:
        try:
            tar.extractall(DATA_DIR, filter="data")
        except TypeError:  # compatible with older Python versions
            tar.extractall(DATA_DIR)
    extracted = DATA_DIR / "yelp_review_polarity_csv"
    extracted_train = extracted / "train.csv"
    extracted_test = extracted / "test.csv"
    extracted_train.replace(train_csv)
    extracted_test.replace(test_csv)
    return train_csv, test_csv


def preprocess_text(text: str) -> List[str]:
    """Lowercase, remove punctuation/special characters, tokenize, and stopword-filter.

    Negation words are retained because removing them would erase sentiment
    information. No stemming/lemmatization is applied: preserving the original
    word forms makes the experiment deterministic and keeps error analysis
    interpretable.
    """
    lowered = str(text).lower()
    tokens = TOKEN_RE.findall(lowered)
    return [token for token in tokens if token not in STOPWORDS or token in NEGATION_WORDS]


def read_reviews(path: Path, limit: int | None = None) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Read a raw Yelp CSV and return cleaned text plus preprocessing counts."""
    raw = pd.read_csv(path, header=None, names=["label", "text"], nrows=limit)
    original_count = len(raw)
    missing_count = int(raw[["label", "text"]].isna().any(axis=1).sum())
    raw = raw.dropna(subset=["label", "text"]).copy()
    malformed_count = 0
    labels: List[int] = []
    for value in raw["label"]:
        try:
            labels.append(normalize_label(int(value)))
        except (TypeError, ValueError):
            labels.append(-1)
            malformed_count += 1
    raw["label"] = labels
    raw = raw[raw["label"].isin([0, 1])].copy()
    raw["tokens"] = raw["text"].map(preprocess_text)
    empty_count = int((raw["tokens"].map(len) == 0).sum())
    raw = raw[raw["tokens"].map(len) > 0].reset_index(drop=True)
    raw["token_count"] = raw["tokens"].map(len)
    counts = {
        "original_rows": int(original_count),
        "missing_rows": missing_count,
        "malformed_label_rows": malformed_count,
        "empty_after_preprocessing": empty_count,
        "final_rows": int(len(raw)),
    }
    return raw, counts


def stratified_split(df: pd.DataFrame, validation_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Make a deterministic stratified train/validation split."""
    rng = np.random.default_rng(seed)
    train_indices: List[int] = []
    validation_indices: List[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(df["label"].to_numpy() == label)
        rng.shuffle(indices)
        n_validation = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:n_validation].tolist())
        train_indices.extend(indices[n_validation:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return df.iloc[train_indices].reset_index(drop=True), df.iloc[validation_indices].reset_index(drop=True)


def build_vocabulary(df: pd.DataFrame, max_vocab: int, min_frequency: int) -> Dict[str, int]:
    counts = Counter(token for row in df["tokens"] for token in row)
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common():
        if count < min_frequency or len(vocabulary) >= max_vocab:
            break
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_texts(df: pd.DataFrame, vocabulary: Dict[str, int], max_length: int) -> np.ndarray:
    encoded = np.zeros((len(df), max_length), dtype=np.int64)
    for row_index, tokens in enumerate(df["tokens"]):
        ids = [vocabulary.get(token, 1) for token in tokens[:max_length]]
        if ids:
            encoded[row_index, : len(ids)] = ids
    return encoded


class EncodedReviews(Dataset):
    def __init__(self, encoded: np.ndarray, labels: Sequence[int]):
        self.encoded = torch.from_numpy(encoded)
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.encoded[index], self.labels[index], index


class MeanEmbeddingClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_dim, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens != 0).unsqueeze(-1)
        embedded = self.embedding(tokens)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(self.dropout(pooled))


class MultiKernelCNNClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, channels: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.convolutions = nn.ModuleList(
            [nn.Conv1d(embedding_dim, channels, kernel_size, padding=kernel_size // 2) for kernel_size in (3, 5, 7)]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 3, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens).transpose(1, 2)
        features = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convolutions]
        return self.classifier(self.dropout(torch.cat(features, dim=1)))


class BiGRUClassifier(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(tokens)
        sequence, _ = self.gru(embedded)
        lengths = (tokens != 0).sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device)
        last_valid = sequence[batch_indices, lengths]
        return self.classifier(self.dropout(last_valid))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def gradient_norm(model: nn.Module) -> float:
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared += float(parameter.grad.detach().norm(2).item() ** 2)
    return math.sqrt(squared)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for tokens, labels, _ in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        total_loss += float(loss_function(logits, labels).item()) * len(labels)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
) -> Tuple[Dict[str, List[float]], float, float, float, int]:
    """Train one model and return history, time, memory, mean/max gradient norm, NaNs."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    loss_function = nn.CrossEntropyLoss()
    history: Dict[str, List[float]] = {"epoch": [], "train_loss": [], "validation_loss": [], "train_accuracy": [], "validation_accuracy": [], "learning_rate": []}
    gradients: List[float] = []
    nan_count = 0
    reset_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for tokens, labels, _ in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = loss_function(logits, labels)
            if not torch.isfinite(loss):
                nan_count += 1
                continue
            loss.backward()
            current_gradient = gradient_norm(model)
            if not math.isfinite(current_gradient):
                nan_count += 1
            else:
                gradients.append(current_gradient)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += len(labels)

        scheduler.step()
        validation_loss, validation_accuracy = evaluate_loss(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "validation_loss": validation_loss,
            "train_accuracy": correct / max(total, 1),
            "validation_accuracy": validation_accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        for key in history:
            history[key].append(record[key])
        print(
            f"{name} epoch {epoch}/{epochs} "
            f"train_loss={record['train_loss']:.4f} val_loss={validation_loss:.4f} "
            f"train_acc={record['train_accuracy']:.4f} val_acc={validation_accuracy:.4f}"
        )

    synchronize(device)
    elapsed = time.perf_counter() - start
    memory = accelerator_memory_mb(device)
    mean_gradient = float(np.mean(gradients)) if gradients else float("nan")
    max_gradient = float(np.max(gradients)) if gradients else float("nan")
    return history, elapsed, memory, mean_gradient, max_gradient, nan_count


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: List[int] = []
    probabilities: List[float] = []
    indices: List[int] = []
    for tokens, batch_labels, batch_indices in loader:
        logits = model(tokens.to(device))
        probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
        labels.extend(batch_labels.numpy().tolist())
        indices.extend(batch_indices.numpy().tolist())
    return np.asarray(labels), np.asarray(probabilities), np.asarray(indices)


def bootstrap_interval(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values: List[float] = []
    for _ in range(samples):
        sample_indices = rng.integers(0, len(labels), size=len(labels))
        try:
            value = float(metric(labels[sample_indices], predictions[sample_indices]))
            if math.isfinite(value):
                values.append(value)
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities <= high if high == 1 else probabilities < high)
        if mask.any():
            total += float(mask.mean()) * abs(float(labels[mask].mean()) - float(probabilities[mask].mean()))
    return total


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    model_name: str,
    train_seconds: float,
    examples_seen: int,
    device: torch.device,
    model_parameters: int,
    peak_memory_mb: float | None,
    gradient_mean: float,
    gradient_max: float,
    nan_count: int,
    bootstrap_samples: int,
    seed: int,
    slice_values: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1]).ravel().tolist()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, predictions, average="micro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=0)
    macro_f1_function = lambda y, p: f1_score(y, p, labels=[0, 1], average="macro", zero_division=0)
    accuracy_ci = bootstrap_interval(labels, predictions, accuracy_score, bootstrap_samples, seed)
    f1_ci = bootstrap_interval(labels, predictions, macro_f1_function, bootstrap_samples, seed + 1)
    mcc_ci = bootstrap_interval(labels, predictions, matthews_corrcoef, bootstrap_samples, seed + 2)
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_class_0": float(precision[0]),
        "precision_class_1": float(precision[1]),
        "precision_macro": float(precision.mean()),
        "precision_micro": float(precision_micro),
        "precision_weighted": float(precision_weighted),
        "recall_class_0": float(recall[0]),
        "recall_class_1": float(recall[1]),
        "recall_macro": float(recall.mean()),
        "recall_micro": float(recall_micro),
        "recall_weighted": float(recall_weighted),
        "f1_class_0": float(f1[0]),
        "f1_class_1": float(f1[1]),
        "f1_macro": float(f1.mean()),
        "f1_micro": float(f1_micro),
        "f1_weighted": float(f1_weighted),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "expected_calibration_error": expected_calibration_error(labels, probabilities),
        "confusion_matrix_tn_fp_fn_tp": cm,
        "accuracy_ci95_low": accuracy_ci[0],
        "accuracy_ci95_high": accuracy_ci[1],
        "macro_f1_ci95_low": f1_ci[0],
        "macro_f1_ci95_high": f1_ci[1],
        "mcc_ci95_low": mcc_ci[0],
        "mcc_ci95_high": mcc_ci[1],
        "parameter_count": int(model_parameters),
        "training_time_seconds": float(train_seconds),
        "examples_per_second": float(examples_seen / max(train_seconds, 1e-9)),
        "peak_memory_mb": peak_memory_mb,
        "device": str(device),
        "gradient_norm_mean": gradient_mean,
        "gradient_norm_max": gradient_max,
        "nan_count": int(nan_count),
        "slice_metrics": slice_values,
    }


def slice_metrics(labels: np.ndarray, probabilities: np.ndarray, texts: Sequence[str]) -> Dict[str, Dict[str, float]]:
    predictions = (probabilities >= 0.5).astype(int)
    lengths = np.asarray([len(preprocess_text(text)) for text in texts])
    negation = np.asarray([any(token in NEGATION_WORDS for token in preprocess_text(text)) for text in texts])
    masks = {
        "short_0_50_tokens": lengths <= 50,
        "medium_51_150_tokens": (lengths > 50) & (lengths <= 150),
        "long_over_150_tokens": lengths > 150,
        "contains_negation": negation,
        "no_negation": ~negation,
    }
    output: Dict[str, Dict[str, float]] = {}
    for name, mask in masks.items():
        if int(mask.sum()) == 0:
            continue
        output[name] = {
            "count": int(mask.sum()),
            "macro_f1": float(f1_score(labels[mask], predictions[mask], labels=[0, 1], average="macro", zero_division=0)),
            "error_rate": float(np.mean(predictions[mask] != labels[mask])),
        }
    return output


def make_prediction_rows(labels: np.ndarray, probabilities: np.ndarray, texts: Sequence[str], model_name: str) -> List[Dict[str, object]]:
    predictions = (probabilities >= 0.5).astype(int)
    rows: List[Dict[str, object]] = []
    for index, (label, probability, prediction, text) in enumerate(zip(labels, probabilities, predictions, texts)):
        rows.append({
            "model": model_name,
            "index": index,
            "text": text,
            "true_label": int(label),
            "predicted_label": int(prediction),
            "probability_positive": float(probability),
            "confidence": float(max(probability, 1 - probability)),
            "correct": bool(prediction == label),
        })
    return rows


def choose_error_review_rows(rows: List[Dict[str, object]], strict: bool) -> List[Dict[str, object]]:
    """Select exactly five rows for each required manual-review category."""
    unused = set(range(len(rows)))
    selected: List[Dict[str, object]] = []

    def add_group(group: str, candidate_indices: Iterable[int], count: int = 5) -> None:
        chosen = 0
        for index in candidate_indices:
            index = int(index)
            if index not in unused:
                continue
            row = dict(rows[index])
            row.update({"review_group": group, "manual_observation": "", "testable_fix": ""})
            selected.append(row)
            unused.remove(index)
            chosen += 1
            if chosen == count:
                return
        if strict and chosen < count:
            raise RuntimeError(f"Could not find five rows for {group}; use the full test set.")

    probabilities = np.asarray([float(row["probability_positive"]) for row in rows])
    labels = np.asarray([int(row["true_label"]) for row in rows])
    predictions = np.asarray([int(row["predicted_label"]) for row in rows])
    errors = predictions != labels
    confident_fp = np.where((labels == 0) & (predictions == 1) & (probabilities >= 0.8))[0]
    confident_fp = confident_fp[np.argsort(-probabilities[confident_fp])]
    confident_fn = np.where((labels == 1) & (predictions == 0) & (probabilities <= 0.2))[0]
    confident_fn = confident_fn[np.argsort(probabilities[confident_fn])]
    near = np.where(errors & (np.abs(probabilities - 0.5) <= 0.1))[0]
    near = near[np.argsort(np.abs(probabilities[near] - 0.5))]
    token_lengths = np.asarray([len(preprocess_text(str(row["text"]))) for row in rows])
    slice_failures = np.where(errors & ((token_lengths > 150) | np.asarray([bool(re.search(r"\b(not|no|never|n't)\b", str(row["text"]).lower())) for row in rows])))[0]
    fallback_errors = np.where(errors)[0]
    fallback_errors = fallback_errors[np.argsort(np.abs(probabilities[fallback_errors] - 0.5))]
    add_group("confident_false_positive", list(confident_fp) + list(fallback_errors))
    add_group("confident_false_negative", list(confident_fn) + list(fallback_errors))
    add_group("near_threshold_error", list(near) + list(fallback_errors))
    add_group("slice_specific_failure", list(slice_failures) + list(fallback_errors))
    return selected


def save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    flattened: Dict[str, object] = {}
    for key, value in metrics.items():
        if key == "confusion_matrix_tn_fp_fn_tp":
            flattened["true_negative"] = value[0]
            flattened["false_positive"] = value[1]
            flattened["false_negative"] = value[2]
            flattened["true_positive"] = value[3]
        elif key == "slice_metrics":
            for slice_name, values in value.items():
                flattened[f"{slice_name}_count"] = values["count"]
                flattened[f"{slice_name}_macro_f1"] = values["macro_f1"]
                flattened[f"{slice_name}_error_rate"] = values["error_rate"]
        elif key != "mcnemar_vs_baseline":
            flattened[key] = value
    mcnemar = metrics.get("mcnemar_vs_baseline")
    if isinstance(mcnemar, dict):
        for key, value in mcnemar.items():
            flattened[f"mcnemar_{key}"] = value
    return flattened


def plot_training_curves(histories: Dict[str, Dict[str, List[float]]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, history in histories.items():
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", label=f"{name} train")
        axes[0].plot(history["epoch"], history["validation_loss"], marker="x", linestyle="--", label=f"{name} validation")
        axes[1].plot(history["epoch"], history["train_accuracy"], marker="o", label=f"{name} train")
        axes[1].plot(history["epoch"], history["validation_accuracy"], marker="x", linestyle="--", label=f"{name} validation")
    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[1].set_title("Training and validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(figure)


def plot_confusion_matrices(predictions: Dict[str, Dict[str, List[int]]]) -> None:
    figure, axes = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 4))
    axes = np.atleast_1d(axes)
    for axis, (name, values) in zip(axes, predictions.items()):
        matrix = confusion_matrix(values["labels"], values["predictions"], labels=[0, 1])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name)
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_xticks([0, 1], ["negative", "positive"], rotation=20)
        axis.set_yticks([0, 1], ["negative", "positive"])
        for row in range(2):
            for column in range(2):
                axis.text(column, row, int(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(figure)


def package_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for module_name in ("torch", "numpy", "pandas", "sklearn", "scipy", "matplotlib"):
        try:
            module = __import__(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[module_name] = "unavailable"
    return versions


def write_results_markdown(
    config: Dict[str, object],
    eda: Dict[str, object],
    metrics_list: List[Dict[str, object]],
) -> None:
    lines = [
        "# Task 2 Results — Nikhil Kanaparthi",
        "",
        "## Dataset and preprocessing",
        "",
        "Yelp Polarity was loaded from the public Yelp Polarity CSV release corresponding to the Hugging Face `fancyzhx/yelp_polarity` dataset. Text was lowercased, punctuation and special characters were removed, text was tokenized, common stopwords were removed, and negation words were retained. No pretrained embedding or language model was used. No stemming or lemmatization was applied because preserving word forms supports the error analysis.",
        "",
        f"The run used {config['train_examples']} training examples, {config['validation_examples']} validation examples, and {config['test_examples']} test examples. The vocabulary contained {config['vocabulary_size']} tokens and the sequence limit was {config['max_length']} tokens. Device: `{config['device']}`.",
        "",
        f"Training class counts: {eda['train_class_counts']}. Review-length statistics after preprocessing: {eda['train_length_statistics']}.",
        "",
        "## Model comparison",
        "",
        "| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Parameters | Time (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in metrics_list:
        lines.append(
            f"| {result['model']} | {result['accuracy']:.4f} | {result['f1_macro']:.4f} | {result['roc_auc']:.4f} | {result['pr_auc']:.4f} | {result['mcc']:.4f} | {result['brier_score']:.4f} | {result['expected_calibration_error']:.4f} | {result['parameter_count']:,} | {result['training_time_seconds']:.1f} |"
        )
    lines.extend([
        "",
        "The baseline uses a mean of learned token embeddings. The CNN tests local n-gram features with kernel widths 3, 5, and 7. The bidirectional GRU tests order-sensitive sequential features. The same preprocessing, split, test set, threshold, and bootstrap procedure are used for all three models.",
        "",
        "## Statistical comparisons",
        "",
        "McNemar tests compare each experimental model's paired predictions with the baseline. The exact p-values and discordant-pair counts are in `metrics_report.json` and `metrics_report.csv`.",
        "",
        "## Outputs",
        "",
        "- `outputs/eda_length_distribution.png` and `outputs/class_distribution.png`: preprocessing analysis.",
        "- `outputs/training_curves.png`: training/validation loss and accuracy.",
        "- `outputs/confusion_matrices.png`: confusion matrices for all models.",
        "- `metrics_report.csv`: flattened rubric metrics for every model.",
        "- `error_review.csv`: 20 candidates for manual review; observations and testable fixes must be completed after reading each review.",
        "- `raw_logs/`: unedited console logs from each run.",
        "",
        "## Limitations and future work",
        "",
        "The models use a fixed vocabulary and truncate long reviews at the sequence limit. The preprocessing removes punctuation and many function words, which may discard some stylistic sentiment cues. Future work could compare learned subword features, calibrated thresholds, attention-free temporal pooling, and additional linguistic slices."
    ])
    (ROOT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete DATA266 Task 2 experiment")
    parser.add_argument("--smoke-test", action="store_true", help="Use a small subset and one epoch for validation")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    torch.set_float32_matmul_precision("high")
    train_path, test_path = download_yelp_csv()
    train_limit = 1200 if args.smoke_test else args.max_train
    test_limit = 400 if args.smoke_test else args.max_test
    raw_train, train_cleaning = read_reviews(train_path, train_limit)
    raw_test, test_cleaning = read_reviews(test_path, test_limit)
    train_df, validation_df = stratified_split(raw_train, args.validation_fraction, args.seed)
    vocabulary = build_vocabulary(train_df, args.max_vocab, args.min_frequency)

    train_encoded = encode_texts(train_df, vocabulary, args.max_length)
    validation_encoded = encode_texts(validation_df, vocabulary, args.max_length)
    test_encoded = encode_texts(raw_test, vocabulary, args.max_length)
    train_dataset = EncodedReviews(train_encoded, train_df["label"].to_numpy())
    validation_dataset = EncodedReviews(validation_encoded, validation_df["label"].to_numpy())
    test_dataset = EncodedReviews(test_encoded, raw_test["label"].to_numpy())
    batch_size = 32 if args.smoke_test else args.batch_size
    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=loader_generator)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    models = {
        "baseline_mean": (MeanEmbeddingClassifier(len(vocabulary), 128, 0.20), 1e-3, 1e-5),
        "experimental_cnn": (MultiKernelCNNClassifier(len(vocabulary), 128, 128, 0.25), 7e-4, 1e-5),
        "experimental_bigru": (BiGRUClassifier(len(vocabulary), 128, 128, 0.30), 8e-4, 1e-5),
    }
    epochs = 1 if args.smoke_test else args.epochs
    config: Dict[str, object] = {
        "seed": args.seed,
        "device": str(device),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "vocabulary_size": len(vocabulary),
        "max_length": args.max_length,
        "min_frequency": args.min_frequency,
        "batch_size": batch_size,
        "epochs": epochs,
        "validation_fraction": args.validation_fraction,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "test_examples": len(test_dataset),
        "train_cleaning": train_cleaning,
        "test_cleaning": test_cleaning,
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (ROOT / "vocabulary.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")

    train_length_values = train_df["token_count"].to_numpy()
    eda = {
        "train_class_counts": {str(int(key)): int(value) for key, value in train_df["label"].value_counts().sort_index().items()},
        "test_class_counts": {str(int(key)): int(value) for key, value in raw_test["label"].value_counts().sort_index().items()},
        "train_length_statistics": {
            "min": int(train_length_values.min()),
            "median": float(np.median(train_length_values)),
            "mean": float(train_length_values.mean()),
            "p90": float(np.percentile(train_length_values, 90)),
            "p99": float(np.percentile(train_length_values, 99)),
            "max": int(train_length_values.max()),
        },
    }
    (METRICS_DIR / "eda_summary.json").write_text(json.dumps(eda, indent=2), encoding="utf-8")
    pd.DataFrame({"length_tokens": train_length_values}).to_csv(OUTPUT_DIR / "review_lengths.csv", index=False)
    figure = plt.figure(figsize=(8, 4))
    plt.hist(train_length_values, bins=50, color="#4472c4", alpha=0.85)
    plt.title("Preprocessed Yelp review lengths")
    plt.xlabel("Token count")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "eda_length_distribution.png", dpi=160)
    plt.close(figure)
    figure = plt.figure(figsize=(5, 4))
    counts = train_df["label"].value_counts().sort_index()
    plt.bar(["negative", "positive"], [counts.get(0, 0), counts.get(1, 0)], color=["#d95f02", "#1b9e77"])
    plt.title("Training class distribution")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_distribution.png", dpi=160)
    plt.close(figure)

    all_metrics: List[Dict[str, object]] = []
    histories: Dict[str, Dict[str, List[float]]] = {}
    all_predictions_for_plot: Dict[str, Dict[str, List[int]]] = {}
    all_prediction_rows: Dict[str, List[Dict[str, object]]] = {}
    for name, (model, learning_rate, weight_decay) in models.items():
        print(f"\n=== {name} ===")
        history, seconds, memory, gradient_mean, gradient_max, nan_count = train_model(
            name,
            model,
            train_loader,
            validation_loader,
            device,
            epochs,
            learning_rate,
            weight_decay,
        )
        labels, probabilities, _ = predict(model, test_loader, device)
        model_slice_metrics = slice_metrics(labels, probabilities, raw_test["text"].tolist())
        result = compute_metrics(
            labels,
            probabilities,
            name,
            seconds,
            len(train_dataset) * epochs,
            device,
            parameter_count(model),
            memory,
            gradient_mean,
            gradient_max,
            nan_count,
            args.bootstrap_samples,
            args.seed,
            model_slice_metrics,
        )
        all_metrics.append(result)
        histories[name] = history
        predictions = (probabilities >= 0.5).astype(int)
        all_predictions_for_plot[name] = {"labels": labels.tolist(), "predictions": predictions.tolist()}
        all_prediction_rows[name] = make_prediction_rows(labels, probabilities, raw_test["text"].tolist(), name)
        checkpoint = {
            "model_name": name,
            "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "vocabulary": vocabulary,
            "config": config,
            "history": history,
        }
        torch.save(checkpoint, CHECKPOINT_DIR / f"{name}.pt")
        (OUTPUT_DIR / f"{name}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    baseline_predictions = np.asarray(all_predictions_for_plot["baseline_mean"]["predictions"])
    for result in all_metrics:
        if result["model"] == "baseline_mean":
            continue
        experimental_predictions = np.asarray(all_predictions_for_plot[str(result["model"])] ["predictions"])
        labels = np.asarray(all_predictions_for_plot[str(result["model"])] ["labels"])
        correct_baseline_wrong_experimental = int(np.sum((baseline_predictions == labels) & (experimental_predictions != labels)))
        wrong_baseline_correct_experimental = int(np.sum((baseline_predictions != labels) & (experimental_predictions == labels)))
        discordant = correct_baseline_wrong_experimental + wrong_baseline_correct_experimental
        p_value = 1.0 if discordant == 0 else float(2 * min(binomtest(correct_baseline_wrong_experimental, discordant, 0.5).pvalue, 0.5))
        result["mcnemar_vs_baseline"] = {
            "baseline_correct_experimental_wrong": correct_baseline_wrong_experimental,
            "baseline_wrong_experimental_correct": wrong_baseline_correct_experimental,
            "exact_two_sided_p_value": min(p_value, 1.0),
        }

    # The GRU is the designated primary model for the manual review. This is
    # documented in results.md and can be changed before a rerun if needed.
    review_rows = choose_error_review_rows(all_prediction_rows["experimental_bigru"], strict=not args.smoke_test)
    save_csv(ROOT / "error_review.csv", review_rows)
    (METRICS_DIR / "all_predictions.json").write_text(json.dumps(all_prediction_rows, indent=2), encoding="utf-8")
    (METRICS_DIR / "error_review_candidates.json").write_text(json.dumps(review_rows, indent=2), encoding="utf-8")
    metrics_json = json.dumps(all_metrics, indent=2)
    (METRICS_DIR / "metrics_report.json").write_text(metrics_json, encoding="utf-8")
    (ROOT / "metrics_report.json").write_text(metrics_json, encoding="utf-8")
    save_csv(ROOT / "metrics_report.csv", [flatten_metrics(result) for result in all_metrics])
    (ROOT / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    plot_training_curves(histories)
    plot_confusion_matrices(all_predictions_for_plot)
    write_results_markdown(config, eda, all_metrics)

    manifest = {
        "seed": args.seed,
        "device": str(device),
        "hardware": {
            "operating_system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "model_checkpoints": [f"checkpoints/{name}.pt" for name in models],
        "metrics_source": "metrics_report.csv and metrics_report.json",
        "raw_log_instruction": "Run with 2>&1 | tee raw_logs/task2_full_run.log; do not edit the resulting log.",
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nTask 2 complete.")
    print(f"Device: {device}")
    print(f"Metrics: {ROOT / 'metrics_report.csv'}")
    print(f"Manual review worksheet: {ROOT / 'error_review.csv'}")


if __name__ == "__main__":
    main()
