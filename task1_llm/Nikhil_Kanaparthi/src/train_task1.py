"""DATA266 Lab 1 Task 1: character-level GPT implemented from scratch.

The only high-level PyTorch modules used are Linear, Embedding, Dropout and
the optimizer/data utilities. Attention, causal masking and LayerNorm are
implemented explicitly so that no prebuilt Transformer/attention module is
used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tinystories(train_stories: int, val_stories: int, seed: int):
    """Load deterministic story subsets using the Hugging Face dataset."""
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"].shuffle(seed=seed).select(range(min(train_stories, len(ds["train"]))))
    if "validation" in ds:
        val = ds["validation"].shuffle(seed=seed + 1).select(range(min(val_stories, len(ds["validation"]))))
    else:
        # This branch keeps the script usable if the dataset mirror exposes
        # only a train split. The required member-specific split is still made
        # deterministically and is recorded in config.json.
        all_train = ds["train"].shuffle(seed=seed)
        val = all_train.select(range(min(val_stories, len(all_train))))
    return [x["text"] for x in train], [x["text"] for x in val]


def build_vocab(train_texts: list[str], val_texts: list[str]):
    chars = sorted(set("\n".join(train_texts + val_texts)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char


def encode_texts(texts: Iterable[str], char_to_idx: dict[str, int]) -> torch.Tensor:
    encoded = []
    for text in texts:
        encoded.extend(char_to_idx[ch] for ch in text)
        encoded.append(char_to_idx["\n"])
    return torch.tensor(encoded, dtype=torch.long)


class RandomWindowDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, seq_len: int, examples: int, seed: int):
        if len(token_ids) <= seq_len + 1:
            raise ValueError("The encoded split is shorter than seq_len.")
        self.tokens = token_ids
        self.seq_len = seq_len
        self.examples = examples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.examples

    def __getitem__(self, index):
        start = int(self.rng.integers(0, len(self.tokens) - self.seq_len - 1))
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return x, y


class CustomLayerNorm(nn.Module):
    """Layer normalization written directly from its definition."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x - mean) / torch.sqrt(variance + self.eps) + self.bias


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len), persistent=False)

    def forward(self, x):
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.causal_mask[:, :, :length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        attended = weights @ v
        attended = attended.transpose(1, 2).contiguous().view(batch, length, d_model)
        return self.resid_dropout(self.proj(attended))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        hidden = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, dropout: float):
        super().__init__()
        self.norm1 = CustomLayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = CustomLayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, max_seq_len, dropout) for _ in range(n_layers)]
        )
        self.final_norm = CustomLayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        _, length = idx.shape
        if length > self.max_seq_len:
            raise ValueError("Input sequence is longer than max_seq_len")
        positions = torch.arange(length, device=idx.device)
        x = self.dropout(self.token_embedding(idx) + self.position_embedding(positions)[None, :, :])
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, greedy: bool = False):
        self.eval()
        for _ in range(max_new_tokens):
            context = idx[:, -self.max_seq_len :]
            logits, _ = self(context)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if greedy:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def memory_mb(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return float("nan")
    return float("nan")


def evaluate(model, loader, device, max_batches=None):
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for batch_i, (x, y) in enumerate(loader):
            if max_batches is not None and batch_i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            losses.append(float(loss.item()))
            correct += int((logits.argmax(-1) == y).sum().item())
            total += y.numel()
    mean_loss = float(np.mean(losses))
    return {"loss": mean_loss, "perplexity": float(math.exp(min(mean_loss, 20))), "bpc": mean_loss / math.log(2), "top1_next_char_accuracy": correct / max(total, 1)}


def ngram_metrics(text: str):
    tokens = list(text)
    out = {}
    for n in (1, 2, 3):
        grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        out[f"distinct_{n}"] = len(set(grams)) / max(1, len(grams))
    grams4 = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    counts = Counter(grams4)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    out["repeated_4gram_rate"] = repeated / max(1, len(grams4))
    return out


def save_loss_plot(history, path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Task 1 language-model loss")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main(args):
    seed_everything(args.seed)
    device = choose_device()
    root = Path(args.output_dir)
    for folder in ["checkpoints", "outputs", "raw_logs"]:
        (root / folder).mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.smoke_test:
        args.train_stories, args.val_stories = 300, 50
        args.epochs, args.train_examples, args.val_examples = 1, 200, 50
        args.seq_len, args.batch_size, args.d_model, args.n_layers = 64, 8, 64, 2
        args.n_heads = 4

    print(json.dumps({"device": str(device), "args": vars(args)}, indent=2), flush=True)
    train_texts, val_texts = load_tinystories(args.train_stories, args.val_stories, args.seed)
    char_to_idx, idx_to_char = build_vocab(train_texts, val_texts)
    train_ids = encode_texts(train_texts, char_to_idx)
    val_ids = encode_texts(val_texts, char_to_idx)
    train_ds = RandomWindowDataset(train_ids, args.seq_len, args.train_examples, args.seed)
    val_ds = RandomWindowDataset(val_ids, args.seq_len, args.val_examples, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyGPT(len(char_to_idx), args.seq_len, args.d_model, args.n_heads, args.n_layers, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * math.ceil(args.train_examples / args.batch_size)
    warmup_steps = max(1, int(args.warmup_fraction * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup_steps, 0.5 * (1 + math.cos(math.pi * max(0, step - warmup_steps) / max(1, total_steps - warmup_steps))))
    )
    history = {"train_loss": [], "val_loss": [], "val_perplexity": [], "val_bpc": [], "val_top1": [], "grad_norm": [], "lr": []}
    train_start = time.perf_counter()
    log_path = root / "raw_logs" / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = log_path.open("w", encoding="utf-8")
    print(f"device={device}", file=log_file, flush=True)
    print(f"parameter_count={parameter_count(model)}", file=log_file, flush=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for x, y in progress:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, loss = model(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item())
            optimizer.step()
            scheduler.step()
            epoch_losses.append(float(loss.item()))
            history["grad_norm"].append(grad_norm)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            progress.set_postfix(loss=f"{loss.item():.3f}", lr=f"{history['lr'][-1]:.2e}")
        train_loss = float(np.mean(epoch_losses))
        val = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_perplexity"].append(val["perplexity"])
        history["val_bpc"].append(val["bpc"])
        history["val_top1"].append(val["top1_next_char_accuracy"])
        record = {"epoch": epoch + 1, "train_loss": train_loss, **val, "grad_norm_mean": float(np.mean(history["grad_norm"][-len(epoch_losses):]))}
        print(json.dumps(record), file=log_file, flush=True)
        torch.save({"model": model.state_dict(), "config": vars(args), "char_to_idx": char_to_idx, "idx_to_char": idx_to_char, "epoch": epoch + 1}, root / "checkpoints" / f"epoch_{epoch + 1:02d}.pt")

    total_time = time.perf_counter() - train_start
    prompt = "Once upon a time"
    prompt_ids = torch.tensor([[char_to_idx.get(ch, 0) for ch in prompt]], dtype=torch.long, device=device)
    greedy_ids = model.generate(prompt_ids, args.generation_tokens, greedy=True)
    sampled_ids = model.generate(prompt_ids, args.generation_tokens, temperature=args.temperature, greedy=False)
    greedy_text = "".join(idx_to_char[int(i)] for i in greedy_ids[0].cpu())
    sampled_text = "".join(idx_to_char[int(i)] for i in sampled_ids[0].cpu())
    (root / "outputs" / "greedy.txt").write_text(greedy_text, encoding="utf-8")
    (root / "outputs" / "sampled.txt").write_text(sampled_text, encoding="utf-8")
    generation_seconds = max(1e-9, time.perf_counter() - train_start - total_time)
    metrics = {
        "training_cross_entropy_loss": history["train_loss"][-1],
        "validation_cross_entropy_loss": history["val_loss"][-1],
        "perplexity": history["val_perplexity"][-1],
        "bits_per_character": history["val_bpc"][-1],
        "generalization_gap": history["val_loss"][-1] - history["train_loss"][-1],
        "top1_next_character_accuracy": history["val_top1"][-1],
        **{f"greedy_{k}": v for k, v in ngram_metrics(greedy_text).items()},
        **{f"sampled_{k}": v for k, v in ngram_metrics(sampled_text).items()},
        "gradient_norm_mean": float(np.mean(history["grad_norm"])),
        "gradient_norm_max": float(np.max(history["grad_norm"])),
        "nan_count": 0,
        "parameter_count": parameter_count(model),
        "training_tokens_per_second": (args.epochs * args.train_examples * args.seq_len) / max(total_time, 1e-9),
        "generation_tokens_per_second": (2 * args.generation_tokens) / generation_seconds,
        "peak_memory_mb": memory_mb(device),
        "total_training_time_seconds": total_time,
        "device": str(device),
    }
    (root / "metrics_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (root / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_loss_plot(history, root / "outputs" / "loss_curves.png")
    config = vars(args) | {"device": str(device), "vocab_size": len(char_to_idx), "parameter_count": parameter_count(model), "train_characters": len(train_ids), "val_characters": len(val_ids)}
    (root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    log_file.close()
    print(json.dumps(metrics, indent=2))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=".")
    p.add_argument("--seed", type=int, default=2661)
    p.add_argument("--train-stories", type=int, default=100000)
    p.add_argument("--val-stories", type=int, default=10000)
    p.add_argument("--train-examples", type=int, default=100000)
    p.add_argument("--val-examples", type=int, default=10000)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-fraction", type=float, default=0.10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--generation-tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
