"""
train_v3.py

CycleGAN baseline for the Monet / Photo task.

Architecture: the v1 models in models.py, unchanged.
Training recipe: N(0, 0.02) weight init, fake image buffer, constant learning
rate then linear decay to zero, local FID every few epochs, and a checkpoint
saved whenever FID improves (instead of choosing by cycle + identity loss).

Usage:
    python train_v3.py --config config_v3.json
    python train_v3.py --config config_v3.json --resume
    python train_v3.py --config config_v3.json --eval_checkpoint path/to/old.pth

Install once:
    pip install torchmetrics torch-fidelity

Notes:
    A = photos, B = Monet paintings. G_AB turns photos into Monet style images.
    The local FID is photo -> Monet only. It uses its own Inception weights, so
    it will not match the Kaggle number. Use it to compare checkpoints and runs,
    always with the same config values (eval_num_photos, image_size).
"""

import argparse
import csv
import itertools
import json
import logging
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image

from data_preprocessing import create_dataloader
from models import Generator, PatchGANDiscriminator

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

METRIC_FIELDS = [
    "epoch",
    "lr_G",
    "lr_D",
    "loss_G",
    "loss_D_A",
    "loss_D_B",
    "loss_cycle",
    "loss_identity",
    "gradnorm_G_mean",
    "gradnorm_G_max",
    "gradnorm_D_A_mean",
    "gradnorm_D_A_max",
    "gradnorm_D_B_mean",
    "gradnorm_D_B_max",
    "nan_count",
    "local_fid_photo_to_monet",
    "epoch_time_sec",
    "pairs_per_sec",
    "peak_memory_gb",
]


# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(module):
    """Normal(0, 0.02) init for conv layers, as in the CycleGAN paper."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_requires_grad(models, requires_grad):
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad


def safe_save(obj, path):
    """Write to a temporary file first so a crash cannot leave a broken file."""
    temporary = str(path) + ".tmp"
    torch.save(obj, temporary)
    os.replace(temporary, path)


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_uint8_images(paths, size):
    """Load images as a uint8 tensor of shape N x 3 x size x size."""
    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BICUBIC)
        array = np.asarray(image).copy()
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(tensors)


# ----------------------------------------------------------------------
# Fake image buffer (the paper keeps the last 50 generated images)
# ----------------------------------------------------------------------

class ImagePool:

    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images

        returned = []

        for image in images:
            image = image.detach().unsqueeze(0)

            if len(self.images) < self.pool_size:
                self.images.append(image.clone())
                returned.append(image)

            elif random.random() > 0.5:
                index = random.randint(0, self.pool_size - 1)
                returned.append(self.images[index].clone())
                self.images[index] = image.clone()

            else:
                returned.append(image)

        return torch.cat(returned, dim=0)


# ----------------------------------------------------------------------
# Local FID (photo -> Monet)
# ----------------------------------------------------------------------

class LocalFID:

    def __init__(self, real_monet_uint8, photos_uint8, device, batch_size):
        self.device = device
        self.batch_size = batch_size
        self.photos = photos_uint8

        self.metric = FrechetInceptionDistance(
            feature=2048,
            normalize=True,
            reset_real_features=False,
        ).to(device)

        # Real Monet statistics are computed once and kept.
        for start in range(0, len(real_monet_uint8), batch_size):
            batch = real_monet_uint8[start:start + batch_size]
            batch = batch.to(device).float() / 255.0
            self.metric.update(batch, real=True)

    @torch.no_grad()
    def score(self, G_AB):
        was_training = G_AB.training
        G_AB.eval()

        for start in range(0, len(self.photos), self.batch_size):
            batch = self.photos[start:start + self.batch_size]
            batch = batch.to(self.device).float() / 255.0 * 2.0 - 1.0
            fake = G_AB(batch)
            fake = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
            self.metric.update(fake, real=False)

        value = float(self.metric.compute())
        self.metric.reset()  # keeps the real statistics

        G_AB.train(was_training)
        return value


@torch.no_grad()
def save_samples(G_AB, G_BA, fixed_photos, path):
    """Rows: input photos, photo -> Monet, Monet -> photo (reconstruction)."""
    G_AB.eval()
    G_BA.eval()

    fake_monet = G_AB(fixed_photos)
    reconstruction = G_BA(fake_monet)
    grid = torch.cat([fixed_photos, fake_monet, reconstruction], dim=0)

    save_image(
        grid,
        path,
        nrow=fixed_photos.size(0),
        normalize=True,
        value_range=(-1, 1),
    )

    G_AB.train()
    G_BA.train()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--eval_checkpoint", default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as file:
        cfg = json.load(file)

    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Raw log: appended, never rewritten.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "train.log", mode="a"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    log = logging.info

    with open(run_dir / "config_used.json", "w") as file:
        json.dump(cfg, file, indent=2)

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "torchmetrics": torchmetrics.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"
            ),
        }
        with open(manifest_path, "w") as file:
            json.dump(manifest, file, indent=2)

    log("=" * 70)
    log(f"Run: {cfg['run_name']} | device: {device}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Config: {json.dumps(cfg)}")

    root = Path(cfg["dataset_root"])
    size = cfg["image_size"]

    # ------------------------------------------------------------------
    # Local FID data: all real Monet images, and a fixed set of photos
    # ------------------------------------------------------------------
    photo_paths = list_images(root / "validation" / "photos")
    photo_paths = photo_paths[:cfg["eval_num_photos"]]

    monet_paths = []
    for split in ("train", "validation", "test"):
        monet_paths += list_images(root / split / "monet")

    fid_eval = None
    fixed_photos = None

    if len(photo_paths) > 0 and len(monet_paths) > 0:
        photos_uint8 = load_uint8_images(photo_paths, size)
        monet_uint8 = load_uint8_images(monet_paths, size)
        log(
            f"Local FID data: {len(photos_uint8)} photos, "
            f"{len(monet_uint8)} real Monet images"
        )
        fid_eval = LocalFID(
            monet_uint8, photos_uint8, device, cfg["eval_batch_size"]
        )
        count = min(8, len(photos_uint8))
        fixed_photos = (
            photos_uint8[:count].to(device).float() / 255.0 * 2.0 - 1.0
        )
    else:
        log("WARNING: no validation photos or Monet images found. "
            "Local FID is disabled.")

    # ------------------------------------------------------------------
    # Evaluate an existing checkpoint with the same local FID, then exit
    # ------------------------------------------------------------------
    if args.eval_checkpoint:
        if fid_eval is None:
            raise RuntimeError("Local FID is disabled, check dataset_root.")

        G_test = Generator(3, 3, 9).to(device)
        checkpoint = torch.load(args.eval_checkpoint, map_location=device)
        G_test.load_state_dict(checkpoint["G_AB"])
        value = fid_eval.score(G_test)
        log(f"Local FID (photo to Monet) for {args.eval_checkpoint}: "
            f"{value:.4f}")
        return

    # ------------------------------------------------------------------
    # Training data
    # ------------------------------------------------------------------
    train_loader = create_dataloader(
        domain_a_dir=root / "train" / "photos",
        domain_b_dir=root / "train" / "monet",
        batch_size=cfg["batch_size"],
        train=True,
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    first_batch = next(iter(train_loader))
    low = first_batch["A"].min().item()
    high = first_batch["A"].max().item()
    log(f"Training batches per epoch: {len(train_loader)}")
    log(f"Photo value range in first batch: {low:.3f} to {high:.3f}")
    if low >= -0.01:
        raise ValueError(
            "The loader does not seem to output values in [-1, 1]. "
            "Local FID and the Tanh output assume [-1, 1]."
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    G_AB = Generator(3, 3, 9).to(device)
    G_BA = Generator(3, 3, 9).to(device)
    D_A = PatchGANDiscriminator(3).to(device)
    D_B = PatchGANDiscriminator(3).to(device)

    for model in (G_AB, G_BA, D_A, D_B):
        model.apply(init_weights)

    log(
        f"Parameters | G_AB: {count_parameters(G_AB):,} | "
        f"G_BA: {count_parameters(G_BA):,} | "
        f"D_A: {count_parameters(D_A):,} | D_B: {count_parameters(D_B):,}"
    )

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    lambda_cycle = cfg["lambda_cycle"]
    lambda_identity = cfg["lambda_identity"]

    G_params = list(G_AB.parameters()) + list(G_BA.parameters())

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=cfg["lr_G"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(),
        lr=cfg["lr_D"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(),
        lr=cfg["lr_D"],
        betas=(cfg["beta1"], cfg["beta2"]),
    )

    epochs = cfg["epochs"]
    decay_start = cfg["decay_start_epoch"]

    def lr_lambda(epoch):
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start))

    scheduler_G = LambdaLR(optimizer_G, lr_lambda)
    scheduler_D_A = LambdaLR(optimizer_D_A, lr_lambda)
    scheduler_D_B = LambdaLR(optimizer_D_B, lr_lambda)

    pool_A = ImagePool(cfg["pool_size"])
    pool_B = ImagePool(cfg["pool_size"])

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    start_epoch = 0
    best_fid = float("inf")
    nan_count = 0
    latest_path = ckpt_dir / "latest.pth"

    if args.resume and latest_path.exists():
        state = torch.load(latest_path, map_location=device)
        G_AB.load_state_dict(state["G_AB"])
        G_BA.load_state_dict(state["G_BA"])
        D_A.load_state_dict(state["D_A"])
        D_B.load_state_dict(state["D_B"])
        optimizer_G.load_state_dict(state["optimizer_G"])
        optimizer_D_A.load_state_dict(state["optimizer_D_A"])
        optimizer_D_B.load_state_dict(state["optimizer_D_B"])
        scheduler_G.load_state_dict(state["scheduler_G"])
        scheduler_D_A.load_state_dict(state["scheduler_D_A"])
        scheduler_D_B.load_state_dict(state["scheduler_D_B"])
        start_epoch = state["epoch"]
        best_fid = state["best_fid"]
        log(f"Resumed from epoch {start_epoch}")

    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as file:
            csv.DictWriter(file, fieldnames=METRIC_FIELDS).writeheader()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    training_start = time.time()
    log("Training started.")

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        epoch_start = time.time()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        G_AB.train()
        G_BA.train()
        D_A.train()
        D_B.train()

        lr_G_used = optimizer_G.param_groups[0]["lr"]
        lr_D_used = optimizer_D_A.param_groups[0]["lr"]

        loss_sums = torch.zeros(5, device=device)
        norm_sums = torch.zeros(3, device=device)
        norm_max = torch.zeros(3, device=device)
        steps = 0
        pairs_seen = 0

        for step, batch in enumerate(train_loader, start=1):

            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

            # ---------------- Generators ----------------
            set_requires_grad([D_A, D_B], False)
            optimizer_G.zero_grad(set_to_none=True)

            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            reconstructed_A = G_BA(fake_B)
            reconstructed_B = G_AB(fake_A)
            identity_A = G_BA(real_A)
            identity_B = G_AB(real_B)

            prediction_fake_B = D_B(fake_B)
            prediction_fake_A = D_A(fake_A)

            loss_gan = (
                criterion_gan(
                    prediction_fake_B, torch.ones_like(prediction_fake_B)
                )
                + criterion_gan(
                    prediction_fake_A, torch.ones_like(prediction_fake_A)
                )
            )
            loss_cycle = (
                criterion_cycle(reconstructed_A, real_A)
                + criterion_cycle(reconstructed_B, real_B)
            )
            loss_identity = (
                criterion_identity(identity_A, real_A)
                + criterion_identity(identity_B, real_B)
            )

            loss_G = (
                loss_gan
                + lambda_cycle * loss_cycle
                + lambda_identity * loss_identity
            )

            loss_G.backward()
            # max_norm=inf measures the gradient norm without clipping.
            norm_G = torch.nn.utils.clip_grad_norm_(G_params, float("inf"))
            optimizer_G.step()

            # ---------------- Discriminator A (photos) ----------------
            set_requires_grad([D_A, D_B], True)

            fake_A_for_D = pool_A.query(fake_A.detach())

            optimizer_D_A.zero_grad(set_to_none=True)
            prediction_real_A = D_A(real_A)
            prediction_fake_A_pool = D_A(fake_A_for_D)

            loss_D_A = 0.5 * (
                criterion_gan(
                    prediction_real_A, torch.ones_like(prediction_real_A)
                )
                + criterion_gan(
                    prediction_fake_A_pool,
                    torch.zeros_like(prediction_fake_A_pool),
                )
            )
            loss_D_A.backward()
            norm_D_A = torch.nn.utils.clip_grad_norm_(
                D_A.parameters(), float("inf")
            )
            optimizer_D_A.step()

            # ---------------- Discriminator B (Monet) ----------------
            fake_B_for_D = pool_B.query(fake_B.detach())

            optimizer_D_B.zero_grad(set_to_none=True)
            prediction_real_B = D_B(real_B)
            prediction_fake_B_pool = D_B(fake_B_for_D)

            loss_D_B = 0.5 * (
                criterion_gan(
                    prediction_real_B, torch.ones_like(prediction_real_B)
                )
                + criterion_gan(
                    prediction_fake_B_pool,
                    torch.zeros_like(prediction_fake_B_pool),
                )
            )
            loss_D_B.backward()
            norm_D_B = torch.nn.utils.clip_grad_norm_(
                D_B.parameters(), float("inf")
            )
            optimizer_D_B.step()

            # ---------------- Bookkeeping ----------------
            losses = torch.stack([
                loss_G.detach(),
                loss_D_A.detach(),
                loss_D_B.detach(),
                loss_cycle.detach(),
                loss_identity.detach(),
            ])
            norms = torch.stack(
                [norm_G.detach(), norm_D_A.detach(), norm_D_B.detach()]
            )

            loss_sums += losses
            norm_sums += norms
            norm_max = torch.maximum(norm_max, norms)
            steps += 1
            pairs_seen += real_A.size(0)

            if step == 1 or step % cfg["log_every"] == 0:
                values = losses.tolist()
                log(
                    f"Epoch [{epoch_number}/{epochs}] "
                    f"Step [{step}/{len(train_loader)}] | "
                    f"G: {values[0]:.4f} | D_A: {values[1]:.4f} | "
                    f"D_B: {values[2]:.4f} | Cycle: {values[3]:.4f} | "
                    f"Identity: {values[4]:.4f}"
                )
                if not all(math.isfinite(v) for v in values):
                    nan_count += 1
                    raise RuntimeError(
                        f"NaN or infinity at epoch {epoch_number}, "
                        f"step {step}."
                    )

        # ---------------- End of epoch ----------------
        means = (loss_sums / steps).tolist()
        norm_means = (norm_sums / steps).tolist()
        norm_maxes = norm_max.tolist()

        if not all(math.isfinite(v) for v in means):
            nan_count += 1
            raise RuntimeError(
                f"NaN or infinity in epoch {epoch_number} averages."
            )

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        epoch_time = time.time() - epoch_start
        pairs_per_sec = pairs_seen / epoch_time
        peak_memory = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if torch.cuda.is_available() else 0.0
        )

        log(
            f"Epoch [{epoch_number}/{epochs}] done in "
            f"{epoch_time / 60:.2f} min | "
            f"G: {means[0]:.4f} | D_A: {means[1]:.4f} | "
            f"D_B: {means[2]:.4f} | Cycle: {means[3]:.4f} | "
            f"Identity: {means[4]:.4f} | lr_G: {lr_G_used:.7f}"
        )

        # ---------------- Local FID ----------------
        fid_value = ""
        evaluate_now = (
            fid_eval is not None
            and cfg["eval_every"] > 0
            and (
                epoch_number == 1
                or epoch_number % cfg["eval_every"] == 0
                or epoch_number == epochs
            )
        )

        if evaluate_now:
            fid_value = fid_eval.score(G_AB)
            log(f"Epoch [{epoch_number}/{epochs}] local FID: {fid_value:.4f}")

            if fid_value < best_fid:
                best_fid = fid_value
                safe_save(
                    {
                        "epoch": epoch_number,
                        "fid": fid_value,
                        "G_AB": G_AB.state_dict(),
                        "G_BA": G_BA.state_dict(),
                    },
                    ckpt_dir / "best_fid.pth",
                )
                log(f"New best FID {best_fid:.4f}, saved best_fid.pth")

        # ---------------- Samples and checkpoints ----------------
        if fixed_photos is not None and (
            epoch_number == 1 or epoch_number % cfg["sample_every"] == 0
        ):
            save_samples(
                G_AB,
                G_BA,
                fixed_photos,
                sample_dir / f"epoch_{epoch_number:03d}.png",
            )

        if epoch_number % cfg["save_every"] == 0:
            safe_save(
                {
                    "epoch": epoch_number,
                    "G_AB": G_AB.state_dict(),
                    "G_BA": G_BA.state_dict(),
                },
                ckpt_dir / f"generators_epoch_{epoch_number:03d}.pth",
            )

        safe_save(
            {
                "epoch": epoch_number,
                "best_fid": best_fid,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D_A": optimizer_D_A.state_dict(),
                "optimizer_D_B": optimizer_D_B.state_dict(),
                "scheduler_G": scheduler_G.state_dict(),
                "scheduler_D_A": scheduler_D_A.state_dict(),
                "scheduler_D_B": scheduler_D_B.state_dict(),
            },
            latest_path,
        )

        row = {
            "epoch": epoch_number,
            "lr_G": lr_G_used,
            "lr_D": lr_D_used,
            "loss_G": means[0],
            "loss_D_A": means[1],
            "loss_D_B": means[2],
            "loss_cycle": means[3],
            "loss_identity": means[4],
            "gradnorm_G_mean": norm_means[0],
            "gradnorm_G_max": norm_maxes[0],
            "gradnorm_D_A_mean": norm_means[1],
            "gradnorm_D_A_max": norm_maxes[1],
            "gradnorm_D_B_mean": norm_means[2],
            "gradnorm_D_B_max": norm_maxes[2],
            "nan_count": nan_count,
            "local_fid_photo_to_monet": fid_value,
            "epoch_time_sec": epoch_time,
            "pairs_per_sec": pairs_per_sec,
            "peak_memory_gb": peak_memory,
        }
        with open(metrics_path, "a", newline="") as file:
            csv.DictWriter(file, fieldnames=METRIC_FIELDS).writerow(row)

    total_minutes = (time.time() - training_start) / 60
    log(f"Training finished in {total_minutes:.2f} minutes.")
    log(f"Best local FID: {best_fid:.4f}")
    log(f"Best checkpoint: {ckpt_dir / 'best_fid.pth'}")


if __name__ == "__main__":
    main()