"""Fine-tune CycleGAN and evaluate the notebook's FID/MiFID after every epoch.

This script preserves the original CycleGAN architecture but adds two changes
for fine-tuning:
  * separate learning rates for G, D_A, and D_B;
  * lambda_identity is read from the fine-tuning config.

The evaluation implementation below matches Part3_Evaluation_Script.ipynb:
Inception-v3 pool features, scipy Frechet distance, and mean paired cosine
distance (MiFID), evaluated in both directions and averaged.

Run from the project's src directory, for example:
  python train_finetune_with_fid_mifid.py \
      --config runs/v3_finetune/config_finetune.json --resume
"""

import argparse
import csv
import itertools
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cosine
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import save_image

from data_preprocessing import create_dataloader
from models import Generator, PatchGANDiscriminator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def set_requires_grad(models, requires_grad):
    for model in models:
        for parameter in model.parameters():
            parameter.requires_grad = requires_grad


def safe_save(obj, path):
    path = Path(path)
    temporary = str(path) + ".tmp"
    torch.save(obj, temporary)
    os.replace(temporary, path)


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


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


# ---------------------------------------------------------------------------
# Exact evaluation logic from Part3_Evaluation_Script.ipynb
# ---------------------------------------------------------------------------

def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm(
            (sigma1 + offset).dot(sigma2 + offset)
        )
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(
        diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    )


class NotebookEvaluator:
    """Reproduces the notebook's Inception/FID/MiFID calculation."""

    def __init__(self, real_monet_paths, real_photo_paths, device,
                 n_eval=300, batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.real_monet_paths = sorted(real_monet_paths)[:n_eval]
        self.real_photo_paths = sorted(real_photo_paths)[:n_eval]

        if not self.real_monet_paths or not self.real_photo_paths:
            raise RuntimeError(
                "Evaluation folders must contain both Monet and photo images."
            )

        self.transform = T.Compose([
            T.Resize(299),
            T.CenterCrop(299),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])

        inception = tv_models.inception_v3(
            weights=tv_models.Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        inception.fc = nn.Identity()
        self.model = inception.to(device).eval()

        # Cache real features once. The notebook recomputes them each run;
        # caching does not change the values and makes per-epoch evaluation
        # practical.
        self.real_monet_features = self.get_activations(
            self.real_monet_paths
        )
        self.real_photo_features = self.get_activations(
            self.real_photo_paths
        )

    def load_batch(self, paths):
        images = []
        for path in paths:
            image = Image.open(path).convert("RGB")
            images.append(self.transform(image))
        return torch.stack(images, dim=0)

    @torch.no_grad()
    def get_activations(self, paths):
        features = []
        for start in range(0, len(paths), self.batch_size):
            batch = self.load_batch(paths[start:start + self.batch_size])
            output = self.model(batch.to(self.device))
            features.append(output.detach().cpu().numpy())
        return np.concatenate(features, axis=0)

    def calculate_from_features(self, real_features, generated_features):
        n = min(len(real_features), len(generated_features))
        real_features = real_features[:n]
        generated_features = generated_features[:n]

        mu_r = real_features.mean(axis=0)
        sig_r = np.cov(real_features, rowvar=False)
        mu_g = generated_features.mean(axis=0)
        sig_g = np.cov(generated_features, rowvar=False)
        fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

        cos_dists = [
            cosine(real_features[i], generated_features[i])
            for i in range(n)
        ]
        mifid = float(np.mean(cos_dists))
        return fid, mifid

    @torch.no_grad()
    def generate_images(self, generator, input_paths, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        generator.eval()

        # The training loader uses 256x256 images normalized to [-1, 1].
        input_transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        for start in range(0, len(input_paths), 8):
            batch_paths = input_paths[start:start + 8]
            batch = torch.stack([
                input_transform(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)
            generated = ((generator(batch) + 1.0) / 2.0).clamp(0.0, 1.0)
            for offset, image in enumerate(generated):
                save_image(
                    image,
                    output_dir / f"image_{start + offset:04d}.png",
                )
        generator.train()

    @torch.no_grad()
    def evaluate(self, G_AB, G_BA, output_dir, epoch):
        epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
        gen_b2a_dir = epoch_dir / "pred_B2A"  # Photo -> Monet
        gen_a2b_dir = epoch_dir / "pred_A2B"  # Monet -> Photo

        self.generate_images(G_AB, self.real_photo_paths, gen_b2a_dir)
        self.generate_images(G_BA, self.real_monet_paths, gen_a2b_dir)

        gen_b2a = self.get_activations(sorted(list_images(gen_b2a_dir)))
        gen_a2b = self.get_activations(sorted(list_images(gen_a2b_dir)))

        fid_b2a, mifid_b2a = self.calculate_from_features(
            self.real_monet_features, gen_b2a
        )
        fid_a2b, mifid_a2b = self.calculate_from_features(
            self.real_photo_features, gen_a2b
        )

        return {
            "fid_photo_to_monet": fid_b2a,
            "mifid_photo_to_monet": mifid_b2a,
            "fid_monet_to_photo": fid_a2b,
            "mifid_monet_to_photo": mifid_a2b,
            "fid_average": (fid_a2b + fid_b2a) / 2.0,
            "mifid_average": (mifid_a2b + mifid_b2a) / 2.0,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def set_optimizer_lr(optimizer, learning_rate):
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as file:
        cfg = json.load(file)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    ckpt_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    evaluation_dir = run_dir / "evaluation_images"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "finetune_fid_mifid.log", mode="a"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    log = logging.info

    root = Path(cfg["dataset_root"])
    eval_monet_dir = Path(cfg.get(
        "eval_real_monet_dir",
        "/app/DATA266-Lab1-Pair45/task3_GAN/dataset/monet_jpg",
    ))
    eval_photo_dir = Path(cfg.get(
        "eval_real_photo_dir",
        "/app/DATA266-Lab1-Pair45/task3_GAN/dataset/photo_jpg",
    ))
    evaluator = NotebookEvaluator(
        list_images(eval_monet_dir),
        list_images(eval_photo_dir),
        device,
        n_eval=cfg.get("eval_num_images", 300),
        batch_size=cfg.get("eval_metric_batch_size", 32),
    )

    train_loader = create_dataloader(
        domain_a_dir=root / "train" / "photos",
        domain_b_dir=root / "train" / "monet",
        batch_size=cfg["batch_size"],
        train=True,
        shuffle=True,
        num_workers=cfg["num_workers"],
    )

    G_AB = Generator(3, 3, 9).to(device)
    G_BA = Generator(3, 3, 9).to(device)
    D_A = PatchGANDiscriminator(3).to(device)
    D_B = PatchGANDiscriminator(3).to(device)
    for model in (G_AB, G_BA, D_A, D_B):
        model.apply(init_weights)

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    lambda_cycle = cfg["lambda_cycle"]
    lambda_identity = cfg["lambda_identity"]

    lr_g = cfg["lr_G"]
    lr_d_a = cfg.get("lr_D_A", cfg["lr_D"])
    lr_d_b = cfg.get("lr_D_B", cfg["lr_D"])

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_AB.parameters(), G_BA.parameters()),
        lr=lr_g,
        betas=(cfg["beta1"], cfg["beta2"]),
    )
    optimizer_D_A = torch.optim.Adam(
        D_A.parameters(), lr=lr_d_a,
        betas=(cfg["beta1"], cfg["beta2"]),
    )
    optimizer_D_B = torch.optim.Adam(
        D_B.parameters(), lr=lr_d_b,
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
    start_epoch = 0
    best_average_fid = float("inf")
    latest_path = ckpt_dir / "latest.pth"

    if args.resume:
        if not latest_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {latest_path}")
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

        # The checkpoint contains the old nearly-zero learning rates. Reset
        # them explicitly for this fine-tuning experiment.
        set_optimizer_lr(optimizer_G, lr_g)
        set_optimizer_lr(optimizer_D_A, lr_d_a)
        set_optimizer_lr(optimizer_D_B, lr_d_b)
        scheduler_G.base_lrs = [lr_g]
        scheduler_D_A.base_lrs = [lr_d_a]
        scheduler_D_B.base_lrs = [lr_d_b]
        log(f"Resumed from epoch {start_epoch}; learning rates reset for fine-tuning.")

    metrics_path = run_dir / "fid_mifid_every_epoch.csv"
    if not metrics_path.exists():
        with open(metrics_path, "w", newline="") as file:
            csv.DictWriter(file, fieldnames=[
                "epoch", "fid_photo_to_monet", "mifid_photo_to_monet",
                "fid_monet_to_photo", "mifid_monet_to_photo",
                "fid_average", "mifid_average",
            ]).writeheader()

    log(f"Training batches per epoch: {len(train_loader)}")
    log(f"Starting at epoch {start_epoch + 1}; ending at epoch {epochs}")

    for epoch in range(start_epoch, epochs):
        epoch_number = epoch + 1
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()
        loss_sums = torch.zeros(5, device=device)
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            real_A = batch["A"].to(device, non_blocking=True)
            real_B = batch["B"].to(device, non_blocking=True)

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
                criterion_gan(prediction_fake_B, torch.ones_like(prediction_fake_B))
                + criterion_gan(prediction_fake_A, torch.ones_like(prediction_fake_A))
            )
            loss_cycle = (
                criterion_cycle(reconstructed_A, real_A)
                + criterion_cycle(reconstructed_B, real_B)
            )
            loss_identity = (
                criterion_identity(identity_A, real_A)
                + criterion_identity(identity_B, real_B)
            )
            loss_G = loss_gan + lambda_cycle * loss_cycle + lambda_identity * loss_identity
            loss_G.backward()
            optimizer_G.step()

            set_requires_grad([D_A, D_B], True)
            fake_A_for_D = pool_A.query(fake_A.detach())
            optimizer_D_A.zero_grad(set_to_none=True)
            loss_D_A = 0.5 * (
                criterion_gan(D_A(real_A), torch.ones_like(D_A(real_A)))
                + criterion_gan(D_A(fake_A_for_D), torch.zeros_like(D_A(fake_A_for_D)))
            )
            loss_D_A.backward(); optimizer_D_A.step()

            fake_B_for_D = pool_B.query(fake_B.detach())
            optimizer_D_B.zero_grad(set_to_none=True)
            loss_D_B = 0.5 * (
                criterion_gan(D_B(real_B), torch.ones_like(D_B(real_B)))
                + criterion_gan(D_B(fake_B_for_D), torch.zeros_like(D_B(fake_B_for_D)))
            )
            loss_D_B.backward(); optimizer_D_B.step()

            loss_sums += torch.stack([
                loss_G.detach(), loss_D_A.detach(), loss_D_B.detach(),
                loss_cycle.detach(), loss_identity.detach(),
            ])
            steps += 1

            if step == 1 or step % cfg["log_every"] == 0:
                log(
                    f"Epoch [{epoch_number}/{epochs}] Step [{step}/{len(train_loader)}] | "
                    f"G: {loss_G.item():.4f} | D_A: {loss_D_A.item():.4f} | "
                    f"D_B: {loss_D_B.item():.4f}"
                )

        scheduler_G.step(); scheduler_D_A.step(); scheduler_D_B.step()
        means = (loss_sums / steps).tolist()
        log(
            f"Epoch [{epoch_number}/{epochs}] done | G: {means[0]:.4f} | "
            f"D_A: {means[1]:.4f} | D_B: {means[2]:.4f} | "
            f"lr_G: {optimizer_G.param_groups[0]['lr']:.7f} | "
            f"lr_D_A: {optimizer_D_A.param_groups[0]['lr']:.7f} | "
            f"lr_D_B: {optimizer_D_B.param_groups[0]['lr']:.7f}"
        )

        scores = evaluator.evaluate(G_AB, G_BA, evaluation_dir, epoch_number)
        log(
            f"Epoch [{epoch_number}/{epochs}] evaluation | "
            f"Photo->Monet FID: {scores['fid_photo_to_monet']:.3f}, "
            f"MiFID: {scores['mifid_photo_to_monet']:.4f} | "
            f"Monet->Photo FID: {scores['fid_monet_to_photo']:.3f}, "
            f"MiFID: {scores['mifid_monet_to_photo']:.4f} | "
            f"Average FID: {scores['fid_average']:.3f}, "
            f"Average MiFID: {scores['mifid_average']:.4f}"
        )

        with open(metrics_path, "a", newline="") as file:
            row = {"epoch": epoch_number, **scores}
            csv.DictWriter(file, fieldnames=row.keys()).writerow(row)

        if scores["fid_average"] < best_average_fid:
            best_average_fid = scores["fid_average"]
            safe_save({
                "epoch": epoch_number,
                "scores": scores,
                "G_AB": G_AB.state_dict(),
                "G_BA": G_BA.state_dict(),
            }, ckpt_dir / "best_fid_mifid.pth")

        if epoch_number % cfg.get("sample_every", 1) == 0:
            # Keep the same compact diagnostic grid style as the original run.
            with torch.no_grad():
                evaluator_input = evaluator.real_photo_paths[:8]
                input_transform = T.Compose([
                    T.Resize((256, 256)), T.ToTensor(),
                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
                fixed = torch.stack([
                    input_transform(Image.open(p).convert("RGB"))
                    for p in evaluator_input
                ]).to(device)
                fake = G_AB(fixed)
                reconstruction = G_BA(fake)
                save_image(
                    torch.cat([fixed, fake, reconstruction]),
                    sample_dir / f"epoch_{epoch_number:03d}.png",
                    nrow=fixed.size(0), normalize=True, value_range=(-1, 1),
                )

        safe_save({
            "epoch": epoch_number,
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
        }, latest_path)

    log(f"Training finished. Best average notebook FID: {best_average_fid:.3f}")
    log(f"Per-epoch results: {metrics_path}")


if __name__ == "__main__":
    main()
