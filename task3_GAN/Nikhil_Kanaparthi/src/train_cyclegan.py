"""DATA266 Lab 1 Task 3: CycleGAN trained from scratch.

Domain A is Monet and domain B is Photo. The implementation contains two
ResNet generators and two PatchGAN discriminators; no pretrained generator,
discriminator, or image model is used for training.

Example commands:
    python src/train_cyclegan.py --smoke-test --data-root ../../data
    python src/train_cyclegan.py --data-root ../../data --epochs 10

The evaluator and human-audit worksheet are separate scripts so that generated
images and the training evidence remain traceable to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid, save_image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 266
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT_DIR = ROOT / "checkpoints"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        reset = getattr(torch.mps, "reset_peak_memory_stats", None)
        if reset is not None:
            try:
                reset()
            except Exception:
                pass


def peak_memory_mb(device: torch.device) -> float | None:
    values: List[float] = []
    if device.type == "cuda":
        values.extend([
            float(torch.cuda.max_memory_allocated(device)) / (1024**2),
            float(torch.cuda.max_memory_reserved(device)) / (1024**2),
        ])
    elif device.type == "mps" and hasattr(torch, "mps"):
        for method_name in ("max_memory_allocated", "driver_allocated_memory", "current_allocated_memory"):
            method = getattr(torch.mps, method_name, None)
            if method is not None:
                try:
                    values.append(float(method()) / (1024**2))
                except Exception:
                    pass
    return max(values) if values else None


def image_paths(folder: Path) -> List[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix in extensions)


class UnpairedImageDataset(Dataset):
    def __init__(self, monet_dir: Path, photo_dir: Path, transform, max_images: int | None = None):
        self.monet_paths = image_paths(monet_dir)
        self.photo_paths = image_paths(photo_dir)
        if max_images is not None:
            self.monet_paths = self.monet_paths[:max_images]
            self.photo_paths = self.photo_paths[:max_images]
        if not self.monet_paths or not self.photo_paths:
            raise FileNotFoundError(f"Expected images in {monet_dir} and {photo_dir}")
        self.transform = transform

    def __len__(self) -> int:
        return max(len(self.monet_paths), len(self.photo_paths))

    def __getitem__(self, index: int):
        monet_path = self.monet_paths[index % len(self.monet_paths)]
        photo_path = self.photo_paths[random.randrange(len(self.photo_paths))]
        monet = Image.open(monet_path).convert("RGB")
        photo = Image.open(photo_path).convert("RGB")
        return {"A": self.transform(monet), "B": self.transform(photo), "A_path": str(monet_path), "B_path": str(photo_path)}


class ReplayBuffer:
    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.images: List[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        returned: List[torch.Tensor] = []
        for image in images.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                returned.append(image)
            elif random.random() > 0.5:
                index = random.randrange(self.capacity)
                old = self.images[index].clone()
                self.images[index] = image.clone()
                returned.append(old)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend([
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3),
            nn.InstanceNorm2d(channels),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, input_channels: int = 3, output_channels: int = 3, base_channels: int = 64, blocks: int = 6):
        super().__init__()
        layers: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, base_channels, kernel_size=7),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(inplace=True),
        ]
        channels = base_channels
        for _ in range(2):
            layers.extend([
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(inplace=True),
            ])
            channels *= 2
        for _ in range(blocks):
            layers.append(ResidualBlock(channels))
        for _ in range(2):
            layers.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(inplace=True),
            ])
            channels //= 2
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(base_channels, output_channels, kernel_size=7), nn.Tanh()])
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int = 3, base_channels: int = 64):
        super().__init__()

        def block(in_channels: int, out_channels: int, normalize: bool = True):
            layers: List[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(input_channels, base_channels, normalize=False),
            *block(base_channels, base_channels * 2),
            *block(base_channels * 2, base_channels * 4),
            *block(base_channels * 4, base_channels * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def initialize_weights(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname and hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm2d" in classname and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


def trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def grad_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().norm(2).item() ** 2)
    return math.sqrt(total)


def set_requires_grad(modules: Sequence[nn.Module], value: bool) -> None:
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad = value


def normalized_image(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor.detach().cpu() * 0.5 + 0.5).clamp(0, 1)


def save_preview(epoch: int, real_a, fake_b, cycle_a, real_b, fake_a, cycle_b) -> None:
    rows = torch.cat([real_a[:1], fake_b[:1], cycle_a[:1], real_b[:1], fake_a[:1], cycle_b[:1]], dim=0)
    grid = make_grid(normalized_image(rows), nrow=3, padding=2)
    save_image(grid, OUTPUT_DIR / f"preview_epoch_{epoch:03d}.png")


def save_checkpoint(epoch: int, state: Dict[str, object], name: str) -> None:
    checkpoint = dict(state)
    checkpoint["epoch"] = epoch
    checkpoint["G_A2B"] = {key: value.detach().cpu() for key, value in state["G_A2B"].state_dict().items()}
    checkpoint["G_B2A"] = {key: value.detach().cpu() for key, value in state["G_B2A"].state_dict().items()}
    checkpoint["D_A"] = {key: value.detach().cpu() for key, value in state["D_A"].state_dict().items()}
    checkpoint["D_B"] = {key: value.detach().cpu() for key, value in state["D_B"].state_dict().items()}
    for key in ("G_A2B_model", "G_B2A_model", "D_A_model", "D_B_model"):
        checkpoint.pop(key, None)
    torch.save(checkpoint, CHECKPOINT_DIR / name)


def build_transform(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size + 30, image_size + 30), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def generate_translations(
    generator: nn.Module,
    paths: Sequence[Path],
    output_dir: Path,
    transform,
    device: torch.device,
    prefix: str,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        for number, path in enumerate(paths):
            image = Image.open(path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            translated = generator(tensor)
            save_image(normalized_image(translated[0]), output_dir / f"{prefix}_{number:05d}_{path.stem}.png")
    return len(paths)


def plot_losses(history: Dict[str, List[float]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    epochs = history["epoch"]
    for key in ("G_total", "D_A", "D_B", "cycle_A", "cycle_B", "identity_A", "identity_B"):
        axes[0].plot(epochs, history[key], label=key)
    for key in ("gradient_G_mean", "gradient_D_mean"):
        axes[1].plot(epochs, history[key], label=key)
    axes[0].set_ylabel("Loss")
    axes[1].set_ylabel("Mean gradient norm")
    axes[1].set_xlabel("Epoch")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    if args.resume and args.smoke_test:
        raise ValueError("Do not combine --resume with --smoke-test. Resume a real training run instead.")
    seed_everything(args.seed)
    device = get_device()
    image_size = 128 if args.smoke_test else args.image_size
    epochs = 1 if args.smoke_test else args.epochs
    max_images = 16 if args.smoke_test else args.max_images
    blocks = 2 if args.smoke_test else args.residual_blocks
    batch_size = 1 if args.smoke_test else args.batch_size
    data_root = Path(args.data_root).expanduser().resolve()
    monet_dir = data_root / "monet_jpg"
    photo_dir = data_root / "photo_jpg"
    if not monet_dir.is_dir() or not photo_dir.is_dir():
        raise FileNotFoundError(f"Expected {monet_dir} and {photo_dir}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "raw_logs").mkdir(parents=True, exist_ok=True)
    transform = build_transform(image_size, train=True)
    dataset = UnpairedImageDataset(monet_dir, photo_dir, transform, max_images=max_images)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    print(f"Device: {device}")
    print(f"Monet images used for training: {len(dataset.monet_paths)}")
    print(f"Photo images used for training: {len(dataset.photo_paths)}")
    print(f"Steps per epoch: {len(loader)}; epochs: {epochs}; image size: {image_size}")

    G_A2B = ResnetGenerator(blocks=blocks).to(device)
    G_B2A = ResnetGenerator(blocks=blocks).to(device)
    D_A = PatchDiscriminator().to(device)
    D_B = PatchDiscriminator().to(device)
    for model in (G_A2B, G_B2A, D_A, D_B):
        model.apply(initialize_weights)
    print("Parameters:", {"G_A2B": trainable_parameters(G_A2B), "G_B2A": trainable_parameters(G_B2A), "D_A": trainable_parameters(D_A), "D_B": trainable_parameters(D_B)})

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    optimizer_G = torch.optim.Adam(
        list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.learning_rate, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    def schedule(epoch: int) -> float:
        if epoch < args.decay_start_epoch:
            return 1.0
        denominator = max(epochs - args.decay_start_epoch + 1, 1)
        return max(0.0, 1.0 - (epoch - args.decay_start_epoch) / denominator)

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, schedule)
    scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, schedule)
    scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, schedule)
    fake_a_buffer = ReplayBuffer()
    fake_b_buffer = ReplayBuffer()
    lambda_cycle = args.lambda_cycle
    lambda_identity = args.lambda_identity
    history: Dict[str, List[float]] = {key: [] for key in [
        "epoch", "G_total", "G_gan_A2B", "G_gan_B2A", "cycle_A", "cycle_B", "identity_A", "identity_B",
        "D_A", "D_B", "gradient_G_mean", "gradient_D_mean", "learning_rate",
    ]}

    start_epoch = 0
    previous_training_seconds = 0.0
    if args.resume:
        resume_path = Path(args.resume).expanduser()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        for module, key in ((G_A2B, "G_A2B"), (G_B2A, "G_B2A"), (D_A, "D_A"), (D_B, "D_B")):
            module.load_state_dict(checkpoint[key])
        if "optimizer_G" in checkpoint:
            optimizer_G.load_state_dict(checkpoint["optimizer_G"])
        if "optimizer_D_A" in checkpoint:
            optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A"])
        if "optimizer_D_B" in checkpoint:
            optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B"])
        start_epoch = int(checkpoint.get("epoch", 0))
        if start_epoch >= epochs:
            raise ValueError(f"Checkpoint is already at epoch {start_epoch}; --epochs must be greater than that.")

        # The first 10-epoch run and the 100-epoch continuation can use the
        # same optimizer weights while following the 100-epoch LR schedule.
        # Position each scheduler immediately after the completed epoch.
        for scheduler, optimizer in (
            (scheduler_G, optimizer_G),
            (scheduler_D_A, optimizer_D_A),
            (scheduler_D_B, optimizer_D_B),
        ):
            scheduler.last_epoch = start_epoch
            factor = schedule(start_epoch)
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = base_lr * factor

        history = checkpoint.get("history", {})
        if not history:
            history = {key: [] for key in [
                "epoch", "G_total", "G_gan_A2B", "G_gan_B2A", "cycle_A", "cycle_B",
                "identity_A", "identity_B", "D_A", "D_B", "gradient_G_mean",
                "gradient_D_mean", "learning_rate",
            ]}
        previous_config_path = ROOT / "config.json"
        if previous_config_path.exists():
            previous_config = json.loads(previous_config_path.read_text(encoding="utf-8"))
            previous_training_seconds = float(previous_config.get("training_time_seconds", 0.0))
        print(f"Resuming from epoch {start_epoch}; target epoch: {epochs}")

    reset_memory(device)
    synchronize(device)
    start_time = time.perf_counter()
    nan_count = 0
    all_generator_grads: List[float] = []
    all_discriminator_grads: List[float] = []

    for epoch in range(start_epoch + 1, epochs + 1):
        epoch_values = {key: [] for key in history if key not in ("epoch", "learning_rate")}
        for batch in loader:
            real_a = batch["A"].to(device)
            real_b = batch["B"].to(device)
            valid_a = torch.ones_like(D_A(real_a))
            valid_b = torch.ones_like(D_B(real_b))
            fake_label_a = torch.zeros_like(valid_a)
            fake_label_b = torch.zeros_like(valid_b)

            set_requires_grad([D_A, D_B], False)
            optimizer_G.zero_grad(set_to_none=True)
            fake_b = G_A2B(real_a)
            fake_a = G_B2A(real_b)
            cycle_a = G_B2A(fake_b)
            cycle_b = G_A2B(fake_a)
            identity_a = G_B2A(real_a)
            identity_b = G_A2B(real_b)
            loss_gan_a2b = criterion_gan(D_B(fake_b), valid_b)
            loss_gan_b2a = criterion_gan(D_A(fake_a), valid_a)
            loss_cycle_a = criterion_cycle(cycle_a, real_a)
            loss_cycle_b = criterion_cycle(cycle_b, real_b)
            loss_identity_a = criterion_identity(identity_a, real_a)
            loss_identity_b = criterion_identity(identity_b, real_b)
            loss_g = (
                loss_gan_a2b + loss_gan_b2a
                + lambda_cycle * (loss_cycle_a + loss_cycle_b)
                + lambda_identity * (loss_identity_a + loss_identity_b)
            )
            if not torch.isfinite(loss_g):
                nan_count += 1
                continue
            loss_g.backward()
            generator_grad = grad_norm(G_A2B) + grad_norm(G_B2A)
            optimizer_G.step()

            set_requires_grad([D_A, D_B], True)
            optimizer_D_A.zero_grad(set_to_none=True)
            loss_real_a = criterion_gan(D_A(real_a), valid_a)
            loss_fake_a = criterion_gan(D_A(fake_a_buffer.query(fake_a)), fake_label_a)
            loss_d_a = 0.5 * (loss_real_a + loss_fake_a)
            loss_d_a.backward()
            discriminator_grad_a = grad_norm(D_A)
            optimizer_D_A.step()

            optimizer_D_B.zero_grad(set_to_none=True)
            loss_real_b = criterion_gan(D_B(real_b), valid_b)
            loss_fake_b = criterion_gan(D_B(fake_b_buffer.query(fake_b)), fake_label_b)
            loss_d_b = 0.5 * (loss_real_b + loss_fake_b)
            loss_d_b.backward()
            discriminator_grad_b = grad_norm(D_B)
            optimizer_D_B.step()

            losses = {
                "G_total": float(loss_g.item()), "G_gan_A2B": float(loss_gan_a2b.item()),
                "G_gan_B2A": float(loss_gan_b2a.item()), "cycle_A": float(loss_cycle_a.item()),
                "cycle_B": float(loss_cycle_b.item()), "identity_A": float(loss_identity_a.item()),
                "identity_B": float(loss_identity_b.item()), "D_A": float(loss_d_a.item()),
                "D_B": float(loss_d_b.item()), "gradient_G_mean": float(generator_grad),
                "gradient_D_mean": float((discriminator_grad_a + discriminator_grad_b) / 2),
            }
            for key, value in losses.items():
                if not math.isfinite(value):
                    nan_count += 1
                else:
                    epoch_values[key].append(value)
            all_generator_grads.append(float(generator_grad))
            all_discriminator_grads.append(float((discriminator_grad_a + discriminator_grad_b) / 2))

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()
        history["epoch"].append(epoch)
        for key in epoch_values:
            history[key].append(float(np.mean(epoch_values[key])) if epoch_values[key] else float("nan"))
        history["learning_rate"].append(float(optimizer_G.param_groups[0]["lr"]))
        print(
            f"epoch {epoch}/{epochs} G={history['G_total'][-1]:.4f} "
            f"D_A={history['D_A'][-1]:.4f} D_B={history['D_B'][-1]:.4f} "
            f"cycle=({history['cycle_A'][-1]:.4f},{history['cycle_B'][-1]:.4f})"
        )
        with torch.no_grad():
            save_preview(epoch, real_a, fake_b, cycle_a, real_b, fake_a, cycle_b)
        state = {
            "G_A2B": G_A2B,
            "G_B2A": G_B2A,
            "D_A": D_A,
            "D_B": D_B,
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D_A": optimizer_D_A.state_dict(),
            "optimizer_D_B": optimizer_D_B.state_dict(),
            "scheduler_G": scheduler_G.state_dict(),
            "scheduler_D_A": scheduler_D_A.state_dict(),
            "scheduler_D_B": scheduler_D_B.state_dict(),
            "config": vars(args),
            "history": history,
        }
        if epoch % args.checkpoint_every == 0 or epoch == epochs:
            save_checkpoint(epoch, state, f"cyclegan_epoch_{epoch:03d}.pt")
        if epoch == epochs:
            save_checkpoint(epoch, state, "cyclegan_final.pt")

    synchronize(device)
    total_seconds = previous_training_seconds + (time.perf_counter() - start_time)
    plot_losses(history)
    data_root = Path(args.data_root).expanduser().resolve()
    eval_transform = build_transform(image_size, train=False)
    photo_output_paths = image_paths(data_root / "photo_jpg")
    monet_output_paths = image_paths(data_root / "monet_jpg")
    if args.smoke_test:
        photo_output_paths = photo_output_paths[:16]
        monet_output_paths = monet_output_paths[:16]
    generate_translations(G_B2A, photo_output_paths, OUTPUT_DIR / "pred_B2A", eval_transform, device, "photo_to_monet")
    generate_translations(G_A2B, monet_output_paths, OUTPUT_DIR / "pred_A2B", eval_transform, device, "monet_to_photo")
    config = {
        **vars(args),
        "device": str(device),
        "operating_system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "data_root_is_external": True,
        "monet_count": len(image_paths(data_root / "monet_jpg")),
        "photo_count": len(image_paths(data_root / "photo_jpg")),
        "parameter_count": {
            "G_A2B": trainable_parameters(G_A2B), "G_B2A": trainable_parameters(G_B2A),
            "D_A": trainable_parameters(D_A), "D_B": trainable_parameters(D_B),
        },
        "training_time_seconds": total_seconds,
        "images_processed": len(loader) * batch_size * epochs,
        "images_per_second": (len(loader) * batch_size * epochs) / max(total_seconds, 1e-9),
        "peak_memory_mb": peak_memory_mb(device),
        "gradient_norm_mean": float(np.mean(all_generator_grads)) if all_generator_grads else float("nan"),
        "gradient_norm_max": float(np.max(all_generator_grads)) if all_generator_grads else float("nan"),
        "discriminator_gradient_norm_mean": float(np.mean(all_discriminator_grads)) if all_discriminator_grads else float("nan"),
        "nan_count": nan_count,
        "checkpoint": "checkpoints/cyclegan_final.pt",
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (ROOT / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (ROOT / "run_manifest.json").write_text(json.dumps({
        "config": config,
        "raw_data_not_committed": True,
        "output_mapping": {"pred_A2B": "Monet to Photo", "pred_B2A": "Photo to Monet"},
    }, indent=2), encoding="utf-8")
    print("Training and translation generation complete.")
    print(f"A2B outputs: {OUTPUT_DIR / 'pred_A2B'}")
    print(f"B2A outputs: {OUTPUT_DIR / 'pred_B2A'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Folder containing monet_jpg and photo_jpg")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", default=None, help="Checkpoint path to continue from")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--decay-start-epoch", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
