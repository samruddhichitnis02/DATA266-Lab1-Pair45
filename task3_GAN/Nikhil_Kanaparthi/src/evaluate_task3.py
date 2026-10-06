"""Evaluate Task 3 translations and write the instructor-compatible outputs.

The FID/MiFID definitions match the supplied instructor notebook:
real photos vs pred_A2B and real Monet paintings vs pred_B2A, with matching
image counts. Additional report-only metrics are calculated locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torchvision.transforms import InterpolationMode

from train_cyclegan import ResnetGenerator, build_transform, get_device, image_paths


ROOT = Path(__file__).resolve().parents[1]


def take_n(paths: Sequence[Path], n: int | None) -> List[Path]:
    return list(paths if n is None else paths[: min(n, len(paths))])


def make_inception(device: torch.device) -> nn.Module:
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    # torchvision requires aux_logits=True when ImageNet weights are loaded;
    # eval() returns the main feature tensor and ignores the auxiliary head.
    model = models.inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    model.to(device).eval()
    return model


INCEPTION_TRANSFORM = transforms.Compose([
    transforms.Resize(299, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


@torch.no_grad()
def activations(model: nn.Module, paths: Sequence[Path], device: torch.device, batch_size: int) -> np.ndarray:
    features: List[np.ndarray] = []
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start : start + batch_size]
        batch = torch.stack([INCEPTION_TRANSFORM(Image.open(path).convert("RGB")) for path in batch_paths]).to(device)
        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]
        features.append(output.detach().cpu().numpy().reshape(len(batch_paths), -1))
    if not features:
        raise ValueError("No images were found for evaluation")
    return np.concatenate(features, axis=0)


def frechet_distance(real: np.ndarray, generated: np.ndarray, eps: float = 1e-6) -> float:
    mu_real = real.mean(axis=0)
    mu_generated = generated.mean(axis=0)
    cov_real = np.cov(real, rowvar=False)
    cov_generated = np.cov(generated, rowvar=False)
    covmean, _ = scipy.linalg.sqrtm(cov_real.dot(cov_generated), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(cov_real.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((cov_real + offset).dot(cov_generated + offset))[0]
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_real - mu_generated
    return float(diff.dot(diff) + np.trace(cov_real + cov_generated - 2 * covmean))


def mifid(real: np.ndarray, generated: np.ndarray) -> float:
    count = min(len(real), len(generated))
    return float(np.mean([cosine(real[index], generated[index]) for index in range(count)]))


def kid(real: np.ndarray, generated: np.ndarray, subsets: int = 20, subset_size: int = 50, seed: int = 266) -> float:
    rng = np.random.default_rng(seed)
    values: List[float] = []
    n = min(len(real), len(generated), subset_size)
    dimension = real.shape[1]

    def polynomial(x, y):
        return (x.dot(y.T) / dimension + 1.0) ** 3

    for _ in range(subsets):
        real_part = real[rng.choice(len(real), n, replace=False)]
        generated_part = generated[rng.choice(len(generated), n, replace=False)]
        k_xx = polynomial(real_part, real_part)
        k_yy = polynomial(generated_part, generated_part)
        k_xy = polynomial(real_part, generated_part)
        np.fill_diagonal(k_xx, 0.0)
        np.fill_diagonal(k_yy, 0.0)
        value = k_xx.sum() / (n * (n - 1)) + k_yy.sum() / (n * (n - 1)) - 2 * k_xy.mean()
        values.append(float(value))
    return float(np.mean(values))


def precision_recall(real: np.ndarray, generated: np.ndarray, k: int = 3) -> Tuple[float, float]:
    k_real = min(k + 1, len(real))
    real_neighbors = NearestNeighbors(n_neighbors=k_real).fit(real)
    real_distances, _ = real_neighbors.kneighbors(real)
    real_radius = real_distances[:, -1]
    generated_to_real = NearestNeighbors(n_neighbors=1).fit(real)
    generated_distances, nearest_real = generated_to_real.kneighbors(generated)
    precision = float(np.mean(generated_distances[:, 0] <= real_radius[nearest_real[:, 0]]))

    k_generated = min(k + 1, len(generated))
    generated_neighbors = NearestNeighbors(n_neighbors=k_generated).fit(generated)
    generated_distances, _ = generated_neighbors.kneighbors(generated)
    generated_radius = generated_distances[:, -1]
    real_to_generated = NearestNeighbors(n_neighbors=1).fit(generated)
    real_distances_to_generated, nearest_generated = real_to_generated.kneighbors(real)
    recall = float(np.mean(real_distances_to_generated[:, 0] <= generated_radius[nearest_generated[:, 0]]))
    return precision, recall


def load_generators(checkpoint_path: Path, device: torch.device, blocks: int) -> Tuple[nn.Module, nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    generator_a2b = ResnetGenerator(blocks=blocks).to(device)
    generator_b2a = ResnetGenerator(blocks=blocks).to(device)
    generator_a2b.load_state_dict(checkpoint["G_A2B"])
    generator_b2a.load_state_dict(checkpoint["G_B2A"])
    generator_a2b.eval()
    generator_b2a.eval()
    return generator_a2b, generator_b2a


def content_vector(path: Path) -> np.ndarray:
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    return transform(Image.open(path).convert("RGB")).numpy().reshape(-1)


def content_cosine(inputs: Sequence[Path], generated: Sequence[Path]) -> float:
    values = []
    for input_path, generated_path in zip(inputs, generated):
        values.append(1.0 - cosine(content_vector(input_path), content_vector(generated_path)))
    return float(np.mean(values))


@torch.no_grad()
def cycle_metrics(
    generator_a2b: nn.Module,
    generator_b2a: nn.Module,
    paths_a: Sequence[Path],
    paths_b: Sequence[Path],
    image_size: int,
    device: torch.device,
) -> Dict[str, float]:
    transform = build_transform(image_size, train=False)
    l1_a: List[float] = []
    l1_b: List[float] = []
    for path in paths_a:
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        reconstruction = generator_b2a(generator_a2b(tensor))
        l1_a.append(float(torch.abs(reconstruction - tensor).mean().item()))
    for path in paths_b:
        tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        reconstruction = generator_a2b(generator_b2a(tensor))
        l1_b.append(float(torch.abs(reconstruction - tensor).mean().item()))
    return {
        "cycle_reconstruction_l1_A": float(np.mean(l1_a)),
        "cycle_reconstruction_l1_B": float(np.mean(l1_b)),
        "cycle_reconstruction_l1_mean": float(np.mean(l1_a + l1_b)),
    }


def lpips_cycle_metric(
    generator_a2b: nn.Module,
    generator_b2a: nn.Module,
    paths_a: Sequence[Path],
    paths_b: Sequence[Path],
    image_size: int,
    device: torch.device,
) -> Dict[str, float | str]:
    try:
        import lpips
    except ImportError:
        return {"lpips_A": float("nan"), "lpips_B": float("nan"), "lpips_mean": float("nan"), "lpips_note": "Install lpips to calculate this metric."}
    loss_fn = lpips.LPIPS(net="alex").to(device).eval()
    transform = build_transform(image_size, train=False)
    values_a: List[float] = []
    values_b: List[float] = []
    with torch.no_grad():
        for path in paths_a:
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            values_a.append(float(loss_fn(tensor, generator_b2a(generator_a2b(tensor))).mean().item()))
        for path in paths_b:
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            values_b.append(float(loss_fn(tensor, generator_a2b(generator_b2a(tensor))).mean().item()))
    return {"lpips_A": float(np.mean(values_a)), "lpips_B": float(np.mean(values_b)), "lpips_mean": float(np.mean(values_a + values_b))}


def human_audit_metrics(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"human_audit_status": "human_audit.csv not found; fill the generated template first."}
    frame = pd.read_csv(path)
    result: Dict[str, object] = {"human_audit_rows": int(len(frame))}
    for criterion in ("style", "content", "artifacts"):
        col_a = f"rater_1_{criterion}"
        col_b = f"rater_2_{criterion}"
        if col_a not in frame or col_b not in frame:
            continue
        valid = frame[[col_a, col_b]].dropna()
        if len(valid) >= 2:
            result[f"{criterion}_mean_rater_1"] = float(valid[col_a].mean())
            result[f"{criterion}_mean_rater_2"] = float(valid[col_b].mean())
            result[f"{criterion}_cohen_kappa"] = float(cohen_kappa_score(valid[col_a], valid[col_b]))
            result[f"{criterion}_percent_agreement"] = float(np.mean(valid[col_a].to_numpy() == valid[col_b].to_numpy()))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "cyclegan_final.pt"))
    parser.add_argument("--n-eval", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--residual-blocks", type=int, default=6)
    parser.add_argument("--seed", type=int, default=266)
    args = parser.parse_args()
    device = get_device()
    data_root = Path(args.data_root).expanduser().resolve()
    monet_real = take_n(image_paths(data_root / "monet_jpg"), args.n_eval)
    photo_real = take_n(image_paths(data_root / "photo_jpg"), args.n_eval)
    generated_a2b = take_n(image_paths(ROOT / "outputs" / "pred_A2B"), args.n_eval)
    generated_b2a = take_n(image_paths(ROOT / "outputs" / "pred_B2A"), args.n_eval)
    if min(len(monet_real), len(photo_real), len(generated_a2b), len(generated_b2a)) < 2:
        raise ValueError("Need real and generated images in all four evaluation folders")
    n = min(len(monet_real), len(photo_real), len(generated_a2b), len(generated_b2a))
    monet_real, photo_real, generated_a2b, generated_b2a = [paths[:n] for paths in (monet_real, photo_real, generated_a2b, generated_b2a)]
    print(f"Device: {device}; evaluating {n} images per direction")
    inception = make_inception(device)
    real_photo_features = activations(inception, photo_real, device, args.batch_size)
    generated_photo_features = activations(inception, generated_a2b, device, args.batch_size)
    real_monet_features = activations(inception, monet_real, device, args.batch_size)
    generated_monet_features = activations(inception, generated_b2a, device, args.batch_size)
    fid_a2b = frechet_distance(real_photo_features, generated_photo_features)
    fid_b2a = frechet_distance(real_monet_features, generated_monet_features)
    mifid_a2b = mifid(real_photo_features, generated_photo_features)
    mifid_b2a = mifid(real_monet_features, generated_monet_features)
    kid_a2b = kid(real_photo_features, generated_photo_features, seed=args.seed)
    kid_b2a = kid(real_monet_features, generated_monet_features, seed=args.seed + 1)
    precision_a2b, recall_a2b = precision_recall(real_photo_features, generated_photo_features)
    precision_b2a, recall_b2a = precision_recall(real_monet_features, generated_monet_features)
    generator_a2b, generator_b2a = load_generators(Path(args.checkpoint), device, args.residual_blocks)
    cycle = cycle_metrics(generator_a2b, generator_b2a, monet_real, photo_real, args.image_size, device)
    lpips_values = lpips_cycle_metric(generator_a2b, generator_b2a, monet_real, photo_real, args.image_size, device)
    content_a2b = content_cosine(monet_real, generated_a2b)
    content_b2a = content_cosine(photo_real, generated_b2a)
    metrics: Dict[str, object] = {
        "n_eval_per_direction": n,
        "device": str(device),
        "FID_A2B_Monet_to_Photo": fid_a2b,
        "FID_B2A_Photo_to_Monet": fid_b2a,
        "MiFID_A2B_Monet_to_Photo": mifid_a2b,
        "MiFID_B2A_Photo_to_Monet": mifid_b2a,
        "KID_A2B_Monet_to_Photo": kid_a2b,
        "KID_B2A_Photo_to_Monet": kid_b2a,
        "precision_A2B": precision_a2b,
        "precision_B2A": precision_b2a,
        "recall_A2B": recall_a2b,
        "recall_B2A": recall_b2a,
        "content_cosine_input_to_translation_A2B": content_a2b,
        "content_cosine_input_to_translation_B2A": content_b2a,
        **cycle,
        **lpips_values,
        **human_audit_metrics(ROOT / "human_audit.csv"),
        "kaggle_public_score": "TO_FILL_AFTER_SUBMISSION",
        "kaggle_private_score": "TO_FILL_AFTER_FINAL_LEADERBOARD",
        "kaggle_rank": "TO_FILL_AFTER_SUBMISSION",
    }
    ROOT.joinpath("metrics").mkdir(exist_ok=True)
    (ROOT / "metrics" / "full_metrics_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv(ROOT / "full_metrics_report.csv", index=False)
    submission = pd.DataFrame([{"ID": 1, "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2}])
    submission.to_csv(ROOT / "submission.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {ROOT / 'full_metrics_report.csv'} and {ROOT / 'submission.csv'}")


if __name__ == "__main__":
    main()
