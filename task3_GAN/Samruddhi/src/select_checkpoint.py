"""
select_checkpoint.py

Scores saved CycleGAN checkpoints the same way the competition evaluation
notebook does, in both directions, and picks the best one.

For every checkpoint it:
  1. translates photos -> Monet (G_AB) and Monet -> photos (G_BA),
  2. saves each output as a JPG (quality 95) in memory and reads it back,
  3. extracts Inception-v3 features (torchvision IMAGENET1K_V1, fc removed,
     resize 299, ImageNet normalisation) exactly as the notebook does,
  4. computes FID (Frechet distance) and MiFID (mean cosine distance, paired
     by sorted index) against the real images, per direction,
  5. averages the two directions, and averages FID with MiFID for the ranking.

The winner's images are written to <save_dir>/pred_A2B (Monet -> photo) and
<save_dir>/pred_B2A (photo -> Monet), the folder names the evaluation notebook
expects. All images come straight from the generators; nothing is edited.

Usage:
    python select_checkpoint.py \
        --ckpt_dir   runs/v3_bs1/checkpoints \
        --real_monet /path/to/monet_jpg \
        --real_photo /path/to/photo_jpg \
        --save_dir   selected_best
"""

import argparse
import csv
import glob
import io
import os
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

from models import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

GENERATOR_TF = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--real_monet", required=True)
    parser.add_argument("--real_photo", required=True)
    parser.add_argument("--save_dir", default="selected_best")
    parser.add_argument("--csv_path", default="checkpoint_scores.csv")
    parser.add_argument("--extra", nargs="*", default=[],
                        help="extra checkpoint files to score as well")
    parser.add_argument("--include_latest", action="store_true")
    parser.add_argument("--n_eval", type=int, default=300)
    parser.add_argument("--gen_batch", type=int, default=8)
    parser.add_argument("--feat_batch", type=int, default=32)
    parser.add_argument("--jpg_quality", type=int, default=95)
    return parser.parse_args()


def sorted_images(folder, n):
    """Same listing rule as the evaluation notebook: sorted, first n."""
    paths = []
    for ext in (".jpg", ".jpeg", ".png"):
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))[:n]


# ----------------------------------------------------------------------
# Metric pieces (same as the evaluation notebook)
# ----------------------------------------------------------------------

def get_inception():
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    inception.fc = nn.Identity()
    return inception.to(device).eval()


@torch.no_grad()
def inception_features(model, pil_images, batch_size):
    features = []
    for start in range(0, len(pil_images), batch_size):
        batch = torch.stack(
            [INCEPTION_TF(img) for img in pil_images[start:start + batch_size]]
        ).to(device)
        features.append(model(batch).cpu().numpy())
    return np.concatenate(features, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_and_mifid(real_feats, gen_feats):
    n = min(len(real_feats), len(gen_feats))
    real_feats, gen_feats = real_feats[:n], gen_feats[:n]

    mu_r, sig_r = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_g, sig_g = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    mifid = float(np.mean([cosine(real_feats[i], gen_feats[i])
                           for i in range(n)]))
    return fid, mifid


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def load_generator_inputs(paths):
    return torch.stack(
        [GENERATOR_TF(Image.open(p).convert("RGB")) for p in paths]
    )


@torch.no_grad()
def translate(generator, inputs, batch_size):
    """Returns uint8 arrays (H, W, 3), straight from the generator."""
    arrays = []
    for start in range(0, len(inputs), batch_size):
        x = inputs[start:start + batch_size].to(device)
        y = (generator(x) * 0.5 + 0.5).clamp(0.0, 1.0)
        y = (y * 255.0).round().byte().permute(0, 2, 3, 1).cpu().numpy()
        arrays.extend(y)
    return arrays


def jpeg_roundtrip(array, quality):
    """Save as JPG in memory and read it back, like the real submission files."""
    buffer = io.BytesIO()
    Image.fromarray(array, "RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def load_generators(path):
    checkpoint = torch.load(path, map_location=device)
    g_ab = Generator(3, 3, 9).to(device)
    g_ba = Generator(3, 3, 9).to(device)
    g_ab.load_state_dict(checkpoint["G_AB"])
    g_ba.load_state_dict(checkpoint["G_BA"])
    return g_ab.eval(), g_ba.eval(), checkpoint.get("epoch", "")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    monet_paths = sorted_images(args.real_monet, args.n_eval)
    photo_paths = sorted_images(args.real_photo, args.n_eval)
    if not monet_paths or not photo_paths:
        raise RuntimeError("No images found in the real Monet / photo folders.")
    print(f"Real Monet: {len(monet_paths)} | Real photo: {len(photo_paths)}")

    checkpoints = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "generators_epoch_*.pth"))
    )
    for name in ("best_fid.pth", "latest.pth" if args.include_latest else None):
        if name and os.path.exists(os.path.join(args.ckpt_dir, name)):
            checkpoints.append(os.path.join(args.ckpt_dir, name))
    for extra_path in args.extra:
        if not os.path.exists(extra_path):
            raise FileNotFoundError(f"--extra file not found: {extra_path}")
    checkpoints += list(args.extra)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {args.ckpt_dir}")
    print(f"Scoring {len(checkpoints)} checkpoints")

    inception = get_inception()

    # Real features are computed once and reused for every checkpoint.
    real_monet_feats = inception_features(
        inception, [Image.open(p).convert("RGB") for p in monet_paths],
        args.feat_batch,
    )
    real_photo_feats = inception_features(
        inception, [Image.open(p).convert("RGB") for p in photo_paths],
        args.feat_batch,
    )

    # Inputs: photos are translated to Monet, Monet paintings to photos.
    photo_inputs = load_generator_inputs(photo_paths)
    monet_inputs = load_generator_inputs(monet_paths)

    rows = []
    for path in checkpoints:
        try:
            g_ab, g_ba, epoch = load_generators(path)
        except Exception as error:  # e.g. file being written by training
            print(f"Skipping {path}: {error}", flush=True)
            continue

        fake_monet = [jpeg_roundtrip(a, args.jpg_quality)
                      for a in translate(g_ab, photo_inputs, args.gen_batch)]
        fake_photo = [jpeg_roundtrip(a, args.jpg_quality)
                      for a in translate(g_ba, monet_inputs, args.gen_batch)]

        fid_b2a, mifid_b2a = fid_and_mifid(
            real_monet_feats,
            inception_features(inception, fake_monet, args.feat_batch),
        )
        fid_a2b, mifid_a2b = fid_and_mifid(
            real_photo_feats,
            inception_features(inception, fake_photo, args.feat_batch),
        )

        avg_fid = (fid_a2b + fid_b2a) / 2
        avg_mifid = (mifid_a2b + mifid_b2a) / 2
        row = {
            "checkpoint": path,
            "epoch": epoch,
            "fid_photo_to_monet": round(fid_b2a, 3),
            "mifid_photo_to_monet": round(mifid_b2a, 4),
            "fid_monet_to_photo": round(fid_a2b, 3),
            "mifid_monet_to_photo": round(mifid_a2b, 4),
            "avg_fid": round(avg_fid, 3),
            "avg_mifid": round(avg_mifid, 4),
            "final_score": round((avg_fid + avg_mifid) / 2, 4),
        }
        rows.append(row)
        print(f"{Path(path).name:28s} epoch {str(epoch):>4s} | "
              f"P->M FID {fid_b2a:7.2f} | M->P FID {fid_a2b:7.2f} | "
              f"avg FID {avg_fid:7.2f} | avg MiFID {avg_mifid:.4f} | "
              f"score {row['final_score']:.4f}", flush=True)

    if not rows:
        raise RuntimeError("No checkpoint could be scored.")
    rows.sort(key=lambda r: r["final_score"])
    with open(args.csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    print("\nRanking (lower is better):")
    for rank, row in enumerate(rows, start=1):
        print(f"{rank:2d}. {Path(row['checkpoint']).name:28s} "
              f"avg FID {row['avg_fid']:8.3f} | "
              f"avg MiFID {row['avg_mifid']:.4f} | "
              f"score {row['final_score']:.4f}")
    print(f"\nBest checkpoint: {best['checkpoint']}")
    print(f"Scores written to: {args.csv_path}")

    # Write the winner's images in the folders the evaluation notebook reads.
    g_ab, g_ba, _ = load_generators(best["checkpoint"])
    out_a2b = Path(args.save_dir) / "pred_A2B"   # Monet -> photo
    out_b2a = Path(args.save_dir) / "pred_B2A"   # photo -> Monet
    out_a2b.mkdir(parents=True, exist_ok=True)
    out_b2a.mkdir(parents=True, exist_ok=True)

    for index, array in enumerate(translate(g_ba, monet_inputs, args.gen_batch)):
        Image.fromarray(array, "RGB").save(
            out_a2b / f"monet_to_photo_{index:05d}.jpg",
            format="JPEG", quality=args.jpg_quality,
        )
    for index, array in enumerate(translate(g_ab, photo_inputs, args.gen_batch)):
        Image.fromarray(array, "RGB").save(
            out_b2a / f"photo_to_monet_{index:05d}.jpg",
            format="JPEG", quality=args.jpg_quality,
        )

    with open(Path(args.save_dir) / "selected_checkpoint.txt", "w") as file:
        file.write(best["checkpoint"] + "\n")

    print(f"Images from the best checkpoint saved in: {args.save_dir}")


if __name__ == "__main__":
    main()