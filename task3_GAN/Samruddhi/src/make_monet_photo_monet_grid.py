"""Generate Monet -> Photo -> Reconstructed Monet samples.

Run from the project's src directory:
  python make_monet_photo_monet_grid.py

The script loads best_fid_mifid.pth when available; otherwise it uses
latest.pth from the v3_finetune run.
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

from models import Generator


DEFAULT_MONET_DIR = Path(
    "/app/DATA266-Lab1-Pair45/task3_GAN/dataset/monet_jpg"
)
DEFAULT_RUN_DIR = Path("runs/v3_finetune")


def list_images(folder):
    return sorted(
        path for path in Path(folder).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def load_generator_weights(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    G_AB = Generator(3, 3, 9).to(device)
    G_BA = Generator(3, 3, 9).to(device)

    G_AB.load_state_dict(checkpoint["G_AB"])
    G_BA.load_state_dict(checkpoint["G_BA"])
    G_AB.eval()
    G_BA.eval()
    return G_AB, G_BA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--monet_dir", default=str(DEFAULT_MONET_DIR))
    parser.add_argument("--output", default=None)
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = DEFAULT_RUN_DIR

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        best_path = run_dir / "checkpoints" / "best_fid_mifid.pth"
        latest_path = run_dir / "checkpoints" / "latest.pth"
        checkpoint_path = best_path if best_path.exists() else latest_path

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    monet_paths = list_images(args.monet_dir)[:args.num_samples]
    if len(monet_paths) < args.num_samples:
        raise RuntimeError(f"Not enough Monet images in {args.monet_dir}")

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    real_monet = torch.stack([
        transform(Image.open(path).convert("RGB"))
        for path in monet_paths
    ]).to(device)

    G_AB, G_BA = load_generator_weights(checkpoint_path, device)

    with torch.no_grad():
        generated_photo = G_BA(real_monet)
        reconstructed_monet = G_AB(generated_photo)

    grid = torch.cat([
        real_monet,
        generated_photo,
        reconstructed_monet,
    ])

    output_path = Path(args.output) if args.output else (
        run_dir / "monet_photo_monet_grid.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        grid,
        output_path,
        nrow=args.num_samples,
        normalize=True,
        value_range=(-1, 1),
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Saved: {output_path}")
    print("Rows: original Monet | generated Photo | reconstructed Monet")


if __name__ == "__main__":
    main()
