"""Data loading utilities for unpaired Photo-to-Monet CycleGAN training."""

from pathlib import Path
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_train_transform():
    """Transform used while training the CycleGAN."""
    return transforms.Compose([
        transforms.Resize((286, 286)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def get_test_transform():
    """Deterministic transform used for testing and inference."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])


def find_images(folder):
    """Return image paths from a folder, including supported extensions."""
    folder = Path(folder)
    paths = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not paths:
        raise ValueError(f"No supported images found in: {folder.resolve()}")

    return paths


class UnpairedImageDataset(Dataset):
    """Dataset that samples Domain A and Domain B independently."""

    def __init__(self, domain_a_dir, domain_b_dir, transform=None):
        self.domain_a_paths = find_images(domain_a_dir)
        self.domain_b_paths = find_images(domain_b_dir)
        self.transform = transform

    def __len__(self):
        # One epoch covers the larger domain.
        return max(len(self.domain_a_paths), len(self.domain_b_paths))

    def __getitem__(self, index):
        # Domain A follows its own index; Domain B is sampled independently.
        path_a = self.domain_a_paths[index % len(self.domain_a_paths)]
        path_b = random.choice(self.domain_b_paths)

        image_a = Image.open(path_a).convert("RGB")
        image_b = Image.open(path_b).convert("RGB")

        if self.transform is not None:
            image_a = self.transform(image_a)
            image_b = self.transform(image_b)

        return {
            "A": image_a,
            "B": image_b,
            "A_path": str(path_a),
            "B_path": str(path_b),
        }


def create_dataloader(
    domain_a_dir,
    domain_b_dir,
    batch_size=10,
    train=True,
    shuffle=True,
    num_workers=0,
):
    """Create a DataLoader for unpaired Photo and Monet images."""
    transform = get_train_transform() if train else get_test_transform()

    dataset = UnpairedImageDataset(
        domain_a_dir=domain_a_dir,
        domain_b_dir=domain_b_dir,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def denormalize(image_tensor):
    """Convert normalized image values from [-1, 1] back to [0, 1]."""
    return (image_tensor * 0.5 + 0.5).clamp(0, 1)
