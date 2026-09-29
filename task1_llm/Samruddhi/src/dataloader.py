from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import load_datasets


def create_dataloaders(
    project_dir,
    block_size=512,
    batch_size=32
):
    """
    Create training and validation DataLoaders.
    """

    train_dataset, val_dataset = load_datasets(
        project_dir=project_dir,
        block_size=block_size
    )

    use_gpu = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=use_gpu
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=use_gpu
    )

    return train_loader, val_loader


if __name__ == "__main__":

    project_dir = Path(
        r"C:\Users\samruddhi\Desktop\Sam\Gen_AI_Lab1"
    )

    train_loader, val_loader = create_dataloaders(
        project_dir=project_dir,
        block_size=512,
        batch_size=32
    )

    input_batch, target_batch = next(iter(train_loader))

    print("DataLoader test completed.")
    print(f"Training batches per epoch: {len(train_loader):,}")
    print(f"Validation batches: {len(val_loader):,}")
    print(f"Input batch shape: {input_batch.shape}")
    print(f"Target batch shape: {target_batch.shape}")