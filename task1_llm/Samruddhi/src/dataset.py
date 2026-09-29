from pathlib import Path

import torch
from torch.utils.data import Dataset


class CharacterSequenceDataset(Dataset):
    """
    Creates fixed-length character sequences.

    Training:
        Randomly selects a starting position.

    Validation:
        Uses fixed non-overlapping chunks for stable evaluation.

    Input:
        Characters start through start + block_size - 1

    Target:
        Characters start + 1 through start + block_size
    """

    def __init__(
        self,
        token_tensor,
        block_size,
        random_sampling=True
    ):
        self.tokens = token_tensor
        self.block_size = block_size
        self.random_sampling = random_sampling

        if len(self.tokens) <= self.block_size:
            raise ValueError(
                "Token tensor must contain more tokens than block_size."
            )

        self.max_start = len(self.tokens) - self.block_size

        if self.random_sampling:
            # Approximately one sample per non-overlapping block
            self.samples_per_epoch = max(
                1,
                self.max_start // self.block_size
            )
        else:
            # Deterministic non-overlapping validation chunks
            self.samples_per_epoch = max(
                1,
                self.max_start // self.block_size
            )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, index):
        if self.random_sampling:
            # Random start position for training
            start = torch.randint(
                low=0,
                high=self.max_start + 1,
                size=()
            ).item()
        else:
            # Deterministic non-overlapping position for validation
            start = index * self.block_size

            # Safety check for the final validation sample
            if start > self.max_start:
                start = self.max_start

        input_sequence = self.tokens[
            start:start + self.block_size
        ]

        target_sequence = self.tokens[
            start + 1:start + self.block_size + 1
        ]

        return input_sequence, target_sequence


def load_datasets(project_dir, block_size=512):
    """
    Load processed token files and create datasets.
    """

    processed_dir = (
        Path(project_dir)
        / "task1_llm"
        / "Samruddhi"
        / "data_processed"
    )

    train_tokens_file = processed_dir / "train_tokens.pt"
    val_tokens_file = processed_dir / "val_tokens.pt"

    train_tokens = torch.load(
        train_tokens_file,
        weights_only=True
    )

    val_tokens = torch.load(
        val_tokens_file,
        weights_only=True
    )

    train_dataset = CharacterSequenceDataset(
        token_tensor=train_tokens,
        block_size=block_size,
        random_sampling=True
    )

    val_dataset = CharacterSequenceDataset(
        token_tensor=val_tokens,
        block_size=block_size,
        random_sampling=False
    )

    return train_dataset, val_dataset


if __name__ == "__main__":

    project_dir = Path(
        r"C:\Users\samruddhi\Desktop\Sam\Gen_AI_Lab1"
    )

    train_dataset, val_dataset = load_datasets(
        project_dir=project_dir,
        block_size=512
    )

    train_input, train_target = train_dataset[0]
    val_input, val_target = val_dataset[0]

    print("Dataset test completed.")
    print(f"Training examples per epoch: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    print(f"Training input shape: {train_input.shape}")
    print(f"Training target shape: {train_target.shape}")
    print(f"Validation input shape: {val_input.shape}")
    print(f"Validation target shape: {val_target.shape}")