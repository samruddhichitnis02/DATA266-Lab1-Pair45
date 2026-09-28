from pathlib import Path
import torch
from torch.utils.data import Dataset


class CharacterSequenceDataset(Dataset):
    """
    Creates input-target pairs for character-level language modeling.
    """

    def __init__(self, token_tensor, block_size):
        self.tokens = token_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, index):
        # Input characters
        x = self.tokens[
            index:index + self.block_size
        ]

        # Same sequence shifted by one character
        y = self.tokens[
            index + 1:index + self.block_size + 1
        ]

        return x, y


def load_datasets(project_dir, block_size=512):
    """
    Load encoded tokens and create training and validation datasets.
    """

    processed_dir = (
        project_dir
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
        train_tokens,
        block_size
    )

    val_dataset = CharacterSequenceDataset(
        val_tokens,
        block_size
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    project_dir = Path(
        r"C:\Users\samruddhi\Desktop\Sam\Gen_AI_Lab1"
    )

    train_dataset, val_dataset = load_datasets(
        project_dir,
        block_size=512
    )

    print("Datasets created successfully.")
    print(f"Training sequences: {len(train_dataset):,}")
    print(f"Validation sequences: {len(val_dataset):,}")

    x, y = train_dataset[0]

    print(f"Input shape: {x.shape}")
    print(f"Target shape: {y.shape}")