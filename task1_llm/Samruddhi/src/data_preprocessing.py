from pathlib import Path
import json
import re
import torch


def load_and_clean_stories(file_path):
    """
    Load one story per line and perform basic text cleaning.
    """
    stories = []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            story = line.strip()

            if not story:
                continue

            # Replace repeated whitespace with one space
            story = re.sub(r"\s+", " ", story)

            stories.append(story)

    return stories


def build_vocabulary(train_stories, processed_dir):
    """
    Build a character-level vocabulary using training data only.
    """
    unique_characters = sorted(set("".join(train_stories)))

    unk_token = "<UNK>"
    vocabulary = [unk_token] + unique_characters

    char_to_id = {
        character: index
        for index, character in enumerate(vocabulary)
    }

    id_to_char = {
        str(index): character
        for character, index in char_to_id.items()
    }

    vocab_file = processed_dir / "vocabulary.json"

    with open(vocab_file, "w", encoding="utf-8") as file:
        json.dump(
            {
                "char_to_id": char_to_id,
                "id_to_char": id_to_char
            },
            file,
            indent=2,
            ensure_ascii=False
        )

    return char_to_id, id_to_char, unk_token


def encode_text(text, char_to_id, unk_token):
    """
    Convert characters into integer IDs.
    Unknown characters are mapped to <UNK>.
    """
    return [
        char_to_id.get(character, char_to_id[unk_token])
        for character in text
    ]


def preprocess_dataset(project_dir):
    """
    Complete preprocessing pipeline:
    1. Load and clean stories
    2. Build vocabulary
    3. Encode text
    4. Save vocabulary and token tensors
    """

    data_dir = project_dir / "task1_llm" / "data"
    processed_dir = (
        project_dir
        / "task1_llm"
        / "Samruddhi"
        / "data_processed"
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_file = data_dir / "tinystories_train_100k.txt"
    val_file = data_dir / "tinystories_val_10k.txt"

    # Load and clean the stories
    train_stories = load_and_clean_stories(train_file)
    val_stories = load_and_clean_stories(val_file)

    # Build vocabulary using training data only
    char_to_id, id_to_char, unk_token = build_vocabulary(
        train_stories,
        processed_dir
    )

    # Join stories with spaces between them
    train_text = " ".join(train_stories)
    val_text = " ".join(val_stories)

    # Encode the text
    train_token_ids = encode_text(
        train_text,
        char_to_id,
        unk_token
    )

    val_token_ids = encode_text(
        val_text,
        char_to_id,
        unk_token
    )

    # Convert encoded data into PyTorch tensors
    train_tokens = torch.tensor(
        train_token_ids,
        dtype=torch.long
    )

    val_tokens = torch.tensor(
        val_token_ids,
        dtype=torch.long
    )

    # Save encoded tensors
    train_tokens_file = processed_dir / "train_tokens.pt"
    val_tokens_file = processed_dir / "val_tokens.pt"

    torch.save(train_tokens, train_tokens_file)
    torch.save(val_tokens, val_tokens_file)

    print("Preprocessing completed successfully.")
    print(f"Training stories: {len(train_stories):,}")
    print(f"Validation stories: {len(val_stories):,}")
    print(f"Vocabulary size: {len(char_to_id):,}")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Validation tokens: {len(val_tokens):,}")
    print(f"Saved vocabulary to: {processed_dir / 'vocabulary.json'}")
    print(f"Saved training tokens to: {train_tokens_file}")
    print(f"Saved validation tokens to: {val_tokens_file}")

    return {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "char_to_id": char_to_id,
        "id_to_char": id_to_char,
        "processed_dir": processed_dir
    }


if __name__ == "__main__":
    project_dir = Path(
        r"C:\Users\samruddhi\Desktop\Sam\Gen_AI_Lab1"
    )

    preprocess_dataset(project_dir)