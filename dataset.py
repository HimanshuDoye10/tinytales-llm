import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


class TinyStoriesDataset(Dataset):
    """
    Autoregressive Language Modeling Dataset for TinyStories.
    Prepares input (X) and target (Y) tensor pairs shifted by 1 token.
    """
    def __init__(self, txt_path="TinyStories-valid.txt", tokenizer_path="tokenizer.json", block_size=256, cache_path="tokens.pt", split="all", split_ratio=0.9):
        self.block_size = block_size
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Check for cached pre-tokenized tensor to save startup time
        if os.path.exists(cache_path):
            all_tokens = torch.load(cache_path, weights_only=True)
        else:
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"Source text file '{txt_path}' not found.")
            print(f"Reading and tokenizing '{txt_path}'...")
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read()

            encoded = self.tokenizer.encode(text)
            all_tokens = torch.tensor(encoded.ids, dtype=torch.long)
            print(f"Saving pre-tokenized tokens ({len(all_tokens):,} tokens) to '{cache_path}'...")
            torch.save(all_tokens, cache_path)

        # Handle train/val splitting
        n_train = int(len(all_tokens) * split_ratio)
        if split == "train":
            self.tokens = all_tokens[:n_train]
        elif split == "val":
            self.tokens = all_tokens[n_train:]
        else:
            self.tokens = all_tokens

        # Total number of full training chunks of length block_size
        self.total_samples = (len(self.tokens) - 1) // self.block_size
        print(f"Dataset [{split}] ready. Tokens: {len(self.tokens):,}, Sequences: {self.total_samples:,}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size

        # X is tokens from start_idx to start_idx + block_size
        x = self.tokens[start_idx:end_idx]
        # Y is the target sequence shifted by 1 token to the right
        y = self.tokens[start_idx + 1:end_idx + 1]

        return x, y


def get_dataloader(
    txt_path="TinyStories-valid.txt",
    tokenizer_path="tokenizer.json",
    block_size=256,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    split="all",
    split_ratio=0.9
):
    dataset = TinyStoriesDataset(
        txt_path=txt_path,
        tokenizer_path=tokenizer_path,
        block_size=block_size,
        split=split,
        split_ratio=split_ratio
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    return dataloader, dataset


if __name__ == "__main__":
    # Test configuration
    BLOCK_SIZE = 256
    BATCH_SIZE = 64

    print("--- Initializing DataLoader ---")
    loader, dataset = get_dataloader(block_size=BLOCK_SIZE, batch_size=BATCH_SIZE)

    # Fetch a single batch
    x_batch, y_batch = next(iter(loader))

    print("\n--- Batch Verification ---")
    print(f"Batch X shape: {x_batch.shape}  (Expected: [{BATCH_SIZE}, {BLOCK_SIZE}])")
    print(f"Batch Y shape: {y_batch.shape}  (Expected: [{BATCH_SIZE}, {BLOCK_SIZE}])")
    print(f"Data type:     {x_batch.dtype}")

    # Inspect the first sequence in the batch
    print("\n--- Shifted Target (X vs Y) Verification ---")
    x_sample = x_batch[0]
    y_sample = y_batch[0]

    # Verify that Y is shifted by exactly 1 token compared to X
    assert torch.equal(x_sample[1:], y_sample[:-1]), "Error: Targets are not shifted by 1 token!"
    print("Verification Passed: Y is correctly shifted by exactly +1 token relative to X!")

    # Decode first 10 tokens to show preview
    preview_len = 10
    x_decoded = dataset.tokenizer.decode(x_sample[:preview_len].tolist(), skip_special_tokens=False)
    y_decoded = dataset.tokenizer.decode(y_sample[:preview_len].tolist(), skip_special_tokens=False)

    print(f"\nFirst {preview_len} tokens of X[0]: {x_sample[:preview_len].tolist()}")
    print(f"Decoded text of X[0]: '{x_decoded}'")

    print(f"\nFirst {preview_len} tokens of Y[0]: {y_sample[:preview_len].tolist()}")
    print(f"Decoded text of Y[0]: '{y_decoded}'")
