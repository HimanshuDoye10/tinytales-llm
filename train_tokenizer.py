import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


def main():
    input_file = "TinyStories-valid.txt"
    output_file = "tokenizer.json"
    vocab_size = 4096
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")

    print(f"Initializing Byte-Level BPE tokenizer...")
    # Initialize tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))

    # Use ByteLevel pre-tokenizer and decoder
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # Trainer configuration
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training tokenizer on {input_file} (target vocab_size={vocab_size})...")
    tokenizer.train([input_file], trainer)

    # Save tokenizer
    tokenizer.save(output_file)
    print(f"Tokenizer successfully trained and saved to '{output_file}'.")
    print(f"Final vocabulary size: {tokenizer.get_vocab_size()}")

    # --- Quick Encode / Decode Test ---
    test_text = (
        'Once upon a time, Spot found a shiny toy! <|endoftext|>\n'
        'Unknown char test: 🚀 1234.'
    )
    print("\n--- Test Encode / Decode ---")
    print(f"Original Text:\n{test_text}\n")

    encoded = tokenizer.encode(test_text)
    print(f"Token IDs: {encoded.ids}")
    print(f"Tokens: {encoded.tokens}")

    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=False)
    print(f"\nDecoded Text (skip_special_tokens=False):\n{decoded}")

    assert test_text == decoded, "Decoded text does not match original text!"
    print("\nVerification Passed: Decoded text matches the original text perfectly!")


if __name__ == "__main__":
    main()
