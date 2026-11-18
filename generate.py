import argparse
import sys
import torch
from tokenizers import Tokenizer
from model import MiniLLM, MiniLLMConfig


def load_model_and_tokenizer(checkpoint_path="checkpoint_best.pt", tokenizer_path="tokenizer.json", device="cpu"):
    print(f"Loading tokenizer from '{tokenizer_path}'...")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    print(f"Loading model checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = MiniLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    step = checkpoint.get("step", "N/A")
    val_loss = checkpoint.get("val_loss", "N/A")
    print(f"Model successfully loaded! (Trained steps: {step}, Validation loss: {val_loss:.4f})")
    return model, tokenizer, config


def generate_story(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_k: int = 40,
    device: str = "cpu"
):
    model.eval()
    encoded = tokenizer.encode(prompt).ids
    idx = torch.tensor([encoded], dtype=torch.long, device=device)

    # Special token for stopping early if the story ends
    eot_token_id = tokenizer.token_to_id("<|endoftext|>")

    generated_ids = list(encoded)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            next_token = idx_next.item()

            if next_token == eot_token_id:
                break

            generated_ids.append(next_token)
            idx = torch.cat((idx, idx_next), dim=1)

    story = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return story


def interactive_mode(model, tokenizer, device, max_tokens, temperature, top_k):
    print("\n" + "=" * 60)
    print(" 📖 Mini-LLM Interactive Storyteller")
    print(" Type a prompt to begin a story, or 'exit'/'quit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("Enter Prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit", "q"]:
                print("Goodbye!")
                break

            print("\nGenerating story...\n")
            story = generate_story(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            print("-" * 60)
            print(story)
            print("-" * 60 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Generate stories using trained Mini-LLM")
    parser.add_argument("--checkpoint", type=str, default="checkpoint_best.pt")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json")
    parser.add_argument("--prompt", type=str, default=None, help="Starting text prompt")
    parser.add_argument("--max_tokens", type=int, default=150, help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (lower = more focused)")
    parser.add_argument("--top_k", type=int, default=40, help="Top-K sampling pool")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer, config = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device
    )

    if args.prompt is not None:
        print(f"\nPrompt: '{args.prompt}'")
        print("\n--- Generated Story ---")
        story = generate_story(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        print(story)
        print("-----------------------")
    else:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )


if __name__ == "__main__":
    main()
