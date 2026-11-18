import argparse
import math
import os
import time
from contextlib import nullcontext

import torch
from tokenizers import Tokenizer

from dataset import get_dataloader
from model import MiniLLM, MiniLLMConfig


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """
    Cosine learning rate schedule with linear warmup.
    """
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) If it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Separate parameters into decay (>=2D weights) and no-decay (<2D vectors: biases, layernorms).
    Handles weight-tying automatically and cleanly.
    """
    decay = []
    no_decay = []
    for pn, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)

    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    use_fused = (device_type == "cuda") and ("fused" in torch.optim.AdamW.__init__.__code__.co_varnames)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer


@torch.no_grad()
def estimate_loss(model, dataloader, eval_iters, ctx, device):
    """
    Estimates average loss across multiple batches.
    """
    model.eval()
    losses = []
    data_iter = iter(dataloader)
    for _ in range(eval_iters):
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def generate_story_preview(model, tokenizer, prompt_text, max_new_tokens, device):
    """
    Generates and prints a sample story to track qualitative progress.
    """
    model.eval()
    encoded = tokenizer.encode(prompt_text).ids
    idx = torch.tensor([encoded], dtype=torch.long, device=device)
    with torch.no_grad():
        generated_ids = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    story = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=False)
    model.train()
    return story


def train(args):
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device & Mixed Precision setup
    device = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"
    print(f"\n=======================================================")
    print(f" Mini-LLM Training Setup")
    print(f" Device:      {device.upper()} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f" Batch Size:  {args.batch_size}")
    print(f" Block Size:  {args.block_size}")
    print(f" Max Steps:   {args.max_iters}")
    print(f"=======================================================\n")

    # Mixed precision context
    ptdtype = torch.float16 if device_type == "cuda" else torch.float32
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
    scaler = torch.amp.GradScaler("cuda", enabled=(device_type == "cuda"))

    # Load Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    # Load DataLoaders
    print("Loading Train & Validation Datasets...")
    train_loader, _ = get_dataloader(
        txt_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        split="train",
        shuffle=True
    )
    val_loader, _ = get_dataloader(
        txt_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        block_size=args.block_size,
        batch_size=args.batch_size,
        split="val",
        shuffle=False
    )

    # Initialize Model
    print("\nInitializing Transformer Architecture...")
    config = MiniLLMConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout
    )
    model = MiniLLM(config).to(device)
    num_params = model.get_num_params()
    print(f"Total Model Parameters: {num_params:,} (~{num_params / 1e6:.2f}M)")

    # Optimizer
    optimizer = configure_optimizers(
        model=model,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=device_type
    )

    train_iter = iter(train_loader)
    best_val_loss = float("inf")
    t0 = time.time()

    print("\n--- Starting Training Loop ---")
    for step in range(args.max_iters + 1):
        # Determine and set learning rate for this step
        lr = get_lr(step, args.warmup_iters, args.max_iters, args.learning_rate, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic Evaluation and Live Story Generation
        if step % args.eval_interval == 0 or step == args.max_iters:
            val_loss = estimate_loss(model, val_loader, args.eval_iters, ctx, device)
            train_loss_est = estimate_loss(model, train_loader, args.eval_iters, ctx, device)
            dt = time.time() - t0

            print(f"\n[Step {step:>5}/{args.max_iters}] | Train Loss: {train_loss_est:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.2e} | Elapsed: {dt:.1f}s")

            # Checkpoint best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "step": step,
                    "val_loss": val_loss,
                }
                torch.save(checkpoint, args.checkpoint_path)
                print(f"--> Saved best model checkpoint to '{args.checkpoint_path}' (Val Loss: {val_loss:.4f})")

            # Live generation preview
            prompt = "Once upon a time"
            story = generate_story_preview(model, tokenizer, prompt, max_new_tokens=45, device=device)
            print(f"\n--- Live Generation Preview [Step {step}] ---")
            print(f"{story}")
            print(f"----------------------------------------------\n")

            t0 = time.time()
            if step == args.max_iters:
                break

        # Fetch next batch
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        # Forward + backward with mixed precision
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits, loss = model(x, y)

        scaler.scale(loss).backward()
        if args.grad_clip > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Step logging
        if step % args.log_interval == 0:
            print(f"Step {step:>5}/{args.max_iters} | Step Loss: {loss.item():.4f} | LR: {lr:.2e}")

    print("\nTraining Complete!")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best model weights saved to: '{args.checkpoint_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Mini-LLM on TinyStories")
    # Data & Architecture
    parser.add_argument("--data_path", type=str, default="TinyStories-valid.txt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint_best.pt")
    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--min_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Logging & Evaluation
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force_cpu", action="store_true")

    args = parser.parse_args()
    train(args)
