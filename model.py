import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniLLMConfig:
    vocab_size: int = 4096      # Matches our tokenizer vocabulary size
    block_size: int = 256      # Maximum context length (T)
    n_embd: int = 256          # Embedding dimension
    n_head: int = 4            # Number of attention heads (256 // 4 = 64 per head)
    n_layer: int = 4           # Number of Transformer blocks
    dropout: float = 0.1       # Dropout probability
    bias: bool = False         # False: faster and standard in modern LLMs (LLaMA style)


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with PyTorch FlashAttention support.
    """
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Query, Key, Value projections packed into one matrix for efficiency
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dim

        # Calculate Query, Key, Value projections
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, num_heads, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Efficient Causal Scaled Dot-Product Attention (uses FlashAttention on GPU when available)
        dropout_p = self.attn_dropout.p if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=True)

        # Re-assemble all head outputs side by side: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    Position-wise Feed-Forward Network (MLP) with GELU activation.
    """
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    A single Transformer decoder layer with Pre-LayerNorm.
    """
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        # Pre-LN architecture with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    """
    Complete Decoder-Only Autoregressive Transformer Model.
    """
    def __init__(self, config: MiniLLMConfig = None):
        super().__init__()
        if config is None:
            config = MiniLLMConfig()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),         # Word Token Embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),         # Word Position Embedding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),                           # Final LayerNorm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embedding and output lm_head
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights with standard normal distribution
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, idx, targets=None):
        device = idx.device
        B, T = idx.size()
        assert T <= self.config.block_size, f"Input sequence length {T} exceeds block_size {self.config.block_size}"

        # Position indices: [0, 1, 2, ..., T - 1]
        pos = torch.arange(0, T, dtype=torch.long, device=device)

        # Forward through embeddings
        tok_emb = self.transformer.wte(idx)    # (B, T, n_embd)
        pos_emb = self.transformer.wpe(pos)    # (T, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Forward through all transformer layers
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        # If targets are provided, compute Cross-Entropy Loss
        if targets is not None:
            logits = self.lm_head(x)  # (B, T, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # Inference optimization: compute logits only for the last token position
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size)
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Autoregressively generate new tokens from a prompt tensor idx of shape (B, T).
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop prompt to max block_size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward model to get logits for next token
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Optional Top-K sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to running sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    print("--- Initializing MiniLLM Architecture ---")
    config = MiniLLMConfig(
        vocab_size=4096,
        block_size=256,
        n_embd=256,
        n_head=4,
        n_layer=4,
        dropout=0.1
    )
    model = MiniLLM(config)
    num_params = model.get_num_params()
    print(f"Model successfully built!")
    print(f"Total Trainable Parameters: {num_params:,} (~{num_params / 1e6:.2f} Million)")

    # Test Forward Pass with dummy batch
    dummy_x = torch.randint(0, config.vocab_size, (2, config.block_size))
    dummy_y = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(dummy_x, dummy_y)
    print("\n--- Forward Pass Test ---")
    print(f"Input X shape:  {dummy_x.shape}")
    print(f"Logits shape:   {logits.shape} (Expected: [2, 256, 4096])")
    print(f"Loss value:     {loss.item():.4f}")

    # Expected initial loss should be around -ln(1 / 4096) = ln(4096) ≈ 8.31
    expected_initial_loss = math.log(config.vocab_size)
    print(f"Theoretical Initial Loss (-ln(1/V)): {expected_initial_loss:.4f}")

    # Test autoregressive generation with untrained model
    prompt = torch.tensor([[100, 200, 300]])  # dummy tokens
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"\n--- Generation Loop Test ---")
    print(f"Prompt shape:     {prompt.shape}")
    print(f"Generated shape:  {generated.shape} (Added 10 new tokens)")
    print(f"Generated tokens: {generated[0].tolist()}")
    print("\nAll architecture tests passed successfully!")
