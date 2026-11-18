# 🧠 Mini-LLM: Training a Decoder-Only Transformer from Scratch

A lightweight, from-scratch implementation of an autoregressive **Decoder-Only Transformer Language Model** (GPT / LLaMA architecture family) built in PyTorch and trained on the **TinyStories** dataset.

The model goes from complete gibberish (loss ~8.3) to writing fluent, coherent children's stories with characters, dialogue, and plot structure (loss ~2.3) in under 15 minutes of training.

---

## 🚀 Model Specifications

| Parameter | Value | Details |
| :--- | :--- | :--- |
| **Architecture** | **Decoder-Only Transformer** | Causal multi-head self-attention |
| **Total Parameters** | **~4.26 Million** | Lightweight, runs inference fast on CPU |
| **Vocabulary Size** | `4,096` | Trained via Byte-Level BPE |
| **Context Length (`block_size`)** | `256` tokens | Covers full TinyStories narratives |
| **Embedding Dimension (`n_embd`)** | `256` | Hidden dimension size |
| **Attention Heads (`n_head`)** | `4` heads | 64 dimensions per head |
| **Transformer Blocks (`n_layer`)** | `4` layers | Pre-LayerNorm + Residual connections |
| **Attention Mechanism** | `FlashAttention` / SDPA | Scaled dot-product attention via `torch.nn.functional` |
| **Weight Tying** | Enabled | Input token embeddings and LM head share weights |

---

## 📂 Project Structure

```text
mini-llm-training/
├── train_tokenizer.py    # Trains custom Byte-Level BPE tokenizer
├── tokenizer.json        # Saved vocabulary (4,096 tokens) & merge rules
├── dataset.py            # Autoregressive DataLoader & token caching (tokens.pt)
├── model.py              # PyTorch Decoder-Only Transformer architecture
├── train.py              # Mixed-precision (FP16) training engine with AdamW
├── generate.py           # Interactive story generation with temperature & top-k
├── requirements.txt      # Project dependencies
├── .gitignore            # Git exclusion rules for large weights and caches
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <YOUR_REPO_URL>
   cd mini-llm-training
   ```

2. **Create and activate a virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Step-by-Step Pipeline

### 1. Train Tokenizer
Trains a Byte-Level Byte-Pair Encoding (BPE) tokenizer on `TinyStories-valid.txt` with special tokens `["<|endoftext|>", "<|pad|>", "<|unk|>"]`:
```bash
python train_tokenizer.py
```
Outputs `tokenizer.json`.

### 2. Prepare & Verify Data Pipeline
Tokenizes raw text and caches the tokens into `tokens.pt` for instant loading in future runs. Creates input ($X$) and target ($Y$) sequence pairs shifted by $+1$ token:
```bash
python dataset.py
```

### 3. Train the Transformer Model
Trains the model using mixed-precision FP16, AdamW optimizer with cosine learning rate schedule, gradient clipping, and live console story previews every 250 steps:

**On Google Colab (T4 GPU recommended):**
```bash
python train.py --batch_size 64 --block_size 256 --max_iters 3000
```

**Quick local test run (CPU):**
```bash
python train.py --batch_size 16 --block_size 256 --max_iters 50 --eval_interval 25
```
Saves best model weights to `checkpoint_best.pt`.

---

## 📖 Generating Stories

### Interactive Mode (Terminal Chat)
Launch the interactive storytelling prompt:
```bash
python generate.py
```
Enter your custom prompt (e.g. `Once upon a time, a dragon wanted to make friends`) and watch the model complete the story in real time!

### Single Prompt Mode
```bash
python generate.py --prompt "One day, a friendly robot" --temperature 0.7 --max_tokens 150
```

#### Sampling Controls:
* `--temperature`: Controls randomness/creativity (`0.5`–`0.7` for focused & logical, `0.8`–`1.0` for creative).
* `--top_k`: Restricts token selection to top $K$ probabilities (default: `40`).
* `--max_tokens`: Maximum new tokens to generate (default: `150`).

---

## ✨ Sample Model Outputs

**Prompt:** *"One day, a friendly robot"*
> *"One day, a friendly robot came to visit. She was very hungry, so she decided to take a special gift.  
> The robot said, "I know what I found!". The little robot smiled and said, "Thank you for trying to find the robot!"  
> The robot smiled and said, "Thank you,, little and I love my robot!"*

**Prompt:** *"Once upon a time, Spot the dog found a red ball"*
> *"Once upon a time, Spot the dog found a red ball. The dog was very happy and wanted to play with the other ball. The dog was happy and said, "Thank you, Spot! I will share the ball with you." Spot agreed to play with the ball all day."*

---

## 📈 Training Progression

| Steps | Train Loss | Val Loss | Output Quality |
| :---: | :---: | :---: | :--- |
| **0** | ~8.40 | ~8.42 | Random characters and punctuation gibberish |
| **250** | ~4.50 | ~4.60 | Basic English syllables and frequent words |
| **1,000** | ~2.90 | ~3.05 | Simple phrases and characters introduced |
| **3,000** | ~2.18 | **~2.36** | Fluent sentences, character interactions, dialog with quotes |

---

## 📜 License
MIT License
