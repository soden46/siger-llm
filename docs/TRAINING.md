# 🏋️ TRAINING.md — Panduan Training MambaLM

## Overview Pipeline

```
Raw Corpus (teks mentah)
      │
      ▼  tokenizer/tokenizer.py
Token IDs (list of integers)
      │
      ▼  training/dataset.py
TextDataset (sliding window chunks)
      │
      ▼  torch.utils.data.DataLoader
Batches (B, seq_len)
      │
      ▼  training/trainer.py
Training Loop
  ├── Forward pass → logits
  ├── Cross-entropy loss (next token prediction)
  ├── Backward pass → gradients
  ├── Gradient clipping
  ├── AdamW optimizer step
  └── Cosine LR scheduler
      │
      ▼  training/checkpoint.py
checkpoints/step_XXXXXXX.pt
      │
      ▼
best_model.pt ✅
```

---

## Konfigurasi Training

```python
# main.py — TRAIN_CONFIG

TRAIN_CONFIG = {
    # ── Model ──────────────────────────────────────────────
    "vocab_size":    100277,   # tiktoken cl100k
    "d_model":       512,      # embedding dimension
    "n_layers":      12,       # jumlah SSM blocks

    # ── Training ───────────────────────────────────────────
    "max_steps":     100_000,  # total training steps
    "batch_size":    8,        # samples per step
    "max_seq_len":   1024,     # panjang sequence maksimum
    "grad_accum_steps": 4,     # effective batch = 8 × 4 = 32

    # ── Optimizer ──────────────────────────────────────────
    "max_lr":        3e-4,     # peak learning rate
    "min_lr":        3e-5,     # minimum LR (10% of max)
    "warmup_steps":  2_000,    # linear warmup steps
    "weight_decay":  0.1,      # L2 regularization
    "grad_clip":     1.0,      # max gradient norm

    # ── Logging & Saving ───────────────────────────────────
    "log_interval":  10,       # print setiap N steps
    "save_every":    500,      # checkpoint setiap N steps
    "checkpoint_dir": "./checkpoints",
    "num_workers":   2,        # DataLoader workers
}
```

---

## Dataset

### Format yang Didukung

Training menggunakan **next token prediction** — model belajar memprediksi token berikutnya dari context.

```python
# Input  : token[0], token[1], ..., token[n-1]
# Target : token[1], token[2], ..., token[n]
# (shift by 1)
```

### Sliding Window

Teks panjang dipotong dengan sliding window agar tidak ada informasi yang terbuang:

```
Text: [t0, t1, t2, ..., t999]   (1000 tokens)
max_seq_len = 512, stride = 512

Chunk 1: [t0   ... t511]
Chunk 2: [t512 ... t999] (padded)
```

### Corpus yang Direkomendasikan

| Dataset | Bahasa | Size | Link |
|---|---|---|---|
| CC-100 (id) | Indonesia | ~10GB | [statmt.org](https://data.statmt.org/cc-100/) |
| Wikipedia ID | Indonesia | ~400MB | [HuggingFace](https://huggingface.co/datasets/wikimedia/wikipedia) |
| OSCAR (id) | Indonesia | ~5GB | [HuggingFace](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) |
| RedPajama | Multilingual | Besar | [HuggingFace](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T) |
| The Pile | English | ~800GB | [HuggingFace](https://huggingface.co/datasets/EleutherAI/pile) |

### Load dari File Lokal

```python
# training/dataset.py — load_corpus()

texts = load_corpus([
    "./data/wikipedia_id.txt",
    "./data/cc100_id.txt",
    "./data/custom_indo.txt",
])

dataset = TextDataset(
    texts       = texts,
    tokenizer   = tok,
    max_seq_len = 1024,
    stride      = 512,    # overlap 50% untuk lebih banyak chunks
)
```

---

## Optimizer — AdamW + Cosine Decay

### Kenapa AdamW?

AdamW memisahkan weight decay dari gradient update, menghasilkan regularisasi yang lebih bersih dibanding Adam + L2.

### Selective Weight Decay

```python
# Parameter 1D (bias, LayerNorm) → NO weight decay
# Parameter 2D+ (Linear weights) → weight decay = 0.1

# Ini trick dari GPT-2 paper yang terbukti lebih stabil
```

### Learning Rate Schedule

```
LR
▲
│     ╭──────────────╮
│    ╱ warmup         ╲ cosine decay
│   ╱                  ╲
│  ╱                    ╲___________
│ ╱                                  min_lr
└──────────────────────────────────→ steps
   warmup_steps        max_steps
```

```
max_lr      = 3e-4
min_lr      = 3e-5 (10% of max)
warmup_steps = 2000 (linear warmup)
decay       = cosine dari step 2000 → max_steps
```

---

## Gradient Accumulation

Karena RAM VPS terbatas, pakai gradient accumulation untuk simulate batch besar:

```python
grad_accum_steps = 4
batch_size       = 8

# Effective batch size = 8 × 4 = 32
# Tapi RAM usage = batch_size = 8 saja
```

```python
# Training step dengan grad accumulation:
for i, (x, y) in enumerate(dataloader):
    loss = model(x, y) / grad_accum_steps
    loss.backward()                           # accumulate gradient

    if (i + 1) % grad_accum_steps == 0:
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

## Hyperparameter Guide

### Ukuran Model vs RAM

| Config | Params | FP32 RAM | INT8 RAM | Speed CPU |
|---|---|---|---|---|
| Tiny (d=128, L=4) | ~5M | ~20MB | ~5MB | Fast |
| Small (d=256, L=8) | ~20M | ~80MB | ~20MB | Medium |
| **Base (d=512, L=12)** | **~66M** | **~260MB** | **~70MB** | **Recommended** |
| Medium (d=768, L=16) | ~200M | ~800MB | ~200MB | Slow |
| Large (d=1024, L=24) | ~500M | ~2GB | ~500MB | Very Slow |

**Recommended untuk VPS 4GB:** `d_model=512, n_layers=12`

### Tips Hyperparameter

```python
# Kalau loss tidak turun:
# → Naikkan learning rate (3e-4 → 5e-4)
# → Cek data pipeline (bisa jadi ada bug di tokenizer)
# → Tambah warmup steps (1000 → 3000)

# Kalau loss meledak (NaN/Inf):
# → Turunkan learning rate
# → Turunkan grad_clip (1.0 → 0.5)
# → Cek ada NaN di input data

# Kalau training lambat banget:
# → Kurangi max_seq_len (1024 → 512)
# → Kurangi batch_size (8 → 4)
# → Naikan grad_accum untuk maintain effective batch
```

---

## Monitoring Training

### Output Log

```
step=     10 | loss=10.8432 | avg_loss=10.9 | ppl=54281 | lr=1.5e-06 | tok/s=1234
step=     20 | loss=9.2341  | avg_loss=10.2 | ppl=2731  | lr=3.0e-06 | tok/s=1289
step=    100 | loss=6.8901  | avg_loss=7.1  | ppl=1212  | lr=1.5e-05 | tok/s=1301
step=   1000 | loss=4.2341  | avg_loss=4.5  | ppl=90    | lr=2.8e-04 | tok/s=1289
step=  10000 | loss=2.8901  | avg_loss=2.9  | ppl=18    | lr=2.1e-04 | tok/s=1277
```

### Target Loss per Phase

```
Phase 1 (0-1k steps)    : loss 8-11  → model baru mulai belajar
Phase 2 (1k-10k steps)  : loss 4-6   → mulai kenal pola bahasa
Phase 3 (10k-50k steps) : loss 2-4   → struktur kalimat terbentuk
Phase 4 (50k+ steps)    : loss 1.5-2.5 → model koheren
```

### Perplexity Target

```
PPL > 1000  → model belum belajar apa-apa
PPL 100-999 → mulai ada pola
PPL 50-100  → cukup OK untuk model kecil
PPL 15-50   → bagus ✅
PPL < 15    → excellent (butuh data & compute besar)
```

---

## Checkpoint Management

```python
# Checkpoint disimpan otomatis setiap save_every steps
# Format: checkpoints/step_0000500_20240101_120000.pt

# Load checkpoint untuk resume training
trainer.train(dataset, resume=True)

# Manual load checkpoint tertentu
step, loss = ckpt_manager.load(
    model, optimizer, scheduler,
    path="./checkpoints/step_0005000_xxx.pt"
)

# Best model disimpan terpisah
# → checkpoints/best_model.pt (loss terendah)
```

---

## Jalanin Training

```bash
# Activate venv
source venv/bin/activate

# Training penuh
python main.py

# Training dengan custom config
python -c "
from main import main
import sys
sys.argv = ['main.py']
main()
"

# Monitor RAM & CPU
watch -n 2 'free -h && ps aux | grep python'

# Background training (VPS)
nohup python main.py > training.log 2>&1 &
tail -f training.log
```

---

## Multi-Dataset Training

```python
# Campurkan beberapa corpus dengan rasio berbeda
import random

indo_texts  = load_corpus(["./data/indo/*.txt"])    # 70%
en_texts    = load_corpus(["./data/english/*.txt"]) # 20%
code_texts  = load_corpus(["./data/code/*.txt"])    # 10%

# Weighted sampling
all_texts = (
    indo_texts  * 7 +
    en_texts    * 2 +
    code_texts  * 1
)
random.shuffle(all_texts)

dataset = TextDataset(all_texts, tokenizer, max_seq_len=1024)
```