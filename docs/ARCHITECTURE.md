# 🏗️ ARCHITECTURE.md — Arsitektur MambaLM

## Kenapa SSM, Bukan Transformer?

Transformer punya kelemahan fundamental: **attention complexity O(n²)**. Artinya kalau panjang sequence dua kali lipat, komputasi empat kali lipat. SSM (State Space Model) menyelesaikan ini dengan kompleksitas **O(n) linear**.

```
Transformer:  setiap token attend ke semua token sebelumnya
              → makin panjang sequence, makin berat

SSM (Mamba):  state ter-compress dalam hidden vector h(t)
              → panjang sequence tidak berpengaruh ke biaya per-step
```

---

## 🔢 Persamaan State Space

Jantung dari SSM adalah sistem persamaan diferensial diskrit:

```
h(t) = A · h(t-1) + B · x(t)    ← state update
y(t) = C · h(t)                  ← output
```

Keterangan:
- `x(t)` — input token pada waktu t
- `h(t)` — hidden state (memori model)
- `y(t)` — output pada waktu t
- `A`    — state transition matrix (dipelajari)
- `B`    — input projection matrix (input-dependent di Mamba)
- `C`    — output projection matrix (input-dependent di Mamba)

Mamba membuat B, C, dan delta **input-dependent** (selective), sehingga model bisa memilih informasi mana yang perlu diingat dan mana yang bisa dilupakan — mirip gating di LSTM tapi lebih efisien.

---

## 🗺️ Arsitektur End-to-End

```
Input Token IDs
      │
      ▼
┌─────────────────────┐
│   Token Embedding   │  vocab_size → d_model
│   (nn.Embedding)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│              SSM Block × N layers               │
│                                                 │
│   ┌─────────────────────────────────────────┐  │
│   │              LayerNorm                  │  │
│   └────────────────┬────────────────────────┘  │
│                    │                            │
│          ┌─────────┴─────────┐                 │
│          │   in_proj (×2)    │  d_model → d_inner×2
│          └────┬──────────────┘                 │
│               │                                 │
│       ┌───────┴───────┐                        │
│       │               │                        │
│   x_branch         z_gate                      │
│       │               │                        │
│   ┌───▼───┐           │                        │
│   │ Conv1D│  local context                     │
│   └───┬───┘                                    │
│       │                                         │
│   ┌───▼───────┐                                │
│   │  SSM Core │  selective state space         │
│   │  A,B,C,D  │                                │
│   └───┬───────┘                                │
│       │                                         │
│       └──── × SiLU(z_gate) ──── out_proj ──┐  │
│                                              │  │
│   residual connection ───────────────────────┘  │
└─────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Final LayerNorm   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     LM Head         │  d_model → vocab_size
│   (nn.Linear)       │  weight tied dengan embedding
└──────────┬──────────┘
           │
           ▼
    Logits (B, L, vocab_size)
           │
           ▼
    Softmax → Next Token Distribution
```

---

## 🔬 SSM Core — Detail Internal

### Selective Scan (Jantung Mamba)

```python
# Persamaan diskrit setelah Zero-Order Hold discretization:

dA = exp(delta * A)                    # (B, L, D, N) — state transition
dB = delta * B                         # (B, L, D, N) — input gate
                                       # delta = input-dependent step size

# Recurrent scan:
for t in range(seq_len):
    h = dA[:, t] * h + dB[:, t] * x[:, t]   # update state
    y = (h * C[:, t]).sum(-1)                 # read output
```

### Kenapa "Selective"?

Mamba berbeda dari SSM klasik karena `B`, `C`, dan `delta` bergantung pada input:

```
SSM klasik: A, B, C fixed → semua informasi diperlakukan sama
Mamba:      B, C, delta = f(x) → model MEMILIH info mana yang disimpan
```

Ini yang membuat Mamba bisa melakukan selective recall — mirip attention tapi O(n).

---

## 📐 Dimensi & Hyperparameter

```python
MambaConfig(
    vocab_size  = 100277,  # tiktoken cl100k vocab
    d_model     = 512,     # embedding dimension
    n_layers    = 12,      # jumlah SSM blocks
    d_state     = 16,      # SSM state dimension (N)
    d_conv      = 4,       # local conv kernel size
    expand      = 2,       # d_inner = d_model * expand = 1024
    dt_rank     = "auto",  # = max(1, d_inner // 16) = 64
)
```

### Hitung Jumlah Parameter

```
Per SSM Block:
  in_proj   : d_model × (d_inner×2)  = 512 × 1024   = 524,288
  conv1d    : d_inner × d_conv        = 1024 × 4     = 4,096
  x_proj    : d_inner × (dt+2N)       = 1024 × 96    = 98,304
  dt_proj   : dt_rank × d_inner       = 64 × 1024    = 65,536
  A_log     : d_inner × d_state       = 1024 × 16    = 16,384
  out_proj  : d_inner × d_model       = 1024 × 512   = 524,288
  layernorm : d_model × 2             = 1,024
              ─────────────────────────────────────
  Per block : ~1,234,000 params

Total (12 layers + embedding + head):
  Embedding  : 100,277 × 512          = 51,341,824
  Blocks ×12 : 12 × 1,234,000         = 14,808,000
  LM Head    : tied dengan embedding  = 0 extra
               ─────────────────────────────────────
  Total      : ~66M params
```

---

## 🔄 Dua Mode Operasi

### 1. Prefill Mode (Proses Prompt)
Jalankan seluruh sequence sekaligus — paralel, cepat.

```
x: (B, L, d_model) → SSM scan seluruh L token → y: (B, L, d_model)
```

### 2. Decode Mode (Generasi Token)
Satu token per step dengan SSM cache — inkremental.

```
x: (B, 1, d_model) → SSM step dengan cached h → y: (B, 1, d_model)
                                                       ↓
                                               next token
```

Tanpa cache: tiap token recompute seluruh sequence → O(n²) total
Dengan cache: tiap token hanya 1 SSM step → O(n) total

---

## ⚖️ Perbandingan Arsitektur

```
                MambaLM     GPT-2 Small   LLaMA-7B
─────────────────────────────────────────────────────
Params          ~66M        117M          7B
Architecture    SSM         Transformer   Transformer
Attention       ✗           ✓             ✓
Sequence comp.  O(n)        O(n²)         O(n²)
Context length  Unlimited*  1024          4096
Min GPU RAM     CPU OK      ~2GB          ~14GB
```

*SSM secara teori unlimited context, tapi praktiknya dibatasi RAM.

---

## 🔗 Referensi

- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao, 2023
- [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) — Gu et al., 2021
- [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083)