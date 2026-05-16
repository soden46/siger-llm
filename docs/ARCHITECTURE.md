# 🏗️ ARCHITECTURE.md — Arsitektur SIGERLM

## Overview

**SIGERLM** adalah proyek eksperimen Large Language Model yang dibangun dari nol menggunakan pendekatan **State Space Model (SSM) / Mamba-like architecture**, bukan Transformer.

Project ini dirancang untuk:

- membangun model bahasa ringan dari scratch,
- mendukung **Indonesian + English + Code**,
- dikembangkan menuju dukungan **Bahasa Lampung Dialek O/Nyo**,
- dapat dilatih, di-fine-tune, dievaluasi, dioptimasi, lalu dideploy ke **CPU/VPS**.

Arsitektur keseluruhan proyek tidak hanya mencakup model inti, tetapi juga:

- tokenizer multilingual,
- training pipeline,
- LoRA fine-tuning,
- translation dataset pipeline Lampung,
- evaluation suite,
- ONNX/quantization optimization,
- FastAPI deployment.

---

# 1. Kenapa SSM, Bukan Transformer?

Transformer punya kelemahan fundamental: **attention complexity O(n²)**.

Artinya, jika panjang sequence menjadi dua kali lipat, komputasi attention dapat meningkat sekitar empat kali lipat.

SSM menyelesaikan masalah ini dengan menyimpan konteks dalam **state terkompresi**, sehingga kompleksitas sequence dapat bergerak menuju **O(n) linear**.

```txt
Transformer:
setiap token melihat semua token sebelumnya
→ semakin panjang sequence, semakin mahal komputasinya

SSM / Mamba:
informasi masa lalu dipadatkan dalam hidden state h(t)
→ biaya pemrosesan lebih stabil terhadap panjang sequence
```

---

# 2. Gagasan Inti State Space Model

Jantung dari SSM adalah sistem state update:

```txt
h(t) = A · h(t-1) + B · x(t)    ← state update
y(t) = C · h(t)                  ← output
```

Keterangan:

- `x(t)` — input token pada waktu ke-t
- `h(t)` — hidden state / memori model
- `y(t)` — output pada waktu ke-t
- `A` — state transition matrix
- `B` — input projection
- `C` — output projection

Dalam pendekatan **Mamba**, `B`, `C`, dan `delta` dibuat **input-dependent**, sehingga model tidak memperlakukan semua token secara sama.

Model belajar:

- informasi mana yang perlu disimpan,
- informasi mana yang bisa dilupakan,
- informasi mana yang perlu dikeluarkan ke output.

Ini disebut **selective state space modeling**.

---

# 3. Arsitektur Sistem SIGERLM Secara Keseluruhan

```txt
┌──────────────────────────────────────────────────────────────┐
│                         DATA LAYER                           │
│                                                              │
│  General Corpus:                                             │
│  - Indonesian text                                           │
│  - English text                                              │
│  - Code snippets                                             │
│                                                              │
│  Lampung Translation Dataset:                                │
│  - Lampung O                                                  │
│  - Lampung Nyo                                                │
│  - Indonesian                                                 │
│  - English                                                    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                       TOKENIZER LAYER                        │
│                                                              │
│  MultilingualTokenizer                                       │
│  - Tiktoken cl100k_base                                       │
│  - special tokens                                             │
│  - language tags                                             │
│  - chat turn tokens                                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                           │
│                                                              │
│  SIGERLM                                                     │
│  - Token Embedding                                           │
│  - N × SSMBlock                                              │
│  - Final LayerNorm                                           │
│  - LM Head                                                   │
│  - Weight tying embedding ↔ LM head                          │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      TRAINING LAYER                          │
│                                                              │
│  Base Training                                               │
│  - TextDataset sliding window                                │
│  - AdamW optimizer                                           │
│  - cosine LR scheduler                                       │
│  - checkpoint manager                                        │
│  - logger                                                    │
│                                                              │
│  Fine-Tuning                                                 │
│  - LoRA adapters                                             │
│  - InstructionDataset                                        │
│  - assistant-only loss mask                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                         │
│                                                              │
│  - Perplexity                                                │
│  - MMLU                                                      │
│  - ARC                                                        │
│  - IndoNLI / IndoQA / Sentiment                              │
│  - BLEU / ROUGE / Diversity                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION LAYER                        │
│                                                              │
│  - Benchmark                                                 │
│  - CPU threading                                             │
│  - KV/SSM cache experiment                                   │
│  - ONNX export                                               │
│  - INT8 / INT4 quantization                                  │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                           │
│                                                              │
│  - Generator                                                 │
│  - Sampler                                                   │
│  - ChatSession                                               │
│  - FastAPI REST API                                          │
│  - SSE streaming                                             │
│  - VPS deployment                                            │
└──────────────────────────────────────────────────────────────┘
```

---

# 4. Struktur Modul Project

```txt
siger-llm/
├── config/
│   └── model_config.py
│
├── model/
│   ├── ssm_core.py
│   ├── ssm_block.py
│   └── siger_model.py
│
├── tokenizer/
│   ├── tokenizer.py
│   ├── special_tokens.py
│   ├── trainer.py
│   ├── vocab_extender.py
│   └── tests/
│
├── training/
│   ├── dataset.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── checkpoint.py
│   └── logger.py
│
├── lora/
│   ├── config.py
│   ├── dataset.py
│   ├── layer.py
│   ├── model.py
│   ├── trainer.py
│   ├── run_lora.py
│   └── merge.py
│
├── inference/
│   ├── generator.py
│   ├── sampler.py
│   ├── chat.py
│   └── api.py
│
├── evaluation/
│   ├── perplexity.py
│   ├── benchmarks.py
│   ├── generation.py
│   ├── indo_eval.py
│   ├── runner.py
│   ├── run_eval.py
│   └── report.py
│
├── optimization/
│   ├── benchmark.py
│   ├── kvcache.py
│   ├── cpu/
│   │   └── threading.py
│   ├── onnx/
│   │   └── export.py
│   └── quantization/
│       ├── calibrate.py
│       └── quantize.py
│
├── tools/
│   ├── extract_kamus_pdf.py
│   ├── extract_smt_paper.py
│   ├── inspect_nusax_format.py
│   ├── normalize_text.py
│   ├── build_lampung_dataset.py
│   └── build_instruction_dataset.py
│
├── data/
│   ├── raw/
│   └── lampung/
│
├── checkpoints/
│   └── tokenizer/
│       └── tokenizer_config.json
│
├── main.py
├── AGENTS.md
├── PROJECT_CONTEXT.md
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 5. Tokenizer Architecture

SIGERLM menggunakan `MultilingualTokenizer` berbasis **Tiktoken `cl100k_base`** dengan tambahan special tokens.

## 5.1 Special Tokens

Contoh special tokens yang digunakan:

```txt
<|endoftext|>
<|pad|>
<|unk|>
<|system|>
<|user|>
<|assistant|>
<|end_turn|>
<|lang_id|>
<|id|>
<|en|>
<|code|>
<|bos|>
<|eos|>
<|sep|>
```

Special token ini penting untuk:

- chat format,
- instruction tuning,
- pembeda role system/user/assistant,
- penanda bahasa,
- BOS/EOS handling.

---

## 5.2 Tokenization Flow

```txt
Raw Text
    │
    ▼
MultilingualTokenizer.encode()
    │
    ▼
Token IDs
    │
    ▼
Model Input Tensor
```

Output decode:

```txt
Generated Token IDs
    │
    ▼
MultilingualTokenizer.decode()
    │
    ▼
Readable Text
```

---

# 6. Model Core: SIGERLM

## 6.1 End-to-End Forward Pass

```txt
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
│   LayerNorm                                      │
│      │                                           │
│      ▼                                           │
│   in_proj                                         │
│      │                                           │
│      ├──────────────► x_branch                   │
│      │                                           │
│      └──────────────► z_gate                     │
│                                                  │
│   x_branch                                       │
│      │                                           │
│      ▼                                           │
│   Depthwise Conv1D                               │
│      │                                           │
│      ▼                                           │
│   SiLU                                           │
│      │                                           │
│      ▼                                           │
│   SSM Core                                       │
│      │                                           │
│      ▼                                           │
│   × SiLU(z_gate)                                 │
│      │                                           │
│      ▼                                           │
│   out_proj                                       │
│      │                                           │
│      ▼                                           │
│   Residual Connection                            │
└──────────┬──────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│   Final LayerNorm   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       LM Head       │  d_model → vocab_size
│   weight tied with  │
│      embedding      │
└──────────┬──────────┘
           │
           ▼
    Logits (B, L, vocab_size)
```

---

# 7. Detail SSM Block

Setiap `SSMBlock` terdiri dari:

1. `LayerNorm`
2. `in_proj`
3. split ke:
   - `x_branch`
   - `z_gate`
4. depthwise `Conv1D`
5. `SSMCore`
6. gated multiplication
7. `out_proj`
8. residual connection

---

## 7.1 SSMBlock Pseudocode

```python
residual = x

x = LayerNorm(x)

xz = in_proj(x)
x_branch, z_gate = split(xz)

x_conv = depthwise_conv1d(x_branch)
x_conv = SiLU(x_conv)

y = SSMCore(x_conv)

y = y * SiLU(z_gate)

out = out_proj(y)

return dropout(out) + residual
```

---

# 8. SSM Core — Selective State Space

## 8.1 Continuous-to-Discrete Idea

Secara sederhana:

```txt
h(t) = A h(t-1) + B x(t)
y(t) = C h(t)
```

Pada implementasi Mamba-like:

- `A` adalah parameter state transition,
- `B` dan `C` dihasilkan dari input,
- `delta` menjadi step size yang juga input-dependent.

---

## 8.2 Selective Scan

```python
A = -exp(A_log)

x_proj = x_proj(x)
delta, B_mat, C_mat = split(x_proj)

delta = softplus(dt_proj(delta))

dA = exp(delta * A)
dB = delta * B

for t in range(seq_len):
    h = dA[:, t] * h + dB[:, t] * x[:, t]
    y = (h * C[:, t]).sum(-1)
```

---

## 8.3 Kenapa Disebut Selective?

```txt
SSM klasik:
A, B, C fixed
→ semua input diproses hampir dengan aturan sama

Mamba-like SSM:
B, C, delta = f(x)
→ input memengaruhi bagaimana memori disimpan dan dibaca
```

Efeknya:

- lebih adaptif,
- lebih relevan untuk token penting,
- mampu menangkap dependensi kontekstual lebih baik dibanding SSM statis.

---

# 9. Konfigurasi Model

## 9.1 Target Config Besar

Target pengembangan model:

```python
SigerConfig(
    vocab_size  = 100271,
    d_model     = 512,
    n_layers    = 12,
    d_state     = 16,
    d_conv      = 4,
    expand      = 2,
    dt_rank     = "auto",
    dropout     = 0.1,
    max_seq_len = 2048,
)
```

---

## 9.2 Smoke Test Config

Saat testing awal CPU, digunakan config kecil:

```python
SigerConfig(
    vocab_size  = 100271,
    d_model     = 64,
    n_layers    = 2,
    d_state     = 16,
    d_conv      = 4,
    expand      = 2,
    dropout     = 0.1,
    max_seq_len = 32,
)
```

Config kecil ini sudah berhasil menjalankan:

- tokenizer,
- dataset chunking,
- model forward,
- optimizer,
- scheduler,
- checkpoint saving,
- full training loop.

Contoh hasil smoke test:

```txt
✅ Tokenizer ready | vocab_size=100271
📚 Dataset: 4 docs → 12 chunks
🖥️  Device: cpu
📊 Model params: 6,485,056 total | 6,485,056 trainable
🚀 Training starts | max_steps=20
💾 Saved checkpoint
🏆 Best model saved!
Training complete!
```

---

# 10. Jumlah Parameter

## 10.1 Target Model Config

Dengan target konfigurasi sekitar:

```txt
vocab_size = 100271
d_model    = 512
n_layers   = 12
expand     = 2
d_state    = 16
```

Estimasi parameter berada pada skala puluhan juta parameter.

Komponen terbesar biasanya berasal dari:

- embedding matrix,
- LM head yang di-weight tie dengan embedding,
- proyeksi pada SSM blocks.

---

## 10.2 Weight Tying

SIGERLM menggunakan weight tying:

```python
self.lm_head.weight = self.embedding.weight
```

Keuntungan:

- mengurangi jumlah parameter,
- membuat embedding dan output projection berbagi representasi,
- trik umum pada language model.

---

# 11. Base Training Architecture

## 11.1 Training Data

Base model menggunakan corpus teks umum:

```txt
data/raw/
├── indonesian.txt
├── english.txt
├── code.txt
└── corpus.txt
```

---

## 11.2 Base Training Flow

```txt
Raw Text Files
      │
      ▼
MultilingualTokenizer
      │
      ▼
Token IDs
      │
      ▼
TextDataset
sliding window chunking
      │
      ▼
DataLoader
      │
      ▼
SIGERLM
      │
      ▼
Cross Entropy Loss
      │
      ▼
AdamW Optimizer
      │
      ▼
Cosine LR Scheduler
      │
      ▼
CheckpointManager
```

---

## 11.3 TextDataset Chunking

Dataset dibagi menjadi sequence chunk dengan ukuran `max_seq_len`.

Contoh:

```txt
Token stream:
[1, 2, 3, 4, 5, 6, 7, 8, ...]

Jika max_seq_len = 32:
→ input  = tokens[0:32]
→ target = tokens[1:33]
```

Ini memungkinkan model belajar **next token prediction**.

---

## 11.4 Trainer Components

Trainer terdiri dari:

- `build_optimizer()`
- `CosineScheduler`
- `CheckpointManager`
- `TrainingLogger`
- gradient clipping
- gradient accumulation
- optional autocast saat CUDA tersedia

---

# 12. LoRA Fine-Tuning Architecture

Setelah base model tersedia, model bisa diadaptasi menggunakan **LoRA**.

LoRA tidak melatih ulang semua bobot model. Sebaliknya:

- base model dibekukan,
- beberapa linear layer diberi adapter matriks kecil `A × B`,
- hanya adapter yang dilatih.

---

## 12.1 LoRA Pipeline

```txt
Base Model (pretrained)
        │
        ▼
LoRAModel.inject()
freeze base model, tambah adapter A × B
        │
        ▼
InstructionDataset
UltraChat / Alpaca-ID / Lampung instruction dataset
        │
        ▼
Loss Masking
- system/user tokens = -100
- assistant tokens   = real token ids
        │
        ▼
LoRATrainer.train()
        │
        ▼
lora_step_*.pt
adapter checkpoint kecil
        │
        ▼
merge_and_export()
        │
        ▼
model_merged.pt
deployable merged model
        │
        ▼
ONNX Export + INT8 Quantization
        │
        ▼
FastAPI Deployment
```

---

## 12.2 LoRA Target Modules

LoRA ditujukan ke layer-layer proyeksi seperti:

```txt
in_proj
out_proj
x_proj
dt_proj
```

Tujuannya:

- memodifikasi perilaku model secara parameter-efficient,
- mempermudah instruction tuning,
- membuat translasi Lampung bisa ditambahkan tanpa full retraining.

---

## 12.3 Loss Masking pada Instruction Dataset

Instruction dataset hanya menghitung loss pada jawaban assistant.

```txt
<|system|> ... <|end_turn|>
<|user|>   ... <|end_turn|>
<|assistant|> ... <|end_turn|>
```

Label masking:

```txt
system tokens    → -100
user tokens      → -100
assistant tokens → actual token IDs
```

Tujuannya supaya model belajar:

- menjawab,
- bukan menyalin prompt user,
- bukan menghafal system prompt.

---

# 13. Arsitektur Dataset Lampung

SIGERLM memiliki pipeline khusus untuk membangun dataset translasi bahasa Lampung.

## 13.1 Sumber Dataset

Dataset Lampung dibangun dari kombinasi:

1. **Kamus Budaya Lampung–Indonesia Dialek O**
2. **Paper SMT Lampung Nyo → Indonesia**
3. **Struktur dataset multilingual ala NusaX**
4. **Manual validated sentence pairs**

---

## 13.2 Struktur Folder Dataset Lampung

```txt
data/lampung/
├── raw/
│   ├── kamus_lampung_o.pdf
│   ├── smt_lampung_nyo_paper.pdf
│   ├── manual_pairs.jsonl
│   └── smt_pairs.jsonl
│
├── processed/
│   ├── kamus_pairs.jsonl
│   ├── smt_paper_text.txt
│   └── nusax_preview/
│
└── final/
    ├── lampung_o_trilingual.jsonl
    ├── lampung_o_trilingual_normalized.jsonl
    ├── train.jsonl
    ├── valid.jsonl
    ├── test.jsonl
    ├── train_instruction.jsonl
    ├── valid_instruction.jsonl
    └── test_instruction.jsonl
```

---

## 13.3 Dataset Builder Pipeline

```txt
Kamus Lampung O PDF
        │
        ▼
extract_kamus_pdf.py
        │
        ▼
kamus_pairs.jsonl
        │
        ▼
normalize_text.py
        │
        ▼
normalized records
        │
        ▼
build_lampung_dataset.py
        │
        ▼
train / valid / test
        │
        ▼
build_instruction_dataset.py
        │
        ▼
instruction tuning dataset
```

---

## 13.4 SMT Paper Pipeline

```txt
SMT Lampung Nyo Paper PDF
        │
        ▼
extract_smt_paper.py
        │
        ▼
smt_paper_text.txt
        │
        ▼
manual sentence-pair curation
        │
        ▼
smt_pairs.jsonl
        │
        ▼
build_lampung_dataset.py
```

---

## 13.5 Parallel Dataset Record

Contoh format translation pair:

```json
{
  "dialect": "o",
  "lampung": "nyak haga mengan",
  "indonesian": "saya mau makan",
  "english": "i want to eat",
  "source": "manual",
  "type": "sentence_pair"
}
```

---

## 13.6 Instruction Dataset Record

Contoh format untuk LoRA fine-tuning:

```json
{
  "instruction": "Terjemahkan Lampung O ke Bahasa Indonesia",
  "input": "api kabar niku",
  "output": "apa kabar kamu"
}
```

```json
{
  "instruction": "Translate Lampung O to English",
  "input": "nyak haga mengan",
  "output": "i want to eat"
}
```

---

# 14. Inference Architecture

## 14.1 Generator

`Generator` melakukan autoregressive decoding:

```txt
Prompt
   │
   ▼
Tokenizer Encode
   │
   ▼
Model Forward
   │
   ▼
Take Last Logits
   │
   ▼
Sampler
   │
   ▼
Next Token
   │
   ▼
Append Token
   │
   ▼
Repeat
```

---

## 14.2 Sampling Pipeline

Sampler mendukung:

- greedy decoding,
- temperature,
- top-k,
- top-p / nucleus sampling,
- repetition penalty.

Pipeline:

```txt
logits
  │
  ▼
repetition penalty
  │
  ▼
temperature scaling
  │
  ▼
top-k filtering
  │
  ▼
top-p filtering
  │
  ▼
multinomial sampling
  │
  ▼
next token
```

---

## 14.3 Chat Session

`ChatSession` menyimpan percakapan:

```txt
System Prompt
    +
User / Assistant History
    +
Current User Message
    ↓
Prompt Builder
    ↓
Generator
    ↓
Assistant Response
```

Format chat:

```txt
<|system|>...<|end_turn|>
<|user|>...<|end_turn|>
<|assistant|>...<|end_turn|>
```

---

# 15. API Serving Architecture

FastAPI menyediakan endpoint:

```txt
GET    /health
POST   /generate
POST   /chat
DELETE /chat/{session_id}
```

---

## 15.1 Generate Endpoint

```txt
HTTP Request
    │
    ▼
Pydantic Validation
    │
    ▼
Generator.generate()
    │
    ▼
Response JSON
```

---

## 15.2 Streaming Endpoint

Jika `stream=true`:

```txt
Generator.stream()
    │
    ▼
Token-by-token yield
    │
    ▼
StreamingResponse
    │
    ▼
SSE client
```

---

# 16. Evaluation Architecture

Evaluation suite dirancang agar model bisa diuji dari beberapa sisi:

## 16.1 Perplexity

Mengukur kualitas language modeling:

```txt
lower PPL = better next-token prediction
```

---

## 16.2 Multiple Choice Benchmark

Mendukung:

- MMLU
- ARC

Cara evaluasi:

```txt
Question + Choices
      │
      ▼
Score log-prob tiap candidate answer
      │
      ▼
Pilih completion dengan score tertinggi
```

---

## 16.3 Indonesian-Specific Evaluation

Dirancang untuk benchmark bahasa Indonesia:

- sentiment,
- NLI,
- QA.

---

## 16.4 Generation Evaluation

Mengukur kualitas output teks:

- BLEU,
- ROUGE,
- diversity.

---

# 17. Optimization Architecture

## 17.1 Optimization Targets

Pipeline optimization diarahkan untuk deployment murah:

- CPU-only VPS,
- RAM kecil,
- latency rendah,
- model footprint lebih kecil.

---

## 17.2 Optimization Flow

```txt
Trained Model
      │
      ▼
Benchmark Baseline
      │
      ▼
ONNX Export
      │
      ▼
Quantization
INT8 / INT4
      │
      ▼
Optimized Runtime
      │
      ▼
FastAPI Serving
```

---

## 17.3 ONNX Export

ONNX export bertujuan:

- memindahkan graph model ke runtime yang lebih efisien,
- menurunkan overhead Python,
- memudahkan deployment CPU.

---

## 17.4 Quantization

Quantization menurunkan precision bobot:

```txt
FP32 → INT8 → INT4
```

Efek yang diharapkan:

- ukuran model turun,
- RAM lebih hemat,
- inference lebih cepat,
- kualitas bisa sedikit menurun tergantung skema quantization.

---

# 18. Dua Mode Operasi Model

## 18.1 Prefill Mode

Saat memproses prompt:

```txt
x: (B, L, d_model)
→ scan seluruh sequence
→ y: (B, L, d_model)
```

Digunakan untuk:

- membaca prompt awal,
- memproses input panjang.

---

## 18.2 Decode Mode

Saat generasi token:

```txt
x: (B, 1, d_model)
→ update state
→ next token
```

Dengan cache/state reuse, model tidak perlu menghitung ulang seluruh konteks dari nol.

---

# 19. Perbandingan Arsitektur

```txt
                SIGERLM        GPT-2 Small    LLaMA-7B
──────────────────────────────────────────────────────────
Params          custom          117M           7B
Architecture    SSM-like        Transformer    Transformer
Attention       Tidak utama     Ya             Ya
Sequence comp.  O(n) target     O(n²)          O(n²)
CPU-friendly    Lebih cocok     Terbatas       Berat
Fine-tuning     LoRA custom     Umum           Umum
Deployment      VPS target      Sedang         Berat
```

---

# 20. Design Principles

Project SIGERLM mengikuti prinsip:

1. **Modular**
   - model, tokenizer, training, inference, LoRA, evaluation dipisah.

2. **Readable**
   - code dibuat cukup eksplisit untuk dipelajari dan dimodifikasi.

3. **Experiment-friendly**
   - mudah mencoba dataset baru, training config baru, dan fine-tuning baru.

4. **CPU-conscious**
   - sejak awal mempertimbangkan deployment murah.

5. **Regional-language aware**
   - mendukung pengembangan dataset bahasa daerah, terutama Lampung.

---

# 21. Current Architecture Status

## Sudah Berjalan

- tokenizer initialization,
- dataset chunking,
- SIGERLM forward graph,
- SSM block + SSM core integration,
- full training loop,
- optimizer + scheduler,
- checkpoint saving,
- smoke training CPU,
- dataset folder architecture,
- Lampung dataset preprocessing tool structure.

---

## Sedang Dikembangkan

- dataset Lampung final dalam skala lebih besar,
- LoRA fine-tuning translasi Lampung,
- ONNX + quantization pipeline validation,
- API deployment final,
- benchmark aktual di VPS,
- evaluation report yang lebih matang.

---

# 22. Arsitektur Akhir yang Dituju

```txt
General Multilingual Corpus
        │
        ▼
Base SIGERLM Pretraining
        │
        ▼
Base Checkpoint
        │
        ├───────────────┐
        │               │
        ▼               ▼
General Chat LoRA   Lampung Translation LoRA
        │               │
        ▼               ▼
Merged Chat Model   Merged Lampung Translator
        │               │
        ▼               ▼
ONNX Export + Quantization
        │
        ▼
FastAPI Deployment on VPS
        │
        ▼
Lightweight Local AI Service
```

---

# 23. Referensi Konseptual

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces — Gu & Dao, 2023
- Efficiently Modeling Long Sequences with Structured State Spaces (S4)
- Language Modeling with Gated Convolutional Networks
- LoRA: Low-Rank Adaptation of Large Language Models
- NusaX: Multilingual dataset format reference for Indonesian regional language experiments