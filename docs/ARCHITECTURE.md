# ARCHITECTURE.md - Arsitektur SigerLM

## Overview

**SigerLM** adalah proyek eksperimen Large Language Model yang dibangun dari nol menggunakan pendekatan **State Space Model (SSM) / Mamba-like architecture**, bukan Transformer murni.

Project ini dirancang untuk:

- membangun model bahasa ringan dari scratch,
- mendukung Indonesian, English, Code, dan domain bahasa daerah,
- dikembangkan menuju dukungan Bahasa Lampung Dialek O/Nyo,
- dapat dilatih, di-fine-tune, dievaluasi, dioptimasi, lalu dideploy ke CPU/VPS,
- menjaga core LM tetap general dan domain-neutral.

Arsitektur keseluruhan proyek tidak hanya mencakup model inti, tetapi juga tokenizer, dataset registry, training pipeline, LoRA fine-tuning, retrieval/domain pipeline Lampung, evaluation suite, ONNX/quantization optimization, dan FastAPI serving.

## 1. Kenapa SSM, Bukan Transformer?

Transformer punya kelemahan fundamental pada attention: **O(n^2)** terhadap panjang sequence.

Jika panjang sequence meningkat dua kali lipat, komputasi attention dapat meningkat sekitar empat kali lipat. SSM mencoba menyimpan konteks dalam state terkompresi sehingga biaya sequence bergerak menuju **O(n)**.

```txt
Transformer:
setiap token melihat semua token sebelumnya
-> semakin panjang sequence, semakin mahal komputasi dan memori

SSM / Mamba-like:
informasi masa lalu dipadatkan dalam hidden state h(t)
-> biaya pemrosesan lebih stabil terhadap panjang sequence
```

Target SigerLM bukan meniru Transformer besar, tetapi mengeksplorasi LM ringan yang lebih masuk akal untuk eksperimen lokal, CPU/VPS, dan low-resource language adaptation.

## 2. Gagasan Inti State Space Model

Jantung SSM adalah state update:

```txt
h(t) = A * h(t-1) + B * x(t)
y(t) = C * h(t)
```

Keterangan:

- `x(t)` adalah input token pada waktu ke-t
- `h(t)` adalah hidden state atau memori model
- `y(t)` adalah output pada waktu ke-t
- `A` adalah state transition matrix
- `B` adalah input projection
- `C` adalah output projection

Dalam pendekatan Mamba-like, `B`, `C`, dan `delta` dibuat input-dependent. Model belajar informasi mana yang perlu disimpan, dilupakan, atau dikeluarkan ke output. Ini disebut **selective state space modeling**.

## 3. System Layers

```txt
┌──────────────────────────────────────────────────────────────┐
│ Data Layer                                                   │
│ - Indonesian text                                            │
│ - English text                                               │
│ - Code snippets                                              │
│ - Lampung O/Nyo translation data                             │
│ - Instruction/chat JSONL                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Dataset Layer                                                │
│ - extraction tools                                           │
│ - domain builders                                            │
│ - dataset registry                                           │
│ - unified instruction corpus                                 │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Tokenizer Layer                                              │
│ - hybrid tokenizer selector                                  │
│ - optional HF ByteLevel BPE                                  │
│ - fallback Tiktoken cl100k_base                              │
│ - special/chat/language tokens                               │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Model Layer                                                  │
│ - token embedding                                            │
│ - N x SSMBlock                                               │
│ - final LayerNorm                                            │
│ - LM head                                                    │
│ - optional weight tying                                      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Training Layer                                               │
│ - base next-token training                                   │
│ - LoRA instruction tuning                                    │
│ - config-driven runs                                         │
│ - checkpoint and logging                                     │
│ - future distributed/runtime-aware  │execution                                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Inference Layer                                              │
│ - generator                                                  │
│ - sampler                                                    │
│ - chat session                                               │
│ - Lampung domain pipeline                                    │
│ - router                                                     │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ Evaluation / Optimization / Serving                          │
│ - PPL, BLEU, ROUGE, task eval scaffolds                      │
│ - CPU threading                                              │
│ - SSM cache experiments                                      │
│ - ONNX export                                                │
│ - INT8 / INT4 quantization                                   │
│ - FastAPI and streaming                                      │
└──────────────────────────────────────────────────────────────┘
```

## 4. Core Model Boundary

Core model harus tetap domain-neutral:

```txt
model/
  ssm_core.py
  ssm_block.py
  siger_model.py
```

Tidak ada lookup, translation rules, lexicon, atau logic Lampung yang boleh masuk ke `model/`.

Lampung-specific behavior berada di:

```txt
retrieval/
inference/lampung_pipeline.py
tools/build_lampung_dataset.py
tools/build_instruction_dataset.py
data/lampung/
```

Router domain berada di:

```txt
inference/router.py
chat_cli.py
```

## 5. Module Structure

```txt
siger_llm/
├── config/
│   └── model_config.py
├── configs/
│   ├── datasets/
│   └── training/
├── model/
│   ├── ssm_core.py
│   ├── ssm_block.py
│   └── siger_model.py
├── tokenizer/
│   ├── hybrid_tokenizer.py
│   ├── hf_tokenizer.py
│   ├── tokenizer.py
│   ├── special_tokens.py
│   ├── trainer.py
│   ├── train_hf_bpe.py
│   └── vocab_extender.py
├── training/
│   ├── dataset.py
│   ├── dataset_registry.py
│   ├── trainer.py
│   ├── optimizer.py
│   ├── checkpoint.py
│   └── logger.py
├── lora/
│   ├── config.py
│   ├── dataset.py
│   ├── layer.py
│   ├── model.py
│   ├── trainer.py
│   ├── run_lora.py
│   └── merge.py
├── inference/
│   ├── generator.py
│   ├── sampler.py
│   ├── chat.py
│   ├── prompt_builder.py
│   ├── router.py
│   ├── lampung_pipeline.py
│   └── api.py
├── retrieval/
│   ├── instruction_lookup.py
│   ├── compositional_translator.py
│   └── lampung_lexicon.py
├── tools/
│   ├── extract_kamus_pdf.py
│   ├── extract_smt_paper.py
│   ├── extract_percakapan_pdf.py
│   ├── scrape_rajotuho.py
│   ├── inspect_nusax_format.py
│   ├── normalize_text.py
│   ├── build_lampung_dataset.py
│   ├── build_compositional_lampung_dataset.py
│   ├── build_instruction_dataset.py
│   ├── build_instruction_corpus.py
│   └── mine_general_assistant_data.py
└── docs/
```

## 6. Tokenizer Architecture

SigerLM memakai `tokenizer/hybrid_tokenizer.py` sebagai selector. Jalur ideal adalah tokenizer HF ByteLevel BPE hasil training lokal. Jika tidak tersedia, sistem menggunakan fallback Tiktoken `cl100k_base` agar smoke test tetap bisa berjalan.

Special tokens digunakan untuk chat dan instruction tuning:

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

Tokenization flow:

```txt
Raw text
  -> tokenizer.encode()
  -> token IDs
  -> model input tensor
  -> generated token IDs
  -> tokenizer.decode()
  -> readable text
```

Special token IDs tidak boleh diubah sembarangan karena berpengaruh ke checkpoint, chat formatting, dan LoRA dataset masking.

## 7. Model Core: SigerLM

Public model identity is anchored to the immutable base name `SIGER`.
If an alias is provided through `SigerConfig(model_alias="soden")`, the canonical public name becomes:

```txt
SIGER-SODEN
```

Aliases cannot replace the base name; they are appended after `SIGER`.

End-to-end forward pass:

```txt
Input token IDs
      │
      ▼
Token Embedding
      │
      ▼
SSMBlock x N
      │
      ▼
Final LayerNorm
      │
      ▼
LM Head
      │
      ▼
Logits (B, L, vocab_size)
```

Setiap `SSMBlock` berisi:

1. `LayerNorm`
2. `in_proj`
3. split ke `x_branch` dan `z_gate`
4. depthwise `Conv1D`
5. `SSMCore`
6. gated multiplication
7. `out_proj`
8. residual connection

Pseudocode:

```python
residual = x
x = layer_norm(x)
xz = in_proj(x)
x_branch, z_gate = split(xz)
x_conv = depthwise_conv1d(x_branch)
x_conv = silu(x_conv)
y = ssm_core(x_conv)
y = y * silu(z_gate)
out = out_proj(y)
return dropout(out) + residual
```

## 8. SSM Core: Selective State Space

Konsep simplified:

```txt
A = -exp(A_log)
x_proj = projection(x)
delta, B, C = split(x_proj)
delta = softplus(dt_proj(delta))

for each timestep:
    h = dA * h + dB * x_t
    y = readout(h, C_t)
```

Disebut selective karena parameter pembacaan dan update state dipengaruhi input. Token penting bisa memengaruhi bagaimana memori disimpan dan dibaca.

## 9. Model Config

Config kecil untuk smoke CPU:

```python
SigerConfig(
    vocab_size=100271,
    d_model=64,
    n_layers=2,
    d_state=16,
    d_conv=4,
    expand=2,
    dropout=0.1,
    max_seq_len=32,
)
```

Target pengembangan yang lebih besar dapat dinaikkan bertahap:

```python
SigerConfig(
    vocab_size=100271,
    d_model=512,
    n_layers=12,
    d_state=16,
    d_conv=4,
    expand=2,
    dt_rank="auto",
    dropout=0.1,
    max_seq_len=2048,
)
```

Default model config jangan diubah tanpa mempertimbangkan backward compatibility checkpoint dan smoke tests.

## 10. Dataset Architecture

General registry flow:

```txt
HuggingFace / Kaggle local files / Laravel docs / SantriKoding
  -> tools/mine_general_assistant_data.py
  -> data/mined/instruction/*.jsonl
  -> configs/datasets/*.json
  -> training/dataset_registry.py
  -> tools/build_instruction_corpus.py
  -> data/corpus/*_instruction_train.jsonl
```

Supported registry source formats:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`

Preferred instruction row:

```json
{"instruction":"...","input":"...","output":"...","system":"optional system prompt","source":"...","type":"..."}
```

Preferred chat row:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## 11. Lampung Dataset Architecture

Lampung domain flow:

```txt
data/lampung/raw/
  -> tools/extract_*.py / scrape_rajotuho.py
  -> data/lampung/processed/
  -> tools/build_compositional_lampung_dataset.py
  -> tools/build_lampung_dataset.py
  -> data/lampung/final/train|valid|test.jsonl
  -> tools/build_instruction_dataset.py
  -> data/lampung/final/*_instruction.jsonl
  -> tools/build_instruction_corpus.py
  -> data/corpus/lampung_instruction_train.jsonl
```

Sumber dataset:

1. Kamus Budaya Lampung-Indonesia Dialek O
2. Paper SMT Lampung Nyo -> Indonesia
3. Rajotuho Bahasa Lampung article scraper
4. Percakapan Lampung Dialek O PDF
5. Manual validated pairs
6. Synthetic compositional pairs
7. Format multilingual parallel corpus ala NusaX sebagai referensi struktur

Contoh parallel record:

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

Contoh instruction record:

```json
{
  "instruction": "Terjemahkan Lampung O ke Bahasa Indonesia",
  "input": "api kabar niku",
  "output": "apa kabar kamu"
}
```

## 12. Base Training Architecture

Base training menggunakan next-token prediction:

```txt
Raw text files
  -> tokenizer
  -> token IDs
  -> TextDataset sliding windows
  -> DataLoader
  -> SigerLM
  -> Cross Entropy Loss
  -> AdamW
  -> Cosine LR scheduler
  -> CheckpointManager
```

TextDataset chunking:

```txt
Token stream:
[1, 2, 3, 4, 5, 6, 7, 8, ...]

Jika max_seq_len = 32:
input  = tokens[0:32]
target = tokens[1:33]
```

Trainer components:

- optimizer builder
- cosine scheduler
- checkpoint manager
- training logger
- gradient clipping
- gradient accumulation
- optional autocast saat CUDA tersedia

## 13. Distributed Training Direction

SigerLM is currently designed around reliable single-device development and experiment workflows.
The future training architecture is planned to remain backward-compatible with local execution while gradually supporting distributed scaling.

Target evolution:

```txt
single CPU / single GPU
  -> single-node multi-GPU execution
  -> multi-node cluster execution
  -> optional larger-model sharding strategies
```

## 14. LoRA Fine-Tuning Architecture

LoRA tidak melatih ulang semua bobot model. Base model dibekukan, lalu beberapa linear layer diberi adapter matriks kecil `A x B`.

LoRA flow:

```txt
Base Model checkpoint
        │
        ▼
LoRAModel.inject()
freeze base model, add adapter A x B
        │
        ▼
InstructionDataset
assistant-only loss mask
        │
        ▼
LoRATrainer.train()
        │
        ▼
lora_step_*.pt
        │
        ▼
merge_and_export()
        │
        ▼
merged checkpoint
```

Target modules biasanya layer proyeksi:

```txt
in_proj
out_proj
x_proj
dt_proj
```

Instruction loss masking:

```txt
<|system|> ... <|end_turn|>     -> label -100
<|user|> ... <|end_turn|>       -> label -100
<|assistant|> ... <|end_turn|>  -> actual token IDs
```

Tujuannya agar model belajar menjawab, bukan menyalin prompt user atau system prompt.

## 14. Inference Architecture

Generator melakukan autoregressive decoding:

```txt
Prompt
  -> tokenizer encode
  -> model forward
  -> take last logits
  -> sampler
  -> next token
  -> append token
  -> repeat
```

Sampler mendukung:

- greedy decoding
- temperature
- top-k
- top-p / nucleus sampling
- repetition penalty

Chat session:

```txt
System prompt
  + user/assistant history
  + current user message
  -> PromptBuilder
  -> Generator
  -> assistant response
```

Chat format:

```txt
<|system|>...<|end_turn|>
<|user|>...<|end_turn|>
<|assistant|>...<|end_turn|>
```

## 15. Lampung Inference Pipeline

Lampung pipeline memakai lookup-first approach karena model generatif belum dianggap cukup matang untuk semua translasi:

```txt
LampungPipeline
  -> InstructionLookup
  -> LampungCompositionalTranslator
  -> LampungLexicon
  -> Generator fallback
```

Router:

```txt
SigerRouter
  -> general_chat
  -> lampung_to_id
  -> id_to_lampung
  -> lampung_to_en
```

CLI default:

```txt
chat_cli.py
  user enters a direct question
  -> SigerRouter auto-detects general chat vs Lampung domain
```

Manual commands remain available for debugging: `/lo-id`, `/id-lo`, `/lo-en`, `/reason`, `/chat`, and `/reorder`. Legacy numeric modes `0` to `6` are still supported.

## 16. API Serving Architecture

FastAPI endpoint target:

```txt
GET    /health
POST   /generate
POST   /chat
DELETE /chat/{session_id}
```

Generate endpoint:

```txt
HTTP request
  -> Pydantic validation
  -> Generator.generate()
  -> response JSON
```

Streaming endpoint:

```txt
Generator.stream()
  -> token-by-token yield
  -> StreamingResponse
  -> SSE client
```

## 17. Evaluation Architecture

Evaluation suite diarahkan untuk mengukur:

- perplexity untuk next-token prediction,
- BLEU/ROUGE untuk translasi/generation,
- diversity untuk generation,
- Indonesian-specific eval scaffolding,
- Lampung ID/EN translation eval,
- MMLU/ARC style multiple-choice scaffolding.

Multiple-choice scoring:

```txt
Question + choices
  -> score log-prob each candidate answer
  -> choose highest-scoring completion
```

## 18. Optimization Architecture

Optimization diarahkan untuk deployment murah:

- CPU-only VPS,
- RAM kecil,
- latency rendah,
- model footprint lebih kecil.

Optimization flow:

```txt
Trained model
  -> benchmark baseline
  -> ONNX export
  -> quantization INT8 / INT4
  -> optimized runtime
  -> FastAPI serving
```

ONNX export bertujuan memindahkan graph ke runtime yang lebih efisien dan menurunkan overhead Python. Quantization menurunkan precision bobot:

```txt
FP32 -> INT8 -> INT4
```

Efek yang diharapkan:

- ukuran model turun,
- RAM lebih hemat,
- inference lebih cepat,
- kualitas bisa sedikit menurun tergantung skema quantization.

## 19. Prefill dan Decode Mode

Prefill mode:

```txt
x: (B, L, d_model)
-> scan seluruh sequence
-> y: (B, L, d_model)
```

Decode mode:

```txt
x: (B, 1, d_model)
-> update state
-> next token
```

Dengan cache/state reuse, model tidak perlu menghitung ulang seluruh konteks dari nol saat generasi token berikutnya.

## 20. Latest Verified State

```txt
Lampung processed PDF conversations: 3100 rows
Synthetic compositional rows: 1968
Final Lampung split: 4325 / 541 / 541
Train rows with English field: 1605
Train augmented instruction: 32059 rows
Lampung unified corpus: 30701 rows
General unified corpus: 30704 rows
```

Latest CLI smoke:

```txt
Input: Nyak haga mengan manuk di warung paghek jalan
Route: lampung_to_id
Source: exact instruction lookup
Output: aku mau makan ayam di warung dekat jalan
```

General corpus saat ini masih kecil di luar Lampung. Arsitektur sudah siap untuk general training, tetapi broad chatbot ability membutuhkan general instruction/chat data yang lebih besar dan lebih bersih.

## 21. Design Principles

1. **Modular**
   Model, tokenizer, training, inference, LoRA, evaluation, dan domain tools dipisah.

2. **Domain-neutral core**
   Core model tidak mengandung logic Lampung.

3. **Readable**
   Code dibuat eksplisit agar mudah dipelajari dan dimodifikasi.

4. **Experiment-friendly**
   Dataset, config training, dan adapter bisa diganti tanpa rewrite besar.

5. **CPU-conscious**
   Sejak awal mempertimbangkan mesin kecil dan deployment VPS.

6. **Regional-language aware**
   Mendukung pengembangan dataset bahasa daerah, terutama Lampung O/Nyo.

## 22. Target Architecture

```txt
General multilingual corpus
        │
        ▼
Base SigerLM pretraining
        │
        ▼
Base checkpoint
        │
        ├───────────────┐
        │               │
        ▼               ▼
General Chat LoRA   Lampung Translation LoRA
        │               │
        ▼               ▼
Merged Chat Model   Merged Lampung Translator
        │               │
        └───────┬───────┘
                ▼
ONNX export + quantization
                │
                ▼
FastAPI deployment on VPS
                │
                ▼
Lightweight local AI service
```

## 23. Referensi Konseptual

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces, Gu & Dao, 2023
- Efficiently Modeling Long Sequences with Structured State Spaces (S4)
- Language Modeling with Gated Convolutional Networks
- LoRA: Low-Rank Adaptation of Large Language Models
- NusaX: multilingual dataset format reference for Indonesian regional language experiments
