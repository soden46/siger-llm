# 🧠 MambaLM — Multilingual LLM dengan State Space Model

> LLM dari nol berbasis **Mamba (SSM)** — ringan, cepat, dan bisa jalan di CPU/VPS sekalipun.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

## 🎯 Overview

MambaLM adalah implementasi Large Language Model yang dibangun dari scratch menggunakan **State Space Model (SSM)** arsitektur Mamba, bukan Transformer. Keunggulan utama:

| Aspek | Transformer | MambaLM (SSM) |
|---|---|---|
| Complexity | O(n²) attention | O(n) linear |
| Memory | Boros (KV Cache besar) | Efisien |
| Long sequence | Lemah | Kuat |
| CPU inference | Lambat | Layak |
| Training | Butuh GPU besar | Bisa CPU |

**Target deployment:** VPS CPU-only (2 core, 4GB RAM) — seperti DewaCloud dari Dewaweb.

---

## ✨ Fitur

- ✅ **SSM/Mamba architecture** — linear complexity, efisien di sequence panjang
- ✅ **Multilingual tokenizer** — Tiktoken cl100k, support Indo + EN + Code
- ✅ **Full training pipeline** — cosine LR, gradient accumulation, checkpoint
- ✅ **LoRA fine-tuning** — parameter-efficient, hanya 0.77% params yang ditraining
- ✅ **Optimasi CPU** — INT8 quantization + ONNX Runtime (2-3x speedup)
- ✅ **KV/SSM Cache** — generasi token jauh lebih cepat
- ✅ **FastAPI endpoint** — REST API + SSE streaming
- ✅ **Evaluasi lengkap** — Perplexity, MMLU, ARC, IndoNLI, BLEU, ROUGE
- ✅ **Chat interface** — terminal CLI dengan session history

---

## 📁 Struktur Project

```
mamba-llm/
├── config/
│   └── model_config.py         # MambaConfig dataclass
│
├── model/
│   ├── embedding.py            # Token embedding
│   ├── ssm_core.py             # SSM state equation (A, B, C, D)
│   ├── ssm_block.py            # Full SSM block dengan gating
│   └── mamba_model.py          # Stack N blocks + LM head
│
├── tokenizer/
│   ├── tokenizer.py            # MultilingualTokenizer
│   ├── special_tokens.py       # Special token definitions
│   └── tests/
│
├── training/
│   ├── dataset.py              # TextDataset sliding window
│   ├── trainer.py              # Main training loop
│   ├── optimizer.py            # AdamW + CosineScheduler
│   ├── checkpoint.py           # Save/load checkpoints
│   └── logger.py               # Training metrics logger
│
├── inference/
│   ├── generator.py            # Autoregressive generator
│   ├── sampler.py              # Top-K, Top-P, Temperature
│   ├── chat.py                 # ChatSession dengan history
│   └── api.py                  # FastAPI endpoints
│
├── lora/
│   ├── config.py               # LoRAConfig
│   ├── layer.py                # LoRALinear layer
│   ├── model.py                # LoRAModel wrapper
│   ├── dataset.py              # InstructionDataset
│   ├── trainer.py              # LoRA training loop
│   └── merge.py                # Merge LoRA → base model
│
├── optimization/
│   ├── quantization/
│   │   ├── quantize.py         # INT8/INT4 quantization
│   │   └── calibrate.py        # Calibration dataset
│   ├── onnx/
│   │   └── export.py           # ONNX export + session
│   ├── cpu/
│   │   ├── threading.py        # CPU thread config
│   │   └── memory.py           # RAM management
│   └── benchmark.py            # Speed & RAM benchmark
│
├── evaluation/
│   ├── perplexity.py           # PPL evaluator
│   ├── benchmarks.py           # MMLU, ARC multiple choice
│   ├── generation.py           # BLEU, ROUGE, Diversity
│   ├── indo_eval.py            # Indo-specific benchmarks
│   ├── runner.py               # Run all evals sekaligus
│   └── report.py               # Generate eval report
│
├── checkpoints/                # Model weights (git-ignored)
├── main.py                     # Entry point training
├── run_chat.py                 # Terminal chat interface
├── run_eval.py                 # Run evaluation suite
├── run_lora.py                 # LoRA fine-tuning
├── deploy.py                   # Deploy ke VPS
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
├── TRAINING.md
├── INFERENCE.md
├── LORA.md
├── OPTIMIZATION.md
├── EVALUATION.md
├── INSTALLATION.md
└── AGENTS.md
```

---

## ⚡ Quick Start

```bash
# 1. Clone & install
git clone https://github.com/yourname/mamba-llm.git
cd mamba-llm
pip install -r requirements.txt

# 2. Train model kecil (test)
python main.py

# 3. Chat di terminal
python run_chat.py

# 4. Jalanin API
python deploy.py
# → http://localhost:8000/docs
```

---

## 📊 Performance di VPS (2 core CPU, 4GB RAM)

| Mode | Model Size | RAM | Speed |
|---|---|---|---|
| Raw FP32 | ~2.1 GB | ~3.8 GB ⚠️ | ~1-2 tok/s |
| INT8 + ONNX | ~550 MB | ~1.5 GB ✅ | ~8-15 tok/s |
| INT4 + ONNX | ~280 MB | ~1.0 GB ✅ | ~12-20 tok/s |

---

## 📖 Dokumentasi Lengkap

| Dokumen | Deskripsi |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arsitektur SSM/Mamba detail |
| [INSTALLATION.md](INSTALLATION.md) | Panduan instalasi step-by-step |
| [TRAINING.md](TRAINING.md) | Training pipeline & hyperparameter |
| [INFERENCE.md](INFERENCE.md) | Inference, sampling, chat API |
| [LORA.md](LORA.md) | LoRA fine-tuning guide |
| [OPTIMIZATION.md](OPTIMIZATION.md) | Quantization & ONNX optimization |
| [EVALUATION.md](EVALUATION.md) | Evaluation metrics & benchmark |
| [AGENTS.md](AGENTS.md) | Panduan untuk AI agent/LLM tools |

---

## 🛠️ Tech Stack

- **PyTorch** — deep learning framework
- **Tiktoken** — BPE tokenizer (OpenAI cl100k_base)
- **ONNX Runtime** — optimized CPU inference
- **FastAPI** — REST API server
- **HuggingFace Datasets** — dataset loading
- **bitsandbytes** — quantization

---

## 📄 License

MIT License — bebas dipakai, dimodifikasi, dan didistribusikan.