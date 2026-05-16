# 🧠 SigerLM — Multilingual LLM dengan State Space Model

> LLM dari nol berbasis **Mamba (SSM)** — ringan, cepat, dan bisa jalan di CPU/VPS sekalipun.

---

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

---

## 🎯 Overview

SigerLM adalah implementasi Large Language Model yang dibangun dari scratch menggunakan **State Space Model (SSM)** arsitektur Mamba, bukan Transformer.

Keunggulan utamanya adalah efisiensi komputasi dan memori yang lebih cocok untuk eksperimen model kecil, long-sequence processing, dan target deployment ringan seperti CPU/VPS.

| Aspek | Transformer | SigerLM (SSM) |
|---|---|---|
| Complexity | O(n²) attention | O(n) linear |
| Memory | Boros, KV cache besar | Lebih efisien |
| Long sequence | Semakin berat | Lebih scalable |
| CPU inference | Cenderung lambat | Lebih layak |
| Target deployment | GPU-heavy | CPU/VPS friendly |

**Target deployment:** VPS CPU-only, misalnya 2 core CPU dan 4GB RAM.

---

## ✨ Fitur

- ✅ **SSM/Mamba architecture** — linear complexity, efisien untuk sequence panjang
- ✅ **Multilingual tokenizer** — Tiktoken `cl100k_base`, support Indonesian + English + Code
- ✅ **Full training pipeline** — scheduler, gradient accumulation, checkpoint, logging
- ✅ **LoRA fine-tuning** — parameter-efficient adaptation untuk instruction tuning
- ✅ **Optimasi CPU** — ONNX Runtime + quantization pipeline
- ✅ **KV/SSM cache exploration** — optimasi generasi token
- ✅ **FastAPI endpoint** — REST API + streaming response
- ✅ **Evaluasi lengkap** — Perplexity, MMLU, ARC, Indo benchmark, BLEU, ROUGE
- ✅ **Chat interface** — terminal chat dengan session history
- ✅ **Lampung language dataset pipeline** — Dialek O/Nyo ↔ Indonesia ↔ English
- ✅ **Regional language instruction tuning**
- ✅ **PDF extraction tools** — parsing kamus dan paper sebagai bahan dataset
- ✅ **Lampung LoRA fine-tuning berhasil berjalan** — 1.770 instruction examples tervalidasi
- ✅ **LoRA adapter sangat ringan** — hanya ±0.1 MB untuk eksperimen translasi awal
- ✅ **Merged Lampung model checkpoint** — siap dipakai untuk tahap inference dan evaluasi berikutnya

---

## 🌏 Lampung Language Dataset Pipeline

SigerLM dikembangkan untuk mendukung eksperimen multilingual dan pelestarian bahasa daerah, khususnya:

- Lampung Dialek O
- Lampung Dialek Nyo
- Bahasa Indonesia
- English

Pipeline dataset Lampung dibangun dari beberapa sumber:

1. **Kamus Budaya Lampung–Indonesia Dialek O**
2. **Paper SMT Lampung Nyo → Indonesia**
3. **Rajotuho Bahasa Lampung article scraper**
4. **Manual validated translation pairs**
5. **Format multilingual parallel corpus ala NusaX sebagai referensi struktur**

Tujuannya adalah membuat dataset translasi yang bisa dipakai untuk:

- Lampung O → Indonesia
- Indonesia → Lampung O
- Lampung O → English
- English → Lampung O
- Lampung Nyo → Indonesia
- Indonesia → Lampung Nyo

---

## 📁 Dataset Structure

```txt
data/
├── raw/
│   ├── code.txt
│   ├── corpus.txt
│   ├── english.txt
│   └── indonesian.txt
│
└── lampung/
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

## 🔄 Lampung Dataset Pipeline

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
normalized dataset
        │
        ▼
build_lampung_dataset.py
        │
        ▼
train.jsonl / valid.jsonl / test.jsonl
        │
        ▼
build_instruction_dataset.py
        │
        ▼
train_instruction.jsonl
        │
        ▼
LoRA Fine-Tuning
```

Untuk paper SMT:

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
manual curation sentence pairs
        │
        ▼
smt_pairs.jsonl
        │
        ▼
build_lampung_dataset.py
```

---

## 🛠️ Dataset Tools

| Tool | Fungsi |
|---|---|
| `tools/extract_kamus_pdf.py` | Extract vocabulary dan entri awal dari PDF Kamus Lampung O |
| `tools/extract_smt_paper.py` | Extract isi PDF paper SMT Lampung Nyo |
| `tools/inspect_nusax_format.py` | Melihat struktur dataset multilingual NusaX sebagai referensi |
| `tools/normalize_text.py` | Membersihkan dan menormalkan teks dataset |
| `tools/build_lampung_dataset.py` | Menggabungkan dataset, deduplicate, dan membuat split |
| `tools/build_instruction_dataset.py` | Mengubah dataset translasi menjadi instruction tuning format |
| `tools/scrape_rajotuho.py` | Scrape artikel Bahasa Lampung dari Rajotuho menjadi pasangan Dialek O ↔ Indonesia |

---

## 📄 Contoh Translation Pair

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

## 📄 Contoh Instruction Tuning Format

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

## 🚀 Menjalankan Dataset Pipeline

```bash
python tools/extract_kamus_pdf.py
python tools/extract_smt_paper.py
python tools/inspect_nusax_format.py
python tools/normalize_text.py
python tools/build_lampung_dataset.py
python tools/build_instruction_dataset.py
```

File utama yang nantinya dipakai untuk LoRA translation training:

```txt
data/lampung/final/train_instruction.jsonl
```

---

## ✅ Progress Terbaru: LoRA Translasi Lampung Berhasil Dilatih

Pipeline fine-tuning translasi Bahasa Lampung sudah berhasil dijalankan secara end-to-end.

### Ringkasan Hasil Terbaru

```txt
Base Model:
- vocab_size : 100271
- d_model    : 64
- n_layers   : 2
- params     : 6,485,056

LoRA:
- LoRA layers : 8
- LoRA params : 13,056
- Persentase  : 0.20% dari base model
- Adapter size: ±0.1 MB
```

## ✅ Progress Terbaru: Rajotuho Scraper & LoRA Retraining

Pipeline dataset Lampung berhasil diperluas menggunakan scraper khusus untuk kategori artikel Bahasa Lampung dari **Rajotuho**.

### Rajotuho Scraper

Scraper baru:

```txt
tools/scrape_rajotuho.py
```

berfungsi untuk:

- mengambil artikel kategori Bahasa Lampung,
- mendeteksi konten Dialek O,
- mengekstrak pasangan Lampung O ↔ Indonesia,
- memfilter pola yang mengandung tag Dialek O/A,
- menyimpan hasil ke JSONL.

Output utama:

```txt
data/lampung/raw/rajotuho_pairs.jsonl
data/lampung/processed/rajotuho_scrape_report.json
```

Scraper berhasil menambahkan ratusan pasangan baru yang lebih natural, termasuk:

- kosakata tematik,
- contoh penggunaan dalam kalimat,
- ekspresi sehari-hari,
- pola percakapan dan pembelajaran bahasa.

### Dataset Setelah Ekspansi

Setelah dataset Rajotuho dimasukkan ke pipeline dan dibersihkan:

```txt
LoRA instruction training input:
- Raw examples  : 2,444
- Valid examples: 2,435
```

Dataset ini lebih kaya dibanding tahap sebelumnya karena tidak lagi hanya didominasi entri kamus, tetapi mulai mencakup **contoh kalimat natural dan konteks penggunaan sehari-hari**.

### LoRA Retraining Terbaru

LoRA fine-tuning berhasil dijalankan ulang menggunakan dataset Lampung yang telah diperluas.

```txt
Base model:
- vocab_size : 100271
- d_model    : 64
- n_layers   : 2

LoRA:
- adapter params : 13,056
- adapter ratio  : 0.20%
- max steps      : 300
- effective batch: 8
- final loss     : 1.5450
```

Output checkpoint terbaru:

```txt
checkpoints/lora/
├── lora_step_000300.pt
└── model_lampung_merged.pt
```

Pipeline yang telah terbukti berjalan:

```txt
Rajotuho Article Scraper
        │
        ▼
rajotuho_pairs.jsonl
        │
        ▼
Lampung Dataset Builder
        │
        ▼
Instruction Dataset Builder
        │
        ▼
LoRA Fine-Tuning
        │
        ▼
Merged Lampung Checkpoint
```

### Dataset Lampung yang Digunakan

Dataset hasil pipeline saat ini:

```txt
Dataset final:
- Total records : 1,086

Instruction tuning:
- Train rows    : 1,778
- Valid rows    : 226
- Test rows     : 228
```

Saat masuk ke `InstructionDataset`, hasil validasi tokenisasi dan loss masking:

```txt
Raw training examples : 1,778
Valid examples        : 1,770
```

### Hasil LoRA Training

LoRA fine-tuning berhasil dijalankan selama:

```txt
Max steps : 300
Batch     : 2
Grad accum: 4
Eff. batch: 8
Final loss: 1.5971
```

Output checkpoint yang berhasil dibuat:

```txt
checkpoints/lora/
├── lora_step_000300.pt
└── model_lampung_merged.pt
```

### Pipeline yang Sudah Terbukti Berjalan

```txt
Base SigerLM Checkpoint
        │
        ▼
Auto Infer Model Config
        │
        ▼
Inject LoRA Adapters
        │
        ▼
Lampung Instruction Dataset
        │
        ▼
LoRA Training
        │
        ▼
LoRA Adapter Saved
        │
        ▼
Merge Adapter into Base Model
        │
        ▼
model_lampung_merged.pt
```

### Catatan Kualitas Model

Model translasi Lampung ini **sudah berhasil dilatih secara teknis**, tetapi masih berada pada tahap eksperimen awal.

Dataset saat ini masih didominasi oleh:

- entri kamus Lampung Dialek O,
- 25 sentence pair dari paper SMT Lampung Nyo,
- beberapa pasangan manual,
- dataset trilingual awal.

Artinya, model saat ini lebih kuat untuk mulai belajar:

- kosakata Lampung,
- relasi Lampung ↔ Indonesia,
- pola instruksi translasi dasar,

tetapi **belum cukup matang untuk translasi percakapan alami yang kompleks**.

Tahap pengembangan berikutnya adalah memperbanyak **parallel sentence dataset** dan **percakapan sehari-hari Lampung Dialek O**.

---

---

## 🎯 Tujuan Dataset Lampung

- Melestarikan bahasa daerah Lampung melalui dataset digital
- Membuat corpus translasi Lampung O/Nyo ↔ Indonesia ↔ English
- Mendukung instruction tuning bahasa lokal
- Membangun AI lokal yang ringan dan dapat berjalan di CPU/VPS
- Membuka peluang aplikasi edukasi, budaya, kamus digital, dan translator lokal

---

## 📁 Struktur Project

```txt
siger-llm/
├── config/
│   └── model_config.py         # SigerConfig dataclass
│
├── model/
│   ├── ssm_core.py             # SSM state equation (A, B, C, D)
│   ├── ssm_block.py            # Full SSM block dengan gating
│   └── siger_model.py          # Stack N blocks + LM head
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
│   ├── run_lora.py             # Run LoRA fine-tuning
│   └── merge.py                # Merge LoRA → base model
│
├── optimization/
│   ├── quantization/
│   │   ├── quantize.py         # INT8/INT4 quantization
│   │   └── calibrate.py        # Calibration dataset
│   ├── onnx/
│   │   └── export.py           # ONNX export + session
│   ├── cpu/
│   │   └── threading.py        # CPU thread config
│   ├── kvcache.py              # KV/SSM cache experiment
│   └── benchmark.py            # Speed & RAM benchmark
│
├── evaluation/
│   ├── perplexity.py           # PPL evaluator
│   ├── benchmarks.py           # MMLU, ARC multiple choice
│   ├── generation.py           # BLEU, ROUGE, Diversity
│   ├── indo_eval.py            # Indo-specific benchmarks
│   ├── runner.py               # Run all evals sekaligus
│   ├── run_eval.py             # Evaluation entry point
│   └── report.py               # Generate eval report
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

## ⚡ Quick Start

```bash
git clone https://github.com/soden46/siger-llm.git
cd siger-llm
```

### Buat virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Jalankan smoke training

```bash
python main.py
```

---

## 🧪 Smoke Test Training

Untuk testing pipeline awal di CPU, gunakan konfigurasi kecil seperti:

```python
TRAIN_CONFIG = {
    "vocab_size": 100271,
    "d_model": 64,
    "n_layers": 2,
    "max_steps": 20,
    "batch_size": 2,
    "max_seq_len": 32,
    "grad_accum_steps": 1,
}
```

Contoh output ketika training pipeline sukses:

```txt
✅ Tokenizer ready | vocab_size=100271
📚 Dataset: 4 docs → 12 chunks
🖥️  Device: cpu
📊 Model params: 6,485,056 total | 6,485,056 trainable

🚀 Training starts | max_steps=20
step=0 ...
💾 Saved checkpoint
🏆 Best model saved!
Training complete!
```

---

## 🧠 Training Base Model

Model dasar dilatih dari teks umum multilingual:

```txt
data/raw/
├── indonesian.txt
├── english.txt
├── code.txt
└── corpus.txt
```

Training flow:

```txt
Raw Text Corpus
        │
        ▼
MultilingualTokenizer
        │
        ▼
TextDataset sliding windows
        │
        ▼
SigerLM forward pass
        │
        ▼
Cross Entropy Loss
        │
        ▼
AdamW + Cosine Scheduler
        │
        ▼
Checkpoint Saving
```

Run:

```bash
python main.py
```

---

## 🔧 LoRA Fine-Tuning Pipeline

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
loss mask:
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
deployable merged checkpoint
        │
        ▼
ONNX Export + INT8 Quantization
        │
        ▼
FastAPI deployment on VPS
```

---

## 💬 Inference & API

SigerLM menyediakan:

- text generation
- chat session
- REST API
- streaming token response

Contoh API flow:

```txt
Prompt
   │
   ▼
Tokenizer Encode
   │
   ▼
SigerLM Forward
   │
   ▼
Sampler
   │
   ▼
Generated Token
   │
   ▼
Decoded Response
```

Contoh request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Apa itu kecerdasan buatan?",
    "max_new_tokens": 100,
    "temperature": 0.8
  }'
```

---

## 📊 Performance di VPS

Target pengembangan adalah inference ringan di VPS CPU-only.

| Mode | Model Size | RAM | Speed |
|---|---|---|---|
| Raw FP32 | ~2.1 GB | ~3.8 GB | ~1-2 tok/s |
| INT8 + ONNX | ~550 MB | ~1.5 GB | ~8-15 tok/s |
| INT4 + ONNX | ~280 MB | ~1.0 GB | ~12-20 tok/s |

> Angka ini adalah target/estimasi pengembangan dan perlu diverifikasi langsung lewat benchmark aktual pada deployment final.

---

## 📖 Dokumentasi Lengkap

| Dokumen | Deskripsi |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Arsitektur SSM/Mamba secara detail |
| [INSTALLATION.md](INSTALLATION.md) | Panduan instalasi step-by-step |
| [TRAINING.md](TRAINING.md) | Training pipeline & hyperparameter |
| [INFERENCE.md](INFERENCE.md) | Inference, sampling, chat API |
| [LORA.md](LORA.md) | LoRA fine-tuning guide |
| [OPTIMIZATION.md](OPTIMIZATION.md) | Quantization & ONNX optimization |
| [EVALUATION.md](EVALUATION.md) | Evaluation metrics & benchmark |
| [AGENTS.md](AGENTS.md) | Panduan untuk AI agent / LLM tools |
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Ringkasan konteks proyek untuk AI assistants |

---

## 🛠️ Tech Stack

- **PyTorch** — deep learning framework
- **Einops** — tensor operation helpers
- **Tiktoken** — BPE tokenizer berbasis `cl100k_base`
- **HuggingFace Datasets** — dataset loading
- **ONNX Runtime** — optimized CPU inference
- **FastAPI** — REST API server
- **Uvicorn** — ASGI server
- **PyMuPDF** — PDF text extraction
- **Pandas** — dataset inspection dan CSV processing
- **BeautifulSoup4** — HTML parsing/scraping
- **Requests** — HTTP request utility
- **bitsandbytes** — quantization experimentation
- **NLTK / Rouge Score** — evaluation metrics

---

## 🧭 Roadmap

- [x] Custom Mamba/SSM model skeleton
- [x] Tokenizer wrapper
- [x] Basic training pipeline
- [x] Checkpoint system
- [x] LoRA module structure
- [x] Evaluation module structure
- [x] Dataset pipeline folder structure
- [x] Lampung dataset processing tools
- [x] Extract Kamus Budaya Lampung Dialek O
- [x] Extract paper SMT Lampung Nyo
- [x] Build final Lampung train/valid/test dataset
- [x] Build Lampung instruction tuning dataset
- [x] LoRA translasi Lampung berhasil dilatih
- [x] Merge LoRA adapter ke base model
- [x] Rajotuho Bahasa Lampung scraper
- [x] Extract 400+ additional Lampung O translation candidates from web articles
- [x] Clean noisy Rajotuho extraction cases
- [x] Retrain Lampung LoRA using expanded instruction dataset
- [ ] Inference demo dari `model_lampung_merged.pt`
- [ ] Lampung O ↔ Indonesia translation evaluation
- [ ] Lampung O ↔ English translation evaluation
- [ ] Tambah 500+ kalimat percakapan harian Lampung Dialek O
- [ ] Tambah 1.000+ parallel sentence Lampung O ↔ Indonesia
- [ ] Native speaker validation workflow
- [ ] Dataset card dan model card
- [ ] Train base model lebih serius dengan corpus yang lebih besar
- [ ] LoRA ulang menggunakan base model yang lebih matang
- [ ] Quantized ONNX deployment
- [ ] FastAPI translation endpoint
- [ ] Web UI sederhana

---

## ⚠️ Project Status

Project ini masih **experimental**.

Beberapa bagian masih dalam tahap:

- implementasi awal
- compatibility fixing
- dataset building
- benchmark validation
- optimization exploration

Fokus utama saat ini:

1. Menstabilkan training pipeline
2. Membangun dataset Lampung yang rapi
3. Menjalankan LoRA fine-tuning untuk translasi
4. Menguji deployment CPU/VPS

---

## 🤝 Bantu Kembangkan Dataset Translasi Bahasa Lampung Dialek O

Project ini membutuhkan dataset yang lebih kaya dan beragam agar model translasi Bahasa Lampung Dialek O menjadi lebih natural, lebih akurat, dan lebih bermanfaat.

Saat ini dataset sudah berhasil dibangun dari:

- Kamus Budaya Lampung–Indonesia Dialek O
- Paper SMT Lampung Nyo → Indonesia
- pasangan manual awal
- format instruction tuning untuk LoRA

Namun, untuk menghasilkan translator Lampung Dialek O yang lebih baik, dataset perlu diperluas dengan **kalimat alami dan percakapan sehari-hari**.

### Dataset yang Sangat Dibutuhkan

Kontribusi dataset yang dicari:

- percakapan keluarga
- sapaan harian
- aktivitas rumah
- aktivitas sekolah
- jual beli di pasar
- bertanya arah
- cerita pendek
- ungkapan sopan
- dialog anak dan orang tua
- percakapan teman sebaya
- kalimat adat dan budaya Lampung
- konteks penggunaan kata yang tidak hanya berbentuk kamus

### Format Dataset yang Diinginkan

Contoh format `JSONL`:

```json
{"dialect":"o","lampung":"api kabar niku?","indonesian":"apa kabar kamu?","english":"how are you?","source":"manual_native_review","type":"daily_conversation"}
{"dialect":"o","lampung":"nyak haga lapah di pasar","indonesian":"saya mau pergi ke pasar","english":"i want to go to the market","source":"manual_native_review","type":"daily_conversation"}
{"dialect":"o","lampung":"niku ago mengan api?","indonesian":"kamu mau makan apa?","english":"what do you want to eat?","source":"manual_native_review","type":"daily_conversation"}
```

### Format Instruction Dataset

Data paralel tersebut nantinya akan dikonversi menjadi instruction tuning format seperti:

```json
{
  "instruction": "Terjemahkan Lampung O ke Bahasa Indonesia",
  "input": "api kabar niku?",
  "output": "apa kabar kamu?"
}
```

dan:

```json
{
  "instruction": "Translate Lampung O to English",
  "input": "nyak haga lapah di pasar",
  "output": "i want to go to the market"
}
```

### Tujuan Pengembangan Dataset Berikutnya

Target dataset tahap selanjutnya:

```txt
500+  kalimat percakapan harian tervalidasi
1.000+ parallel sentence Lampung O ↔ Indonesia
3.000+ pasangan translasi untuk LoRA tahap menengah
10.000+ pasangan untuk eksperimen translator yang lebih serius
```

### Ajakan Kontribusi

Bantuan sangat dibutuhkan untuk:

- menambahkan pasangan kalimat Lampung Dialek O,
- memvalidasi terjemahan Indonesia,
- menambahkan terjemahan English,
- menulis dialog sehari-hari yang natural,
- memperkaya konteks budaya dan penggunaan bahasa asli.

> **Bantu saya mengembangkan translasi Bahasa Lampung Dialek O ini dengan menambahkan dataset yang lebih beragam, terutama percakapan sehari-hari, kalimat natural, dan pasangan terjemahan Lampung–Indonesia–English yang tervalidasi.**

---

## 📄 License

MIT License — bebas dipakai, dimodifikasi, dan didistribusikan.