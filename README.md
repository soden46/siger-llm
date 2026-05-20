# SigerLM

> Multilingual LLM dari nol berbasis State Space Model/Mamba-like architecture: ringan, modular, dan ditargetkan bisa berjalan di CPU/VPS.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-experimental-orange)

## Overview

SigerLM adalah framework eksperimen Large Language Model yang dibangun dari scratch menggunakan pendekatan **State Space Model (SSM) / Mamba-like architecture**, bukan Transformer murni.

Tujuan utamanya adalah membangun LM general yang:

- ringan dan modular,
- bisa diuji di mesin kecil,
- ramah CPU/VPS,
- mendukung Indonesian, English, Code, dan domain bahasa daerah,
- dapat diadaptasi lewat LoRA dan retrieval/rule layer.

Bahasa Lampung dipakai sebagai domain adapter dan testbed training pertama, bukan sebagai batas kemampuan model. Core LM tetap general; logic Lampung ditempatkan di `retrieval/`, `inference/lampung_pipeline.py`, dan tools dataset.

| Aspek | Transformer | SigerLM (SSM/Mamba-like) |
|---|---|---|
| Sequence complexity | O(n^2) attention | Target O(n) linear scan |
| Memory | KV cache besar | State lebih ringkas |
| Long sequence | Semakin mahal | Lebih scalable |
| CPU inference | Cenderung berat | Lebih layak dieksplorasi |
| Target deployment | GPU-heavy | CPU/VPS friendly |

Target deployment awal: VPS CPU-only, misalnya 2 core CPU dan 4GB RAM.

## Open Collaboration

SigerLM adalah proyek open-source eksperimental dari Indonesia. Kontributor sangat dibutuhkan untuk:

- model architecture dan training,
- dataset dan validasi data,
- resource Bahasa Lampung O/Nyo,
- evaluation dan benchmarking,
- CPU inference dan optimization,
- dokumentasi, tutorial, dan tooling komunitas.

Kalau kamu tertarik dengan low-resource language AI, infrastruktur AI Indonesia, atau riset custom language model, kontribusi sangat terbuka.

## Fitur Utama

- Custom **SSM/Mamba-like language model** di PyTorch
- Hybrid tokenizer: HF ByteLevel BPE bila tersedia, fallback Tiktoken `cl100k_base`
- Base training pipeline: dataset, optimizer, scheduler, checkpoint, logging
- Custom LoRA fine-tuning untuk instruction tuning
- Config-driven LoRA runner untuk general dan Lampung adapters
- Adaptive training pipeline: Dense SSM -> hardware/metric-aware MoE -> LoRA
- Automatic LoRA curriculum runner: foundation -> general -> advanced -> final polish from one command
- Adaptive Sparse MoE sizing: jumlah expert, `top_k`, dan frekuensi layer MoE dipilih dari budget hardware dan loss checkpoint
- Anti-collapse MoE routing: load-balance loss, router importance penalty, jitter eksplorasi, dan metrik `moe_dead`
- Unified instruction corpus builder berbasis dataset registry
- Inference generator, chat session, router, dan Lampung domain pipeline
- Retrieval/rule layer untuk Lampung lookup dan compositional translation
- Dataset pipeline Lampung O/Nyo -> Indonesia -> English
- PDF extraction tools untuk kamus, paper SMT, dan dataset percakapan
- Evaluation dan optimization scaffolding: PPL, BLEU/ROUGE, ONNX, quantization, CPU benchmark
- FastAPI endpoint dan streaming response scaffolding

## Status

Project ini masih **experimental**. Arsitektur, dataset, dan training recipe masih berubah cepat.

Repository saat ini sudah berisi model core, pipeline data, workflow training, LoRA runner, inference router, retrieval/domain tooling, dan dokumentasi. Public model checkpoint dan benchmark final akan ditambahkan bertahap setelah training dan evaluasi makin matang.

## Documentation Map

| File | Purpose |
|---|---|
| [PROJECT_CONTEXT.md](PROJECT_CONTEXT.md) | Ringkasan konteks proyek untuk contributor dan AI assistant |
| [AGENTS.md](AGENTS.md) | Rules dan workflow notes untuk coding agents |
| [CLAUDE.md](CLAUDE.md) | Compact assistant-facing project guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution workflow, dataset format, dan commit style |
| [SECURITY.md](SECURITY.md) | Security policy dan private reporting |
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Setup, install, dan smoke checks |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arsitektur SSM/Mamba-like dan module boundaries |
| [docs/DATA_MINING.md](docs/DATA_MINING.md) | Mining dataset Q&A, instruction, dan Laravel menjadi instruction JSONL |
| [docs/RUN_COMMANDS.md](docs/RUN_COMMANDS.md) | Cheatsheet command setup, training, mining, dan testing |
| [DISTRIBUTED_TRAINING_ROADMAP.md](DISTRIBUTED_TRAINING_ROADMAP.md) | Future multi-GPU, cluster, and distributed training scaling roadmap |
| [docs/TRAINING.md](docs/TRAINING.md) | Base training, corpus builder, dan LoRA training flow |
| [docs/LORA.md](docs/LORA.md) | LoRA config, dataset formatting, dan merge workflow |
| [docs/MODALITY_AGNOSTIC_BACKBONE.md](docs/MODALITY_AGNOSTIC_BACKBONE.md) | Roadmap backbone modality-agnostic untuk text, vision, audio, video, sensor, graph, dan agent |
| [docs/INFERENCE.md](docs/INFERENCE.md) | Generator, chat, Lampung pipeline, dan router CLI |
| [docs/EVALUATION.md](docs/EVALUATION.md) | Current evaluation scope dan smoke results |
| [docs/OPTIMIZATION.md](docs/OPTIMIZATION.md) | CPU/VPS, ONNX, quantization, dan benchmark notes |

Suggested reading order:

```txt
README.md
  -> PROJECT_CONTEXT.md
  -> docs/INSTALLATION.md
  -> docs/ARCHITECTURE.md
  -> docs/TRAINING.md or docs/INFERENCE.md
```

## System Architecture

```txt
raw data / domain data
  -> tools/build_* or dataset registry
  -> unified instruction corpus
  -> tokenizer/hybrid_tokenizer.py
  -> base training / adaptive Dense -> MoE -> LoRA pipeline or LoRA curriculum
  -> merged checkpoint
  -> inference router
  -> general chat or Lampung domain tools
```

Core model boundary:

```txt
model/
  ssm_core.py
  ssm_block.py
  siger_model.py
```

Lampung-specific behavior belongs in:

```txt
data/lampung/
tools/build_lampung_dataset.py
tools/build_instruction_dataset.py
retrieval/
inference/lampung_pipeline.py
```

Routing between general chat and domain tools:

```txt
inference/router.py
chat_cli.py
```

## Project Structure

```txt
siger_llm/
├── config/          SigerConfig and config package guard
├── configs/         dataset registries and training configs
├── model/           SSM core, SSM block, and SigerLM
├── tokenizer/       hybrid tokenizer, HF BPE trainer, special tokens
├── training/        base training, dataset registry, optimizer, checkpoint
├── lora/            LoRA config, layers, wrapper, dataset, trainer, runner
├── inference/       generator, sampler, chat, router, Lampung pipeline, API
├── retrieval/       Lampung lookup, lexicon, compositional translator
├── tools/           dataset builders, extractors, normalization, scraper
├── data/            local corpora and generated datasets
├── docs/            architecture, training, inference, optimization docs
├── evaluation/      evaluation scaffolding
├── optimization/    ONNX, quantization, CPU/cache experiments
├── checkpoints/     local checkpoints
├── chat_cli.py      local CLI smoke tests
└── main.py          base training smoke entry point
```

## Lampung Language Dataset Pipeline

SigerLM dikembangkan untuk eksperimen multilingual dan pelestarian bahasa daerah, khususnya:

- Lampung Dialek O
- Lampung Dialek Nyo
- Bahasa Indonesia
- English

Dataset Lampung dibangun dari beberapa sumber:

1. Kamus Budaya Lampung-Indonesia Dialek O
2. Paper SMT Lampung Nyo -> Indonesia
3. Rajotuho Bahasa Lampung article scraper
4. Dataset percakapan Lampung Dialek O
5. Manual validated translation pairs
6. Synthetic compositional pairs
7. Struktur multilingual parallel corpus ala NusaX sebagai referensi

Target arah translasi:

- Lampung O -> Indonesia
- Indonesia -> Lampung O
- Lampung O -> English
- English -> Lampung O
- Lampung Nyo -> Indonesia
- Indonesia -> Lampung Nyo

## Dataset Structure

```txt
data/
├── indonesian.txt
├── english.txt
├── code.txt
├── corpus.txt
├── corpus/
│   ├── general_instruction_train.jsonl
│   └── lampung_instruction_train.jsonl
└── lampung/
    ├── raw/
    │   ├── kamus_lampung_o.pdf
    │   ├── smt_lampung_nyo_paper.pdf
    │   ├── dataset_1000_percakapan_lampung_dialek_o.pdf
    │   ├── manual_pairs.jsonl
    │   ├── smt_pairs.jsonl
    │   └── rajotuho_pairs.jsonl
    ├── processed/
    │   ├── kamus_pairs.jsonl
    │   ├── smt_paper_text.txt
    │   ├── percakapan_1000_pairs.jsonl
    │   ├── compositional_pairs.jsonl
    │   └── rajotuho_scrape_report.json
    └── final/
        ├── lampung_o_trilingual.jsonl
        ├── lampung_o_trilingual_normalized.jsonl
        ├── train.jsonl
        ├── valid.jsonl
        ├── test.jsonl
        ├── train_instruction.jsonl
        ├── valid_instruction.jsonl
        ├── test_instruction.jsonl
        ├── train_augmented_instruction.jsonl
        ├── valid_augmented_instruction.jsonl
        └── test_augmented_instruction.jsonl
```

## Dataset Pipeline

Lampung domain pipeline:

```txt
Kamus / SMT paper / Rajotuho / Percakapan PDF / Manual pairs
        │
        ▼
extract_*.py / scrape_rajotuho.py
        │
        ▼
data/lampung/processed/*.jsonl
        │
        ▼
normalize_text.py
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
*_instruction.jsonl / *_augmented_instruction.jsonl
        │
        ▼
build_instruction_corpus.py
        │
        ▼
data/corpus/lampung_instruction_train.jsonl
        │
        ▼
LoRA fine-tuning
```

Build Lampung domain data:

```powershell
python tools\extract_kamus_pdf.py
python tools\extract_smt_paper.py
python tools\extract_percakapan_pdf.py
python tools\scrape_rajotuho.py
python tools\build_compositional_lampung_dataset.py
python tools\build_lampung_dataset.py
python tools\build_instruction_dataset.py
```

Build unified instruction corpora:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage4_full.json
```

Mine Q&A Indonesia, general instruction, dan Laravel docs/tutorial:

```powershell
python tools\mine_general_assistant_data.py --preset all
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json
```

Dataset registry source formats:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`
- `mined_parallel_jsonl`

## Dataset Examples

Preferred translation pair:

```json
{"dialect":"o","lampung":"nyak haga mengan","indonesian":"saya mau makan","english":"i want to eat","source":"manual","type":"sentence_pair"}
```

Preferred instruction row:

```json
{"instruction":"Terjemahkan Lampung O ke Bahasa Indonesia","input":"api kabar niku","output":"apa kabar kamu","system":"optional system prompt","source":"manual","type":"translation"}
```

Preferred chat row for registry sources:

```json
{"messages":[{"role":"user","content":"Apa itu AI?"},{"role":"assistant","content":"AI adalah teknologi yang membuat komputer dapat melakukan tugas yang biasanya membutuhkan kecerdasan manusia."}]}
```

## Latest Verified Data State

```txt
Lampung processed PDF conversations: 3100 rows
Synthetic compositional rows: 1968
Final train/valid/test: 4325 / 541 / 541
Train rows with English field: 1605
Train augmented instruction: 32059 rows
General instruction corpus: 30704 rows
Lampung instruction corpus: 30701 rows
Kaggle local instruction rows: 29358 rows
Kaggle local corpus: 51969 rows
Curriculum stage1 foundation: 84605 rows
Curriculum stage2 general: 186672 rows
Curriculum stage3 advanced: 218596 rows
Curriculum stage4 full: 218596 rows
```

`general_instruction_train.jsonl` belum menjadi corpus general-chat yang kuat. Saat ini isinya masih banyak dipengaruhi data Lampung plus file lokal Indonesian/English/code kecil. Tambahkan sumber yang lebih besar ke `configs/datasets/general_instruction.json` sebelum mengharapkan behavior chatbot umum yang matang.

## Training

Adaptive Dense -> MoE -> LoRA pipeline:

```powershell
python train_pipeline.py --lora-config configs\training\general_lora.json
```

The pipeline starts from the dense SSM baseline, expands to Sparse MoE only after the dense checkpoint passes the configured loss/step gate, then trains LoRA after the MoE curve plateaus. During the MoE expansion stage, SigerLM chooses a conservative expert layout from the current hardware and the latest dense loss, so small CPU/VPS runs get fewer active experts while stronger CUDA runs can use more capacity.

Automatic easy-to-hard LoRA curriculum:

```powershell
python train_pipeline.py --mode lora-curriculum
```

This command reads `configs/training/lora_curriculum.json`, rebuilds each stage corpus, trains LoRA stage 1 through stage 4, skips stages with existing merged outputs, and writes logs to `logs/lora_curriculum/`.

Useful curriculum flags:

```powershell
python train_pipeline.py --mode lora-curriculum --dry-run
python train_pipeline.py --mode lora-curriculum --no-rebuild-corpora
python train_pipeline.py --mode lora-curriculum --force-curriculum
```

Lampung-only LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python lora\run_lora.py --config configs\training\lampung_lora.json
```

General LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python lora\run_lora.py --config configs\training\general_lora.json
```

Default fallback:

```powershell
python lora\run_lora.py
```

Default `python lora\run_lora.py` tetap Lampung-safe untuk backward compatibility.

Base model smoke:

```powershell
python -c "import torch; from config.model_config import SigerConfig; from model.siger_model import SigerLM; c=SigerConfig(vocab_size=1000,d_model=64,n_layers=2); m=SigerLM(c); x=torch.randint(0,1000,(2,32)); y,_=m(x); assert y.shape==(2,32,1000); print('ok')"
```

## LoRA Fine-Tuning Flow

```txt
Base SigerLM checkpoint
        │
        ▼
Auto infer model config
        │
        ▼
Inject LoRA adapters
freeze base model, train adapter A x B
        │
        ▼
InstructionDataset
assistant-only loss mask
        │
        ▼
LoRA training
        │
        ▼
lora_step_*.pt
        │
        ▼
merge adapter into base model
        │
        ▼
model_lampung_merged.pt / deployable merged checkpoint
```

Earlier Lampung LoRA milestones:

```txt
Base smoke config:
- vocab_size : 100271
- d_model    : 64
- n_layers   : 2
- params     : ~6.5M

LoRA:
- adapter params : 13,056
- adapter ratio  : ~0.20%
- adapter size   : ~0.1 MB
- max steps      : 300
- effective batch: 8
```

These numbers are experiment notes, not final benchmark claims.

## Inference CLI

```powershell
python chat_cli.py
```

CLI default sekarang menerima pertanyaan langsung. Router otomatis memilih general chat atau tool Lampung.

```txt
You: Nyak haga mengan manuk di warung paghek jalan
Assistant: aku mau makan ayam di warung dekat jalan
Route: lampung_to_id
Source: exact instruction lookup
```

Command opsional tetap tersedia untuk debug:

```txt
/help      tampilkan bantuan
/lo-id     Lampung O -> Indonesia
/id-lo     Indonesia -> Lampung O
/lo-en     Lampung O -> English
/reason    reasoning Lampung O -> Indonesia
/chat      paksa general chat
/reorder   susun kata Lampung O
```

Mode angka lama masih didukung: `0` auto, `1` LO->ID, `2` ID->LO, `3` LO->EN, `4` reasoning, `5` chat, `6` susun kata.

Lampung O -> English examples:

```txt
Nyak haga mengan manuk di warung paghek jalan
-> i want to eat chicken at the stall near the road

Nyak ago belei buku di pasar
-> i want to buy a book at the market
```

## Inference and API

Core generation flow:

```txt
Prompt
   │
   ▼
Tokenizer encode
   │
   ▼
SigerLM forward
   │
   ▼
Sampler
   │
   ▼
Generated token
   │
   ▼
Decoded response
```

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Apa itu kecerdasan buatan?",
    "max_new_tokens": 100,
    "temperature": 0.8
  }'
```

## CPU/VPS Optimization Target

Target pengembangan adalah inference ringan di VPS CPU-only.

| Mode | Model Size | RAM | Speed |
|---|---|---|---|
| Raw FP32 | target experiment | higher | baseline |
| INT8 + ONNX | smaller | lower | faster target |
| INT4 + ONNX | smallest | lowest | experimental |

Angka final harus diverifikasi lewat benchmark aktual pada checkpoint deployment, bukan dianggap klaim performa permanen.

## Development Checks

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py
python lora\run_lora.py --help
```

CLI smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```

## Roadmap

## Distributed Training and Hardware Scaling

SigerLM currently prioritizes correct single-device training, serious LoRA fine-tuning, and reproducible experiment workflows.
As the project matures, the architecture is planned to expand toward distributed and cluster-ready execution.

Planned hardware scaling directions include:

- distributed runtime abstraction for rank, world size, and device context
- `torchrun`-friendly training entrypoints
- single-node multi-GPU training with distributed data parallel execution
- distributed samplers, rank-aware logging, and rank-safe checkpointing
- globally correct validation metric aggregation
- multi-node cluster launch documentation
- optional SLURM templates for research clusters
- future investigation of FSDP, ZeRO-style scaling, activation checkpointing, and sharded checkpoints

The detailed roadmap is tracked in:

```txt
DISTRIBUTED_TRAINING_ROADMAP.md
```

### Data and Language

- Expand general instruction/chat corpus
- Add stronger Lampung ID/EN evaluation
- Add native speaker validation workflow for Lampung data
- Add 500+ validated daily conversation sentences for Lampung Dialek O
- Add 1,000+ Lampung O <-> Indonesia parallel sentences
- Build dataset card and model card

### Training and Model Development

- Use `train_pipeline.py` for reproducible Dense -> adaptive MoE -> LoRA experiments
- Use `train_pipeline.py --mode lora-curriculum` for one-command easy-to-hard LoRA runs
- Train and evaluate `general_lora.json`
- Improve base training recipes
- Train base model more seriously on larger corpus
- Re-run LoRA on a stronger base checkpoint
- Prepare reproducible experiment reports

### Engineering and Deployment

- Add automated tests for corpus builder and router
- Improve CPU inference performance
- Export and test ONNX/quantized checkpoints
- Add FastAPI endpoints for router/domain tasks
- Add simple web UI for demo translation/chat

## Bantu Kembangkan Dataset Translasi Bahasa Lampung Dialek O

Project ini membutuhkan dataset yang lebih kaya agar translator Lampung Dialek O menjadi lebih natural dan bermanfaat.

Kontribusi yang sangat dibutuhkan:

- percakapan keluarga,
- sapaan harian,
- aktivitas rumah dan sekolah,
- jual beli di pasar,
- bertanya arah,
- cerita pendek,
- ungkapan sopan,
- dialog anak dan orang tua,
- percakapan teman sebaya,
- kalimat adat dan budaya Lampung,
- konteks penggunaan kata yang tidak hanya berbentuk kamus.

Contoh format JSONL:

```json
{"dialect":"o","lampung":"api kabar niku?","indonesian":"apa kabar kamu?","english":"how are you?","source":"manual_native_review","type":"daily_conversation"}
{"dialect":"o","lampung":"nyak haga lapah di pasar","indonesian":"saya mau pergi ke pasar","english":"i want to go to the market","source":"manual_native_review","type":"daily_conversation"}
{"dialect":"o","lampung":"niku ago mengan api?","indonesian":"kamu mau makan apa?","english":"what do you want to eat?","source":"manual_native_review","type":"daily_conversation"}
```

Target dataset berikutnya:

```txt
500+   kalimat percakapan harian tervalidasi
1,000+ parallel sentence Lampung O <-> Indonesia
3,000+ pasangan translasi untuk LoRA tahap menengah
10,000+ pasangan untuk eksperimen translator yang lebih serius
```

## Tech Stack

- PyTorch
- Einops
- Tiktoken / HF ByteLevel BPE
- HuggingFace Datasets
- ONNX Runtime
- FastAPI / Uvicorn
- PyMuPDF
- Pandas
- BeautifulSoup4 / Requests
- NLTK / Rouge Score

## License

MIT License. Bebas dipakai, dimodifikasi, dan didistribusikan.
