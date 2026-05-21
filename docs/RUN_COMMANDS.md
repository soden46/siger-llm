# RUN_COMMANDS.md - Command Training dan Test SigerLM

Dokumen ini adalah cheatsheet command untuk setup, mining data, training, dan smoke test.

## 1. Setup Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

CPU-only PyTorch alternatif:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 2. Compile Check

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py train_pipeline.py
python -m py_compile tools\mine_general_assistant_data.py
python -m py_compile tools\mine_indonesian_hf_mix.py
```

## 3. Model Forward Smoke

```powershell
python -c "import torch; from config.model_config import SigerConfig; from model.siger_model import SigerLM; c=SigerConfig(vocab_size=1000,d_model=64,n_layers=2); m=SigerLM(c); x=torch.randint(0,1000,(2,32)); y,_=m(x); assert y.shape==(2,32,1000); print('ok')"
```

## 4. Base Training Smoke

`main.py` menjalankan training kecil dari file `.txt` di `data/`.

```powershell
python main.py
```

Jika runtime punya lebih dari satu GPU CUDA, `main.py` otomatis re-launch lewat `torchrun` dan memakai DDP.
Log yang diharapkan:

```txt
Auto-launching DDP with torchrun on 2 GPUs...
Runtime plan
  strategy  : ddp
  devices   : 2
  precision : fp16
```

`main.py` juga mengaktifkan AMP otomatis, dataloader auto worker, dan VRAM-aware batch tuning secara konservatif. Kalau ingin mematikan auto-DDP dan memakai fallback single-process:

```bash
SIGER_DISABLE_AUTO_DDP=1 python main.py
```

Base training text sources bisa diatur tanpa edit kode:

```bash
SIGER_TEXT_SOURCES=data/indonesian_hf_mix.txt,data/kaggle/kaggle_extra_text.txt python main.py
SIGER_TEXT_SOURCES=data SIGER_TEXT_INCLUDE=*.txt SIGER_MAX_TEXT_FILES=8 python main.py
```

`SIGER_TEXT_SOURCES` menerima file, folder, atau glob yang dipisahkan koma/semicolon. Default tetap `data` dengan include `*.txt`.

Pilih kapasitas model base yang lebih kuat untuk reasoning jika VRAM/waktu cukup:

```bash
SIGER_MODEL_PROFILE=small_context python main.py
SIGER_MODEL_PROFILE=moe_dense_base python main.py
SIGER_MODEL_PROFILE=base python main.py
SIGER_MODEL_PROFILE=reasoning_base python main.py
```

Profile default tetap `small` (`d_model=256`, `n_layers=8`, `max_seq_len=128`) agar smoke test Kaggle lebih aman. Profile `moe_dense_base` (`d_model=384`, `n_layers=10`, `max_seq_len=512`) adalah dense base yang shape-compatible dengan `small_moe` untuk upcycling Dense -> MoE. Profile `base` memakai `d_model=512`, `n_layers=12`; profile `reasoning_base` menaikkan context training ke `max_seq_len=512`.

Adaptive dense -> MoE -> LoRA pipeline:

```powershell
python train_pipeline.py --mode auto --lora-config configs\training\general_lora.json
python train_pipeline.py --mode auto --dense-profile moe_dense_base --moe-profile small_moe
```

Automatic easy-to-hard LoRA curriculum:

```powershell
python train_pipeline.py --mode lora-curriculum
```

Kaggle/CUDA:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode lora-curriculum
```

Preview without training:

```powershell
python train_pipeline.py --mode auto --dry-run
python train_pipeline.py --mode lora-curriculum --dry-run
```

Default trigger:

```txt
dense profile: moe_dense_base
MoE profile  : small_moe
dense -> MoE: step >= 1500 and loss <= 3.5
MoE -> LoRA: latest checkpoint loss delta <= 0.005
```

Kaggle staged small -> medium -> high:

```bash
# 1) Small CPU-safe base training.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt SIGER_MODEL_PROFILE=small SIGER_RESUME=1 SIGER_SAVE_EVERY=50 SIGER_MAX_STEPS=1000 SIGER_DEVICE=cpu python main.py

# 2) Medium dense base compatible with small_moe.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt SIGER_MODEL_PROFILE=moe_dense_base SIGER_CHECKPOINT_DIR=checkpoints/auto/dense_moe_base SIGER_RESUME=1 SIGER_SAVE_EVERY=100 SIGER_MAX_STEPS=3000 SIGER_DEVICE=auto SIGER_PRECISION=auto PYTORCH_ALLOC_CONF=expandable_segments:True python main.py

# 3) High/capacity path: gated Dense -> MoE -> LoRA.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode auto --dense-profile moe_dense_base --moe-profile small_moe --dense-max-steps 3000 --moe-max-steps 5000
```

`SIGER_RESUME=1` resumes if possible. If `latest.json` points at a missing checkpoint, the loader falls back to the newest `step_*.pt`; if no checkpoint exists, it starts a new run.

Lampung adapter pipeline:

```powershell
python train_pipeline.py --lora-config configs\training\lampung_lora.json
```

Manual adaptive MoE base training:

```powershell
$env:SIGER_MODEL_PROFILE="small_moe"
python main.py
```

Exact static MoE fallback:

```powershell
$env:SIGER_DISABLE_ADAPTIVE_MOE="1"
$env:SIGER_MODEL_PROFILE="small_moe"
python main.py
```

Jika ingin menguji satu stage saja:

```powershell
python train_pipeline.py --force-stage dense
python train_pipeline.py --force-stage moe
python train_pipeline.py --force-stage lora --lora-config configs\training\general_lora.json
```

## 5. Build Dataset Lampung

```powershell
python tools\extract_kamus_pdf.py
python tools\extract_smt_paper.py
python tools\extract_percakapan_pdf.py
python tools\scrape_rajotuho.py
python tools\build_compositional_lampung_dataset.py
python tools\build_lampung_dataset.py
python tools\build_instruction_dataset.py
```

## 6. Build Unified Corpus

Lampung-only corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json --max-row-tokens 2048
```

General corpus lama:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json --max-row-tokens 2048
```

General assistant mining corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json --max-row-tokens 2048
```

Curriculum corpora:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage1_foundation.json --max-row-tokens 2048
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage2_general.json --max-row-tokens 2048
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage3_advanced.json --max-row-tokens 2048
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage4_full.json --max-row-tokens 2048
```

## 7. Mining Q&A, Reasoning, Code, dan Laravel

Smoke kecil dulu:

```powershell
python tools\mine_general_assistant_data.py --preset all --max-items 200 --max-laravel-pages 5 --max-santrikoding-articles 10
```

Commercial-safe reasoning saja:

```powershell
python tools\mine_general_assistant_data.py --preset reasoning --max-items 2000
```

Q&A Indonesia saja:

```powershell
python tools\mine_general_assistant_data.py --preset qa
```

Laravel docs/tutorial saja:

```powershell
python tools\mine_general_assistant_data.py --preset laravel --laravel-versions 9 10 11 12 13
```

Kaggle IndoNewsQA setelah dataset di-download lokal:

```powershell
python tools\mine_general_assistant_data.py --preset qa --local-qa-file data\external\indonewsqa\train.jsonl
```

Build corpus setelah mining:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json --max-row-tokens 2048
```

## 8. Mining Indonesian HF Mix di Kaggle

Install dependency HF jika environment belum punya:

```bash
pip install -q datasets
```

Mining source Indonesia campuran:

```bash
python tools/mine_indonesian_hf_mix.py --max-items-per-source 50000
```

Mining source Indonesia campuran dengan sebagian row dibuat Chain-of-Thought:

```bash
python tools/mine_indonesian_hf_mix.py --max-items-per-source 50000 --cot-ratio 0.25
```

`--cot-ratio 0.25` berarti sekitar 25% row diberi format `<thought>...</thought>` secara deterministik. Hindari langsung 100% untuk dataset besar agar model tetap bisa menjawab ringkas saat user tidak meminta penalaran.

Ingest dataset dari Kaggle Add Input:

```bash
python tools/ingest_kaggle_inputs.py
```

The ingest step creates:

```txt
data/kaggle/kaggle_extra_instruction.jsonl
data/kaggle/kaggle_extra_text.txt
configs/datasets/kaggle_local_inputs.json
```

Build corpus instruction:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix.json --max-row-tokens 2048
```

Build corpus instruction dengan CoT conversion dari registry apa pun:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix.json --cot-ratio 0.25 --max-row-tokens 2048
```

Build corpus HF mix + Kaggle Add Input:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle.json --max-row-tokens 2048
```

Build software engineering capability seed:

```bash
python tools/build_software_engineering_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/software_engineering_instruction.json --max-row-tokens 2048
```

Build reasoning / Chain-of-Thought seed:

```bash
python tools/build_reasoning_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/reasoning_instruction.json --max-row-tokens 2048
```

Build HF mix + Kaggle + Lampung + software engineering capability:

```bash
python tools/build_software_engineering_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle_software.json --max-row-tokens 2048
```

Build HF mix + Kaggle + Lampung + reasoning + software engineering:

```bash
python tools/build_software_engineering_seed.py
python tools/build_reasoning_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 2048
```

Output penting:

```txt
data/mined/hf_indonesia/indonesian_hf_mix_instruction.jsonl
data/corpus/indonesian_hf_mix_train.jsonl
data/indonesian_hf_mix.txt
data/kaggle/kaggle_extra_text.txt
data/kaggle/kaggle_extra_instruction.jsonl
```

`data/indonesian_hf_mix.txt` otomatis bisa ikut base training karena `main.py` membaca semua file `.txt` di `data/`.

Quality report dibuat otomatis di samping corpus, misalnya:

```txt
data/corpus/indonesian_hf_mix_plus_kaggle_train.report.json
```

Audit instruction corpus noise before training:

```bash
python tools/audit_instruction_corpus.py data/corpus/lampung_instruction_train.jsonl --report logs/audit_lampung_instruction.json
python tools/audit_instruction_corpus.py data/corpus/curriculum_stage1_foundation_train.jsonl --report logs/audit_curriculum_stage1.json
```

The audit flags likely Lampung rows contaminated by Batak/Toba markers, translation instruction mismatches, web/search-result artifacts, and malformed JSON rows. Treat this as a triage report: clean or move noisy rows to pretraining text before using them as supervised instruction data.

Build the clean stage-1 foundation corpus:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/curriculum_stage1_foundation_clean.json
python tools/audit_instruction_corpus.py data/corpus/curriculum_stage1_foundation_clean_train.jsonl --report logs/audit_curriculum_stage1_clean.json
```

`curriculum_stage1_foundation_clean.json` excludes the noisy Cendol mixed translation source and filters `corpus_indonesia_translation` rows from the Indonesian corpus source while keeping the cleaner Indonesian, English, Kaggle hoax classification, Lampung train, and Lampung dictionary rows.

Pantau disk Kaggle sebelum mining besar:

```bash
du -sh data data/mined data/corpus checkpoints 2>/dev/null || true
df -h /kaggle/working
```

## 9. LoRA Training

One-command curriculum LoRA:

```powershell
python train_pipeline.py --mode lora-curriculum
```

Resume/skip behavior is automatic: a stage is skipped when its merged output already exists. Force a full rerun:

```powershell
python train_pipeline.py --mode lora-curriculum --force-curriculum
```

Skip corpus rebuild when corpora are already built:

```powershell
python train_pipeline.py --mode lora-curriculum --no-rebuild-corpora
```

Lampung-only LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\lampung_lora.json
```

General LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\general_lora.json
```

Indonesian HF mix LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\indonesian_hf_mix_lora.json
```

Indonesian HF mix + Kaggle Add Input LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\indonesian_hf_mix_plus_kaggle_lora.json
```

Software engineering capability LoRA:

```powershell
python tools\build_software_engineering_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\software_engineering_instruction.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\software_engineering_lora.json
```

Reasoning / Chain-of-Thought LoRA:

```powershell
python tools\build_reasoning_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\reasoning_instruction.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\reasoning_lora.json
```

Indonesian HF mix + Kaggle + Lampung + reasoning LoRA:

```powershell
python tools\build_software_engineering_seed.py
python tools\build_reasoning_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\indonesian_hf_mix_plus_kaggle_reasoning_lora.json
```

Manual curriculum stages, if you do not want the one-command runner:

```powershell
python lora\run_lora.py --config configs\training\curriculum_stage1_foundation_lora.json
python lora\run_lora.py --config configs\training\curriculum_stage2_general_lora.json
python lora\run_lora.py --config configs\training\curriculum_stage3_advanced_lora.json
python lora\run_lora.py --config configs\training\curriculum_stage4_full_lora.json
```

Config Indonesian HF mix memakai `batch_size=2` sebagai titik awal aman. Trainer LoRA bisa menaikkan batch secara konservatif lewat VRAM-aware tuning.

Untuk cluster atau multi-process launch, runtime planner juga membaca environment `WORLD_SIZE`, `RANK`, dan `LOCAL_RANK`. Jalur ini dipakai jika training diluncurkan dengan `torchrun`.

Contoh single-node DDP:

```bash
torchrun --standalone --nproc_per_node=2 main.py
torchrun --standalone --nproc_per_node=2 lora/run_lora.py --config configs/training/indonesian_hf_mix_lora.json
```

Di Kaggle T4x2, command sederhana juga bisa:

```bash
python main.py
python lora/run_lora.py --config configs/training/indonesian_hf_mix_plus_kaggle_lora.json
PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode lora-curriculum
```

Default backward-compatible Lampung run:

```powershell
python lora\run_lora.py
```

## 10. CLI Inference Test

One-shot Kaggle-friendly test:

```powershell
python chat_cli.py --checkpoint checkpoints\lora\model_indonesian_hf_mix_plus_kaggle_merged.pt --prompt "Apa itu bahasa Lampung?"
python chat_cli.py --checkpoint checkpoints\lora\model_indonesian_hf_mix_plus_kaggle_merged.pt --mode lo-id --prompt "Nyak haga mengan manuk di warung paghek jalan"
```

Jika `--checkpoint` tidak diberikan, CLI akan mencoba checkpoint terbaru yang umum dipakai:

```txt
checkpoints/lora/model_indonesian_hf_mix_plus_kaggle_merged.pt
checkpoints/lora/model_indonesian_hf_mix_merged.pt
checkpoints/lora/model_general_merged.pt
checkpoints/lora/model_lampung_merged.pt
checkpoints/best_model.pt
```

Direct auto-router input:

```powershell
@'
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```

Interactive:

```powershell
python chat_cli.py
```

Then type:

```txt
Apa itu Laravel middleware?
Nyak haga mengan manuk di warung paghek jalan
/help
/exit
```

Legacy mode smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```
