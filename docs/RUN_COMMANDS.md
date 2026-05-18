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
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py
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
SIGER_MODEL_PROFILE=base python main.py
SIGER_MODEL_PROFILE=reasoning_base python main.py
```

Profile default tetap `small` agar smoke test Kaggle lebih aman. Profile `base` memakai `d_model=512`, `n_layers=12`; profile `reasoning_base` menaikkan context training ke `max_seq_len=512`.

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

Pantau disk Kaggle sebelum mining besar:

```bash
du -sh data data/mined data/corpus checkpoints 2>/dev/null || true
df -h /kaggle/working
```

## 9. LoRA Training

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
