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
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
```

General corpus lama:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

General assistant mining corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json
```

## 7. Mining Q&A dan Laravel

Smoke kecil dulu:

```powershell
python tools\mine_general_assistant_data.py --preset all --max-items 200 --max-laravel-pages 5 --max-santrikoding-articles 10
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
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json
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

Build corpus instruction:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix.json
```

Output penting:

```txt
data/mined/hf_indonesia/indonesian_hf_mix_instruction.jsonl
data/corpus/indonesian_hf_mix_train.jsonl
data/indonesian_hf_mix.txt
```

`data/indonesian_hf_mix.txt` otomatis bisa ikut base training karena `main.py` membaca semua file `.txt` di `data/`.

## 9. LoRA Training

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

Indonesian HF mix LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix.json
python lora\run_lora.py --config configs\training\indonesian_hf_mix_lora.json
```

Default backward-compatible Lampung run:

```powershell
python lora\run_lora.py
```

## 10. CLI Inference Test

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
