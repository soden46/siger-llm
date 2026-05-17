# Installation

## Requirements

Recommended development environment:

```txt
Python: 3.11
RAM: 4GB minimum, 8GB+ recommended
CPU: 2 cores minimum
GPU: optional
OS: Windows, Linux, or WSL2
```

## Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Linux / WSL

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Verify Imports

```powershell
python -c "import torch; import fastapi; import datasets; print('ok')"
python -m py_compile config\model_config.py model\siger_model.py tokenizer\hybrid_tokenizer.py
```

## Model Smoke Test

```powershell
python -c "import torch; from config.model_config import SigerConfig; from model.siger_model import SigerLM; c=SigerConfig(vocab_size=1000,d_model=64,n_layers=2); m=SigerLM(c); x=torch.randint(0,1000,(2,32)); y,_=m(x); assert y.shape==(2,32,1000); print('model ok')"
```

## Build Current Corpora

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

## Run CLI

```powershell
python chat_cli.py
```

CLI smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```

Expected route:

```txt
Route: lampung_to_id
Source: exact instruction lookup
```

## Train LoRA

Lampung:

```powershell
python lora\run_lora.py --config configs\training\lampung_lora.json
```

General:

```powershell
python lora\run_lora.py --config configs\training\general_lora.json
```

## Notes

- If tokenizer backend changes, old checkpoints may not load.
- If `config.model_config` import fails when running a script directly, ensure `config/__init__.py` exists.
- For low-RAM machines, keep LoRA `batch_size=1` and use gradient accumulation.
