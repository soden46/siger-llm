# AGENTS.md - Panduan AI Agents untuk SigerLM

Dokumen ini dipakai oleh AI coding assistants yang bekerja di repository SigerLM.

## Project Summary

SigerLM adalah framework eksperimen Large Language Model berbasis State Space Model/Mamba-like architecture. Target utama proyek sekarang adalah membangun LM general yang modular, ringan, dan bisa diuji di CPU/VPS. Dataset Bahasa Lampung dipakai sebagai domain adapter dan testbed training, bukan sebagai batas kemampuan model.

Stack utama:

- Python 3.11
- PyTorch
- Custom SSM/Mamba-like model
- Hybrid tokenizer: HF ByteLevel BPE bila tersedia, fallback tiktoken
- LoRA fine-tuning custom
- FastAPI/ONNX/quantization untuk deployment
- Retrieval/rule layer untuk domain Lampung

## Current Architecture

```txt
raw data / domain data
  -> tools/build_* or tools/build_instruction_corpus.py
  -> unified instruction corpus
  -> tokenizer/hybrid_tokenizer.py
  -> base training or LoRA fine-tuning
  -> merged checkpoint
  -> inference router
  -> general chat or Lampung domain tools
```

Core LM harus tetap general:

- `model/` berisi arsitektur model.
- `training/` berisi base training dan dataset registry.
- `lora/` berisi instruction tuning.
- `inference/` berisi generator, chat, router, dan pipeline domain.
- `retrieval/` berisi lookup/lexicon/compositional translator untuk Lampung.
- `modalities/` berisi kontrak adapter modality-agnostic untuk vision/audio/video/sensor/graph/dll.
- `tools/` berisi builder/extractor dataset.
- `configs/` berisi registry dataset dan config training.
- `train_pipeline.py` berisi pipeline Dense -> MoE -> LoRA dan mode LoRA curriculum otomatis.

Lampung-specific code tidak boleh ditaruh di core model. Taruh di `retrieval/`, `inference/lampung_pipeline.py`, dan `tools/build_lampung_dataset.py`.

## Important Files

```txt
config/model_config.py                 SigerConfig, baca sebelum ubah model
config/__init__.py                     import package guard
model/siger_model.py                   full LM
tokenizer/hybrid_tokenizer.py          tokenizer selector
training/dataset_registry.py           general dataset registry
tools/build_instruction_corpus.py      unified corpus builder
configs/datasets/general_instruction.json
configs/datasets/lampung_instruction.json
configs/datasets/curriculum_stage1_foundation.json
configs/datasets/curriculum_stage2_general.json
configs/datasets/curriculum_stage3_advanced.json
configs/datasets/curriculum_stage4_full.json
configs/training/general_lora.json
configs/training/lampung_lora.json
configs/training/lora_curriculum.json
configs/training/curriculum_stage*_lora.json
lora/run_lora.py                       config-driven LoRA runner
lora/dataset.py                        instruction/chat formatting
inference/router.py                    general vs domain router
inference/lampung_pipeline.py          Lampung lookup-first pipeline
retrieval/instruction_lookup.py        exact and bag-of-words lookup
retrieval/compositional_translator.py  rule composer ID/LO/EN
chat_cli.py                            local CLI smoke tests
modalities/base.py                     kontrak adapter modality -> sequence embedding
modalities/registry.py                 daftar capability modality dan objective
docs/MODALITY_AGNOSTIC_BACKBONE.md     roadmap Siger sebagai backbone modality-agnostic
```

## Coding Rules

- Prefer incremental changes. Do not rewrite the whole project unless asked.
- Keep the model architecture independent from Lampung-specific logic.
- Keep the SSM backbone independent from any specific modality; non-text inputs should enter as projected `inputs_embeds`.
- Use type hints for public functions.
- Use `rg` for search.
- Use `apply_patch` for manual edits.
- Do not change special token IDs casually.
- Do not change default model config unless backward compatibility is considered.
- Do not revert user changes. The worktree can be dirty.
- Keep CPU/VPS constraints in mind: 2 cores, 4GB RAM target.

## Dataset Rules

Preferred instruction row:

```json
{"instruction":"...","input":"...","output":"...","system":"optional system prompt","source":"...","type":"..."}
```

Preferred chat row for registry sources:

```json
{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Dataset registry sources currently support:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`
- `mined_parallel_jsonl`

Build corpora:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\curriculum_stage4_full.json
```

## Training Commands

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

Automatic easy-to-hard LoRA curriculum:

```powershell
python train_pipeline.py --mode lora-curriculum
```

Use `--dry-run` to preview commands, `--no-rebuild-corpora` to reuse existing corpora, and `--force-curriculum` to rerun stages whose merged outputs already exist.

Default `python lora\run_lora.py` remains a Lampung-safe default for backward compatibility.

## Current Verified Results

Latest local smoke results:

```txt
Lampung processed PDF conversations: 3100 rows
Synthetic compositional rows: 1968
final train/valid/test: 4325 / 541 / 541
train rows with English field: 1605
train_augmented_instruction: 32059 rows
general_instruction_train: 30704 rows
lampung_instruction_train: 30701 rows
kaggle_local_inputs_train: 51969 rows
curriculum_stage1_foundation_train: 84605 rows
curriculum_stage2_general_train: 186672 rows
curriculum_stage3_advanced_train: 218596 rows
curriculum_stage4_full_train: 218596 rows
```

CLI mode `0` auto-routes:

```txt
Input: Nyak haga mengan manuk di warung paghek jalan
Route: lampung_to_id
Source: exact instruction lookup
Output: aku mau makan ayam di warung dekat jalan
```

Lampung O -> English examples:

```txt
Nyak haga mengan manuk di warung paghek jalan
-> i want to eat chicken at the stall near the road

Nyak ago belei buku di pasar
-> i want to buy a book at the market
```

## Testing

Run relevant compile checks after edits:

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py train_pipeline.py
python train_pipeline.py --mode lora-curriculum --dry-run
```

CLI smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```

Model smoke:

```powershell
python -c "import torch; from config.model_config import SigerConfig; from model.siger_model import SigerLM; c=SigerConfig(vocab_size=1000,d_model=64,n_layers=2); m=SigerLM(c); x=torch.randint(0,1000,(2,32)); y,_=m(x); assert y.shape==(2,32,1000); print('ok')"
```

## Safe Modification Order

For model changes:

1. `config/model_config.py`
2. `model/ssm_core.py`
3. `model/ssm_block.py`
4. `model/siger_model.py`
5. tests/smoke

For dataset/training changes:

1. `configs/datasets/*.json`
2. `tools/build_instruction_corpus.py` or domain builder
3. `lora/dataset.py`
4. `configs/training/*.json`
5. `lora/run_lora.py`
6. compile + corpus build

For inference changes:

1. `inference/prompt_builder.py`
2. `inference/lampung_pipeline.py` if domain-specific
3. `inference/router.py` if routing behavior changes
4. `chat_cli.py`
5. CLI smoke
