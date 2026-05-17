# PROJECT_CONTEXT.md

This file gives contributors and AI assistants the shortest accurate map of the current SigerLM repository.

## Summary

SigerLM is a custom Python LLM framework built around a State Space Model/Mamba-like architecture. The project is moving toward a general LM workflow while using Bahasa Lampung as a domain adapter and training testbed.

The system currently supports:

- custom SSM language model
- hybrid tokenizer selection
- base training pipeline
- config-driven LoRA fine-tuning
- unified instruction corpus builder
- Lampung dataset extraction and instruction generation
- lookup-first Lampung inference
- general/domain routing in CLI
- evaluation and optimization scaffolding

## Architecture Snapshot

```txt
data sources
  -> domain builders or dataset registry
  -> unified instruction corpus
  -> tokenizer
  -> base model / LoRA training
  -> merged checkpoint
  -> inference router
  -> general chat or Lampung tools
```

## Core Principle

The model architecture is general. Lampung is implemented as a domain layer:

- data: `data/lampung/`, `tools/build_lampung_dataset.py`
- instruction tasks: `tools/build_instruction_dataset.py`
- lookup/rules: `retrieval/`
- runtime pipeline: `inference/lampung_pipeline.py`
- auto-routing: `inference/router.py`

Do not put domain-specific Lampung behavior into `model/`.

## Important Modules

```txt
config/model_config.py
model/siger_model.py
model/ssm_block.py
model/ssm_core.py
tokenizer/hybrid_tokenizer.py
training/dataset.py
training/dataset_registry.py
tools/build_instruction_corpus.py
lora/config.py
lora/dataset.py
lora/run_lora.py
inference/generator.py
inference/chat.py
inference/router.py
inference/lampung_pipeline.py
retrieval/instruction_lookup.py
retrieval/compositional_translator.py
```

## Config Files

```txt
configs/datasets/lampung_instruction.json
configs/datasets/general_instruction.json
configs/training/lampung_lora.json
configs/training/general_lora.json
```

## Current Dataset State

Latest verified build:

```txt
data/lampung/processed/percakapan_1000_pairs.jsonl: 3100 rows
data/lampung/processed/compositional_pairs.jsonl: 1968 rows
data/lampung/final/train.jsonl: 4325 rows
data/lampung/final/valid.jsonl: 541 rows
data/lampung/final/test.jsonl: 541 rows
data/lampung/final/train_augmented_instruction.jsonl: 32059 rows
data/corpus/lampung_instruction_train.jsonl: 30701 rows
data/corpus/general_instruction_train.jsonl: 30704 rows
```

`general_instruction_train.jsonl` is currently still dominated by Lampung because local general text files are small. To make SigerLM more general, add larger instruction/chat/text sources to `configs/datasets/general_instruction.json`.

## Main Commands

Build Lampung data:

```powershell
python tools\extract_percakapan_pdf.py
python tools\build_compositional_lampung_dataset.py
python tools\build_lampung_dataset.py
python tools\build_instruction_dataset.py
```

Build unified corpora:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

Train LoRA:

```powershell
python lora\run_lora.py --config configs\training\lampung_lora.json
python lora\run_lora.py --config configs\training\general_lora.json
```

Run CLI:

```powershell
python chat_cli.py
```

CLI modes:

```txt
0 auto/general router
1 Lampung O -> Indonesia
2 Indonesia -> Lampung O
3 Lampung O -> English
4 Lampung reasoning
5 general chat
6 Lampung word order
```

## Latest Smoke Result

```txt
Mode: 0
Input: Nyak haga mengan manuk di warung paghek jalan
Assistant: aku mau makan ayam di warung dekat jalan
Route: lampung_to_id
Source: exact instruction lookup
```

## Current Priorities

1. Expand general instruction/chat corpora.
2. Keep Lampung domain as an adapter, not the whole architecture.
3. Train/evaluate `general_lora.json` once general corpus is meaningful.
4. Add small automated tests for corpus builder, router, and lookup.
5. Improve evaluation coverage for Lampung ID/EN and general chat.
6. Keep CPU/VPS memory use under control.
