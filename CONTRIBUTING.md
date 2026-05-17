# Contributing to SigerLM

Thanks for helping improve SigerLM.

SigerLM is an experimental general-purpose LM framework built around a custom SSM/Mamba-like architecture. Bahasa Lampung is currently the first domain adapter and training testbed.

## Useful Contributions

- bug fixes
- documentation updates
- dataset builders
- legal and well-sourced datasets
- general instruction/chat corpora
- Lampung O/Nyo validation data
- evaluation scripts
- CPU inference improvements
- LoRA training recipes

## Before Opening a PR

1. Keep changes scoped.
2. Do not commit checkpoints or large raw datasets unless explicitly requested.
3. Run relevant compile/smoke commands.
4. Update docs when commands, architecture, or behavior changes.
5. Mention dataset sources and license/usage notes.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Smoke Checks

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\run_lora.py
```

## Dataset Contribution Format

Instruction JSONL:

```json
{"instruction":"Apa itu AI?","input":"","output":"AI adalah teknologi yang membantu komputer belajar dari data.","source":"manual","type":"qa"}
```

Lampung pair JSONL:

```json
{"dialect":"o","lampung":"api kabar niku?","indonesian":"apa kabar kamu?","english":"how are you?","source":"manual_native_review","type":"daily_conversation"}
```

Chat JSONL:

```json
{"messages":[{"role":"user","content":"Apa itu AI?"},{"role":"assistant","content":"AI adalah teknologi yang membantu komputer melakukan tugas cerdas."}]}
```

## Dataset Rules

- Do not add data from sources that prohibit reuse.
- Include the source field.
- Mark synthetic data with `"synthetic": true`.
- Prefer native-speaker validation for Lampung data.
- Keep general-chat data separate through `configs/datasets/general_instruction.json`.

## Current Training Commands

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python lora\run_lora.py --config configs\training\lampung_lora.json

python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python lora\run_lora.py --config configs\training\general_lora.json
```

## Commit Style

Use clear commit messages:

```txt
feat: add general dataset registry
fix: handle empty text completion source
docs: update SigerLM architecture docs
test: add router smoke coverage
```
