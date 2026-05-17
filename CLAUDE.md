# CLAUDE.md

You are working on SigerLM, a custom Python LLM framework.

## Current Direction

SigerLM is a general LM framework using a custom SSM/Mamba-like architecture. Bahasa Lampung is currently a domain adapter and training testbed, not the whole product.

Keep the system modular:

- core model stays general
- dataset registry handles multiple corpora
- LoRA configs select training targets
- inference router chooses general chat vs Lampung domain tools

## Rules

- Do not rewrite the whole project.
- Keep changes small, typed, and modular.
- Do not move Lampung-specific behavior into `model/`.
- Update docs whenever behavior or commands change.
- Prefer config files over hardcoded training paths.
- Do not invent results. Mark estimates as estimates.
- Run compile/smoke commands after edits when feasible.
- If the worktree is dirty, do not revert unrelated user changes.

## Key Modules

```txt
model/                      SigerLM + SSM blocks
tokenizer/                  tokenizer wrappers and HF BPE trainer
training/dataset_registry.py general dataset registry
tools/build_instruction_corpus.py unified corpus builder
lora/                       LoRA config, dataset, trainer, runner
inference/router.py         general/domain routing
inference/lampung_pipeline.py Lampung lookup-first pipeline
retrieval/                  Lampung lookup, lexicon, compositional rules
configs/datasets/           dataset registry JSON files
configs/training/           LoRA training config JSON files
chat_cli.py                 local CLI smoke testing
```

## Latest Verified Commands

Build Lampung instruction corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
```

Build general instruction corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

Train Lampung LoRA:

```powershell
python lora\run_lora.py --config configs\training\lampung_lora.json
```

Train general LoRA:

```powershell
python lora\run_lora.py --config configs\training\general_lora.json
```

CLI smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```

## Latest Verified Data State

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

## Priority

1. Keep imports stable.
2. Keep dataset and training paths config-driven.
3. Improve general corpus coverage.
4. Preserve Lampung lookup accuracy.
5. Add tests/evals around router, corpus builder, and LoRA data formatting.
6. Improve docs when commands or architecture change.
