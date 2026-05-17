# Architecture

SigerLM is a custom LLM framework using a State Space Model/Mamba-like architecture. The system is designed as a general LM stack with optional domain adapters. Bahasa Lampung is currently the first serious domain adapter and training testbed.

## System Layers

```txt
Data Layer
  raw text, JSONL instruction data, chat data, domain corpora

Dataset Layer
  extraction tools
  domain builders
  dataset registry
  unified instruction corpus

Tokenizer Layer
  hybrid tokenizer selector
  optional HF ByteLevel BPE
  fallback tokenizer

Model Layer
  embedding
  SSM blocks
  final norm
  LM head

Training Layer
  base next-token training
  LoRA instruction tuning
  config-driven runs

Inference Layer
  generator
  chat session
  domain pipelines
  router

Optimization/Evaluation Layer
  evaluation scaffolds
  CPU threading
  ONNX export
  quantization
```

## Core Model Boundary

The model is intentionally domain-neutral:

```txt
model/
  ssm_core.py
  ssm_block.py
  siger_model.py
```

No Lampung-specific translation, lookup, or rule behavior belongs in `model/`.

## Dataset Architecture

General corpus flow:

```txt
configs/datasets/*.json
  -> training/dataset_registry.py
  -> tools/build_instruction_corpus.py
  -> data/corpus/*_instruction_train.jsonl
```

Supported registry source formats:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`

Lampung domain flow:

```txt
data/lampung/raw/
  -> tools/extract_*.py
  -> data/lampung/processed/
  -> tools/build_lampung_dataset.py
  -> data/lampung/final/train|valid|test.jsonl
  -> tools/build_instruction_dataset.py
  -> data/lampung/final/*_augmented_instruction.jsonl
  -> tools/build_instruction_corpus.py
  -> data/corpus/lampung_instruction_train.jsonl
```

## Training Architecture

Base training uses `training/` for next-token prediction.

LoRA training uses:

```txt
lora/config.py
lora/dataset.py
lora/model.py
lora/trainer.py
lora/run_lora.py
configs/training/*.json
```

Current training configs:

```txt
configs/training/lampung_lora.json
configs/training/general_lora.json
```

`lora/run_lora.py` accepts:

```powershell
python lora\run_lora.py --config configs\training\general_lora.json
```

## Inference Architecture

General chat:

```txt
ChatSession
  -> ContextManager
  -> Generator
```

Lampung domain:

```txt
LampungPipeline
  -> InstructionLookup
  -> LampungCompositionalTranslator
  -> LampungLexicon
  -> Generator fallback
```

Router:

```txt
SigerRouter
  -> general_chat
  -> lampung_to_id
  -> id_to_lampung
  -> lampung_to_en
```

CLI:

```txt
chat_cli.py
  mode 0: auto/router
  mode 1: Lampung O -> Indonesia
  mode 2: Indonesia -> Lampung O
  mode 3: Lampung O -> English
  mode 4: reasoning
  mode 5: general chat
  mode 6: word order
```

## Latest Verified State

```txt
Lampung PDF conversation extraction: 3100 rows
Synthetic compositional pairs: 1968 rows
Final Lampung split: 4325 / 541 / 541
Train rows with English field: 1605
Train augmented instruction: 32059 rows
Lampung unified corpus: 30701 rows
General unified corpus: 30704 rows
```

The general corpus is currently small outside Lampung. The architecture is ready for general training, but broad chatbot ability requires larger and better general instruction/chat data.
