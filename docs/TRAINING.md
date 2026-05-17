# Training

SigerLM has two training paths:

1. Base LM training with next-token prediction.
2. LoRA instruction tuning with config-driven datasets.

The current generalization work is mostly in the second path: unified instruction corpora and configurable LoRA runs.

## Base Training

Base training uses:

```txt
training/dataset.py
training/trainer.py
training/optimizer.py
training/checkpoint.py
main.py
```

Flow:

```txt
raw text
  -> tokenizer
  -> token chunks
  -> next-token labels
  -> SigerLM
  -> checkpoint
```

If tokenizer backend changes, old checkpoints may not load because token IDs and embedding size can change.

## Instruction Corpus Builder

Unified instruction corpus builder:

```txt
training/dataset_registry.py
tools/build_instruction_corpus.py
configs/datasets/*.json
```

Build Lampung corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
```

Build general corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

Supported source formats:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`

Instruction JSONL:

```json
{"instruction":"Apa itu AI?","input":"","output":"AI adalah teknologi yang membuat komputer dapat melakukan tugas yang biasanya membutuhkan kecerdasan manusia."}
```

Chat JSONL:

```json
{"messages":[{"role":"user","content":"Apa itu AI?"},{"role":"assistant","content":"AI adalah teknologi yang membantu komputer belajar dan mengambil keputusan dari data."}]}
```

Text completion source:

```json
{
  "name": "indonesian_text",
  "path": "data/indonesian.txt",
  "format": "text_completion"
}
```

## LoRA Training

LoRA runner:

```txt
lora/run_lora.py
lora/config.py
lora/dataset.py
lora/trainer.py
configs/training/*.json
```

Lampung LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python lora\run_lora.py --config configs\training\lampung_lora.json
```

General LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python lora\run_lora.py --config configs\training\general_lora.json
```

Default:

```powershell
python lora\run_lora.py
```

The default remains Lampung-safe for backward compatibility.

## Current Configs

```txt
configs/training/lampung_lora.json
  dataset_path: data/corpus/lampung_instruction_train.jsonl
  max_steps: 3000
  max_seq_len: 256
  merged_output: checkpoints/lora/model_lampung_merged.pt

configs/training/general_lora.json
  dataset_path: data/corpus/general_instruction_train.jsonl
  max_steps: 5000
  max_seq_len: 384
  merged_output: checkpoints/lora/model_general_merged.pt
```

## Latest Verified Data Counts

```txt
data/corpus/lampung_instruction_train.jsonl: 30701 rows
data/corpus/general_instruction_train.jsonl: 30704 rows
```

The general corpus is currently mostly Lampung plus tiny local text sources. For real general chatbot ability, add larger instruction/chat corpora to `configs/datasets/general_instruction.json`.

## CPU/VPS Tips

For 2 core / 4GB RAM:

- keep `batch_size` at 1 or 2
- use `grad_accum` for effective batch size
- keep `max_seq_len` 256-512 for LoRA
- prefer LoRA before full fine-tuning
- use hardware detection in `lora/run_lora.py`

## Verification

```powershell
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py
python lora\run_lora.py --help
```
