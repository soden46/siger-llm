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

For the current SigerLM experiments, keep the dense profile as the baseline:

```powershell
python main.py
```

The default `small` profile uses `d_model=256`, `n_layers=8`, and `max_seq_len=128` for cheap smoke/base runs. If the dense baseline is stable and GPU time is available, use `small_context` to keep the same parameter scale while training with `max_seq_len=256`:

```powershell
$env:SIGER_MODEL_PROFILE="small_context"
python main.py
```

Sparse MoE is opt-in and should be treated as a second experiment after the dense run is healthy:

```powershell
$env:SIGER_MODEL_PROFILE="small_moe"
python main.py
```

The `small_moe` profile keeps the same general SSM/Mamba-like core but adds sparse experts on selected blocks. It increases capacity without making Lampung, Laravel, or any domain rule part of the core model.

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

For Kaggle/general experiments, the main mixed registry is:

```powershell
python tools\build_software_engineering_seed.py
python tools\build_reasoning_seed.py
python tools\build_uncertainty_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 512
```

Use `--max-row-tokens 512` for the first 11.8M-parameter LoRA run. Increase to `768` or `1024` only after label inspection and short training look healthy.

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

Mixed Indonesian HF + Kaggle + reasoning + uncertainty LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 512
python tools\inspect_lora_dataset.py data\corpus\indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl --limit 10 --stats-limit 500 --max-seq-len 512
python lora\run_lora.py --config configs\training\indonesian_hf_mix_plus_kaggle_reasoning_lora.json
```

Default:

```powershell
python lora\run_lora.py
```

The default remains Lampung-safe for backward compatibility.

Before a long LoRA run, always inspect the assistant-only mask. Healthy rows should have supervised assistant tokens greater than zero, and the supervised preview should show answer text rather than the system/user prompt.

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

## Data Method Notes

Current data strategy is data-centric:

- Normalize noisy HF/Kaggle schemas into the SigerLM instruction row.
- Keep the core model general; encode Lampung/Laravel capability through data, adapters, routing, and retrieval.
- Add reasoning rows using `<thought>...</thought>` for structured problem solving.
- Add a small amount of uncertainty-awareness rows so the model can say what it knows, what it infers, and what needs verification.
- Avoid hard-refusal for ordinary unknowns. Use helpful caveats and verification steps instead.
- Keep hard refusal only for unsafe categories such as leaked secrets, credential handling, harmful instructions, medical diagnosis certainty, and financial certainty.

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
