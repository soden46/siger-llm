# LoRA Fine-Tuning

SigerLM uses a custom LoRA implementation for parameter-efficient instruction tuning.

## Files

```txt
lora/layer.py      LoRALinear
lora/model.py      adapter injection, save/load/merge
lora/config.py     LoRAConfig dataclass and JSON loader
lora/dataset.py    instruction formatting and assistant-only loss mask
lora/trainer.py    training loop
lora/run_lora.py   config-driven runner
```

## Config-Driven Runner

Run with a training config:

```powershell
python lora\run_lora.py --config configs\training\lampung_lora.json
python lora\run_lora.py --config configs\training\general_lora.json
```

Default run:

```powershell
python lora\run_lora.py
```

The default remains a Lampung-safe recipe for backward compatibility.

## Current Training Configs

Lampung:

```txt
configs/training/lampung_lora.json
dataset_path: data/corpus/lampung_instruction_train.jsonl
merged_output: checkpoints/lora/model_lampung_merged.pt
max_steps: 3000
max_seq_len: 256
```

General:

```txt
configs/training/general_lora.json
dataset_path: data/corpus/general_instruction_train.jsonl
merged_output: checkpoints/lora/model_general_merged.pt
max_steps: 5000
max_seq_len: 384
```

## Build Data Before Training

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

## Dataset Formatting

`lora/dataset.py` accepts local instruction rows:

```json
{"system":"Kamu adalah SigerLM...","instruction":"Apa itu AI?","input":"","output":"AI adalah..."}
```

It formats them as:

```txt
<|system|>...<|end_turn|>
<|user|>instruction + input<|end_turn|>
<|assistant|>output<|end_turn|>
```

Only assistant tokens contribute to the loss. System/user tokens are masked with `-100`.

## LoRA Targets

Current target modules:

```txt
in_proj
out_proj
x_proj
dt_proj
```

## Merge

`run_lora.py` merges the adapter after training using `merged_output` from the config.

Example:

```txt
checkpoints/lora/model_lampung_merged.pt
checkpoints/lora/model_general_merged.pt
```

## Verification

```powershell
python -m py_compile lora\config.py lora\dataset.py lora\run_lora.py
python lora\run_lora.py --help
```
