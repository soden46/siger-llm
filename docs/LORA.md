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

## One-Command Curriculum Runner

For automatic easy-to-hard LoRA training, use `train_pipeline.py`:

```powershell
python train_pipeline.py --mode lora-curriculum
```

Kaggle/CUDA example:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode lora-curriculum
```

`lora/run_lora.py` applies an automatic hardware policy before training. CUDA runs keep the requested config when VRAM allows it and use conservative caps on low-VRAM GPUs. CPU-only runs become bounded smoke/debug runs by default, so losing Kaggle GPU quota does not accidentally launch a multi-day FP32 training job.

Override only when intentional:

```bash
SIGER_ALLOW_CPU_FULL_TRAIN=1 python lora/run_lora.py --config configs/training/general_lora.json
python lora/run_lora.py --config configs/training/general_lora.json --no-hardware-policy
```

The runner reads:

```txt
configs/training/lora_curriculum.json
```

It runs:

```txt
stage1_foundation -> stage2_general -> stage3_advanced -> stage4_full
```

For each stage it can rebuild the corpus, run `lora/run_lora.py`, merge the adapter, and then use that merged checkpoint as the base checkpoint for the next stage. Logs are written to `logs/lora_curriculum/`, and progress is written to `checkpoints/lora/curriculum_state.json`.

Useful flags:

```powershell
python train_pipeline.py --mode lora-curriculum --dry-run
python train_pipeline.py --mode lora-curriculum --no-rebuild-corpora
python train_pipeline.py --mode lora-curriculum --force-curriculum
```

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

Indonesian HF mix:

```txt
configs/training/indonesian_hf_mix_lora.json
dataset_path: data/corpus/indonesian_hf_mix_train.jsonl
merged_output: checkpoints/lora/model_indonesian_hf_mix_merged.pt
max_steps: 5000
max_seq_len: 384
```

Indonesian HF mix + Kaggle + reasoning + uncertainty:

```txt
configs/training/indonesian_hf_mix_plus_kaggle_reasoning_lora.json
dataset_path: data/corpus/indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl
merged_output: checkpoints/lora/model_indonesian_hf_mix_plus_kaggle_reasoning_merged.pt
max_steps: 3000
max_seq_len: 1024
```

Curriculum final:

```txt
configs/training/curriculum_stage4_full_lora.json
dataset_path: data/corpus/curriculum_stage4_full_train.jsonl
merged_output: checkpoints/lora/model_curriculum_stage4_full_merged.pt
max_steps: 2000
max_seq_len: 1024
```

## Build Data Before Training

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix.json
```

For the current mixed Kaggle recipe:

```powershell
python tools\build_software_engineering_seed.py
python tools\build_reasoning_seed.py
python tools\build_uncertainty_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 512
python tools\inspect_lora_dataset.py data\corpus\indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl --limit 10 --stats-limit 500 --max-seq-len 512
```

For the current curriculum recipe:

```powershell
python train_pipeline.py --mode lora-curriculum --dry-run
python tools\debug_lora_dataset.py data\corpus\curriculum_stage4_full_train.jsonl --tokenizer auto --limit 3 --stats-limit 500 --max-seq-len 1024
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

`lora/trainer.py` uses normal causal LM alignment:

```txt
logits[:, :-1] predicts labels[:, 1:]
```

`lora/dataset.py` stores labels at their original token positions, so the trainer shift lets the `<|assistant|>` position predict the first answer token. Do not remove this shift unless the dataset label semantics are changed everywhere.

Reasoning and uncertainty rows may include `<thought>...</thought>` inside the assistant output. These tokens are supervised like normal assistant text. The goal is to teach structured reasoning and honest confidence statements, not to make the model refuse ordinary tasks.

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
python -m py_compile train_pipeline.py
python lora\run_lora.py --help
python train_pipeline.py --mode lora-curriculum --dry-run
```
