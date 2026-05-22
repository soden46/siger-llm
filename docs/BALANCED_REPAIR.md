# Balanced Repair Training

Balanced repair is a safer Stage 2 recipe for small SigerLM checkpoints.

The normal `curriculum_stage2_general` corpus mixes many domains at once. For a
small base model, that can make batch loss jump around because each step may see
a very different domain. Balanced repair uses smaller, capped source slices and
keeps the original SigerLM LoRA concept across both projection and SSM timing
paths, but lowers the update pressure with a gentler learning rate and dropout.

## CPU Repair

Use this when Kaggle has no GPU or when you only want a bounded repair run:

```bash
python tools/build_instruction_corpus.py \
  --registry configs/datasets/curriculum_stage1_foundation_clean.json \
  --max-row-tokens 512

python lora/run_lora.py \
  --config configs/training/curriculum_stage1_foundation_lora.json

python tools/build_instruction_corpus.py \
  --registry configs/datasets/cpu_repair_general_balanced.json \
  --max-row-tokens 512

python lora/run_lora.py \
  --config configs/training/cpu_repair_general_balanced_lora.json
```

Output:

```txt
checkpoints/lora/model_cpu_repair_general_balanced_merged.pt
```

Continue for another bounded CPU repair pass:

```bash
python lora/run_lora.py \
  --config configs/training/cpu_repair_general_balanced_continue_lora.json
```

The continue config uses the first pass output as its base checkpoint:

```txt
checkpoints/lora/model_cpu_repair_general_balanced_merged.pt
```

Continue output:

```txt
checkpoints/lora/model_cpu_repair_general_balanced_continue_merged.pt
```

Smoke test:

```bash
python chat_cli.py \
  --checkpoint checkpoints/lora/model_cpu_repair_general_balanced_merged.pt \
  --mode chat \
  --prompt "Jelaskan secara singkat apa itu machine learning." \
  --device cpu

python chat_cli.py \
  --checkpoint checkpoints/lora/model_cpu_repair_general_balanced_merged.pt \
  --mode auto \
  --prompt "Nyak haga mengan manuk di warung paghek jalan" \
  --device cpu
```

## GPU Stage 2 Balanced

Use this when CUDA is available:

```bash
python tools/build_instruction_corpus.py \
  --registry configs/datasets/curriculum_stage1_foundation_clean.json \
  --max-row-tokens 512

python tools/build_instruction_corpus.py \
  --registry configs/datasets/gpu_stage2_balanced.json \
  --max-row-tokens 768

python lora/run_lora.py \
  --config configs/training/gpu_stage2_balanced_lora.json
```

Output:

```txt
checkpoints/lora/model_gpu_stage2_balanced_merged.pt
```

## Why This Is Gentler

The balanced configs:

- cap each source with `max_items`
- avoid huge text-completion blocks in repair
- keep Lampung as a bridge slice, not the whole stage
- keep code/reasoning present but small
- keep SigerLM's full LoRA target concept:
  `in_proj`, `out_proj`, `x_proj`, and `dt_proj`
- use lower learning rate and higher dropout so `x_proj`/`dt_proj` stay useful
  without dominating the small model

Target modules stay aligned with the main curriculum:

```json
["in_proj", "out_proj", "x_proj", "dt_proj"]
```

For the current small model, the stability improvement comes from cleaner data
composition, `rank=8`, gentler LR, and `dropout=0.08`, not from removing the SSM
adapter targets.
