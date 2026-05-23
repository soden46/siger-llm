# Kaggle T4x2 Continue Plan

This plan continues the small 5.7M SigerLM checkpoint trained with the HF BPE
tokenizer (`vocab_size=8035`). Keep the same tokenizer files when resuming or
warm-starting. A larger tokenizer needs a new base training run because the
embedding and LM head shapes change.

## 1. Continue Base Training

Use this when you have only `checkpoints/best_model.pt` or a merged model
state_dict. This warm-starts model weights and starts a fresh optimizer and
scheduler for additional updates.

```bash
export SIGER_MODEL_PROFILE=small
export SIGER_INIT_CHECKPOINT=checkpoints/best_model.pt
export SIGER_CHECKPOINT_DIR=checkpoints/t4x2_base_continue
export SIGER_RESUME=0
export SIGER_MAX_STEPS=4000
export SIGER_MAX_SEQ_LEN=256
export SIGER_BATCH_SIZE=16
export SIGER_GRAD_ACCUM_STEPS=2
export SIGER_WARMUP_STEPS=0
export SIGER_MAX_LR=2e-4
export SIGER_MIN_LR=2e-5
export SIGER_SAVE_EVERY=250
export SIGER_DEVICE=auto
export SIGER_PRECISION=auto

python main.py
```

With T4x2, `python main.py` auto-relaunches with DDP. Effective batch is:

```txt
batch_size * grad_accum_steps * world_size = 16 * 2 * 2 = 64
```

If VRAM gets tight, use:

```bash
export SIGER_BATCH_SIZE=8
export SIGER_GRAD_ACCUM_STEPS=4
```

That keeps the same global effective batch of 64.

## 2. True Resume From a Full Step Checkpoint

Use this when you still have a full checkpoint like
`step_0000850_*.pt` or `step_0001000_*.pt` with optimizer and scheduler state.
Put it in the checkpoint directory and make `latest.json` point to it, or keep
the original directory layout from training.

```bash
export SIGER_MODEL_PROFILE=small
export SIGER_CHECKPOINT_DIR=checkpoints
export SIGER_RESUME=1
export SIGER_MAX_STEPS=5000
export SIGER_MAX_SEQ_LEN=256
export SIGER_WARMUP_STEPS=0

python main.py
```

This continues global step numbering to 5000. Warm-start mode in section 1
instead runs the requested number of new optimizer updates.

## 3. Rebuild Balanced LoRA Corpus

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/gpu_stage2_balanced.json
```

## 4. LoRA Polish on T4x2

```bash
python lora/run_lora.py --config configs/training/gpu_t4x2_repair_polish_lora.json
```

The LoRA config uses `batch_size=2`, `grad_accum=8`. On T4x2 DDP this is a
global effective batch of 32. On one GPU, set `grad_accum=16` to keep effective
batch 32.

## Notes

- Do not retrain the tokenizer mid-continuation. A 16k-32k tokenizer is a good
  next-generation run, not a drop-in change for this 8035-vocab checkpoint.
- If general chat still sounds like news continuation, reduce or exclude
  text-completion/news rows for the next LoRA polish corpus.
