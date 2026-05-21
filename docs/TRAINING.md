# Training

SigerLM has two training paths:

1. Base LM training with next-token prediction.
2. LoRA instruction tuning with config-driven datasets.
3. Automatic easy-to-hard LoRA curriculum through `train_pipeline.py --mode lora-curriculum`.

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

For Dense -> MoE upcycling, the dense checkpoint must match the MoE profile's base tensor shape. Use `moe_dense_base` before `small_moe`; both use `d_model=384`, `n_layers=10`, and `max_seq_len=512`.

## Adaptive Training Pipeline

`train_pipeline.py` can run the dense -> MoE -> LoRA flow automatically with metric gates:

```powershell
python train_pipeline.py --mode auto --lora-config configs\training\general_lora.json
```

Default gates:

```txt
dense -> MoE: step >= 1500 and loss <= 3.5
MoE -> LoRA: latest checkpoint loss delta <= 0.005
```

The default dense stage uses `moe_dense_base` and writes checkpoints to `checkpoints/auto/dense_moe_base`. When the dense gate passes, the MoE stage uses `small_moe` and is warm-started from the dense checkpoint with shared SSM/tokenizer weights loaded and new MoE tensors initialized as additional capacity. The dense SigerLM block has no standalone dense FFN to clone into experts, so the transition intentionally copies compatible SSM weights instead of pretending an FFN copy exists.

`train_pipeline.py` validates Dense -> MoE profile compatibility before training. If `d_model` or `n_layers` differ, the run fails early with a clear profile mismatch error. Override the pair only when the shapes are intentionally compatible:

```powershell
python train_pipeline.py --mode auto --dense-profile moe_dense_base --moe-profile small_moe
```

The MoE stage writes checkpoints to `checkpoints/auto/moe`. Before that stage starts, `optimization/moe_sizing.py` chooses `moe_num_experts`, `moe_top_k`, and `moe_layers_every` from the current hardware profile and the latest dense loss. This keeps CPU/VPS runs conservative while giving stronger CUDA runs more expert capacity. When the plateau gate passes, LoRA is trained from the latest MoE checkpoint and merged according to the selected LoRA config.

MoE training includes anti-collapse routing pressure:

```txt
loss = language_model_loss + moe_aux_loss_weight * load_balance_loss
```

`SparseMoE` combines Switch-style load balancing, a router importance penalty, and small training-time router jitter. During MoE runs, the training log can include:

```txt
moe_aux=...
moe_dead=...
```

`moe_dead` should trend toward `0.0`; sustained high values mean some experts are not receiving tokens and `moe_aux_loss_weight` or `moe_router_jitter` may need to be increased.

Useful overrides:

```powershell
python train_pipeline.py --dense-loss-threshold 4.0 --dense-min-steps 1000
python train_pipeline.py --moe-plateau-delta 0.01 --lora-config configs\training\lampung_lora.json
python train_pipeline.py --force-stage lora --lora-config configs\training\general_lora.json
```

Kaggle staged base training:

```bash
# Small CPU-safe run when GPU quota is unavailable.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt SIGER_MODEL_PROFILE=small SIGER_RESUME=1 SIGER_SAVE_EVERY=50 SIGER_MAX_STEPS=1000 SIGER_DEVICE=cpu python main.py

# Dense base compatible with the MoE stage.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt SIGER_MODEL_PROFILE=moe_dense_base SIGER_CHECKPOINT_DIR=checkpoints/auto/dense_moe_base SIGER_RESUME=1 SIGER_SAVE_EVERY=100 SIGER_MAX_STEPS=3000 SIGER_DEVICE=auto SIGER_PRECISION=auto PYTORCH_ALLOC_CONF=expandable_segments:True python main.py

# Full gated auto pipeline.
SIGER_TEXT_SOURCES=data/corpus/base_pretrain_text.txt PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode auto --dense-profile moe_dense_base --moe-profile small_moe --dense-max-steps 3000 --moe-max-steps 5000
```

`SIGER_MAX_STEPS` is the total target optimizer step, not "additional steps". Resume is dynamic: if `latest.json` is stale, the loader falls back to the newest `step_*.pt`; if no checkpoint exists, training starts fresh.

## Automatic LoRA Curriculum

For one-command instruction tuning from easier data to harder data:

```powershell
python train_pipeline.py --mode lora-curriculum
```

On Kaggle/CUDA, the same command can be launched with the allocator hint:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode lora-curriculum
```

LoRA stages are hardware-aware. If CUDA is available, SigerLM uses CUDA mixed precision and conservative VRAM-aware batch scaling. If the runtime falls back to CPU, the run is automatically capped to a small smoke/debug profile (`cpu_max_steps`, `cpu_max_samples`, `cpu_max_seq_len`) so a Kaggle notebook does not spend days on accidental FP32 CPU training.

For a deliberate full CPU run:

```bash
SIGER_ALLOW_CPU_FULL_TRAIN=1 PYTORCH_ALLOC_CONF=expandable_segments:True python train_pipeline.py --mode lora-curriculum
```

The curriculum definition lives in:

```txt
configs/training/lora_curriculum.json
```

Current stage order:

```txt
stage1_foundation -> stage2_general -> stage3_advanced -> stage4_full
```

Each stage can rebuild its corpus from `configs/datasets/curriculum_stage*.json`, run the matching LoRA config, merge the adapter into a normal checkpoint, then use that merged checkpoint as the base for the next stage.

Generated outputs:

```txt
data/corpus/curriculum_stage1_foundation_train.jsonl
data/corpus/curriculum_stage2_general_train.jsonl
data/corpus/curriculum_stage3_advanced_train.jsonl
data/corpus/curriculum_stage4_full_train.jsonl
checkpoints/lora/model_curriculum_stage1_foundation_merged.pt
checkpoints/lora/model_curriculum_stage2_general_merged.pt
checkpoints/lora/model_curriculum_stage3_advanced_merged.pt
checkpoints/lora/model_curriculum_stage4_full_merged.pt
logs/lora_curriculum/*.log
checkpoints/lora/curriculum_state.json
```

Useful flags:

```powershell
python train_pipeline.py --mode lora-curriculum --dry-run
python train_pipeline.py --mode lora-curriculum --no-rebuild-corpora
python train_pipeline.py --mode lora-curriculum --force-curriculum
```

The runner skips a stage when its `merged_output` already exists unless `--force-curriculum` is set.

For manual MoE experiments through `main.py`, adaptive sizing is enabled for `SIGER_MODEL_PROFILE=small_moe` by default. Disable it only when exact static expert counts are required:

```powershell
$env:SIGER_DISABLE_ADAPTIVE_MOE="1"
$env:SIGER_MODEL_PROFILE="small_moe"
python main.py
```

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
- `mined_parallel_jsonl`

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

For current curriculum runs, `max_seq_len=1024` is used from stage 2 onward because Kaggle/news rows truncate heavily at 512.

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

Curriculum LoRA:

```powershell
python train_pipeline.py --mode lora-curriculum
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

configs/training/curriculum_stage4_full_lora.json
  dataset_path: data/corpus/curriculum_stage4_full_train.jsonl
  max_steps: 2000
  max_seq_len: 1024
  merged_output: checkpoints/lora/model_curriculum_stage4_full_merged.pt
```

## Latest Verified Data Counts

```txt
data/corpus/lampung_instruction_train.jsonl: 30701 rows
data/corpus/general_instruction_train.jsonl: 30704 rows
data/corpus/kaggle_local_inputs_train.jsonl: 51969 rows
data/corpus/curriculum_stage1_foundation_train.jsonl: 84605 rows
data/corpus/curriculum_stage2_general_train.jsonl: 186672 rows
data/corpus/curriculum_stage3_advanced_train.jsonl: 218596 rows
data/corpus/curriculum_stage4_full_train.jsonl: 218596 rows
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
python -m py_compile train_pipeline.py
python lora\run_lora.py --help
python train_pipeline.py --mode lora-curriculum --dry-run
```
