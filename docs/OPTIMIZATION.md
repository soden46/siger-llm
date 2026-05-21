# Optimization

SigerLM targets modest hardware, especially CPU/VPS deployments. Optimization work is still experimental.

## Hardware Detection

```python
from optimization.hardware import detect_hardware, print_hardware_profile

hardware = detect_hardware(prefer_gpu=True)
print_hardware_profile(hardware)
```

`lora/run_lora.py` uses this to choose GPU when available, reduce CPU thread usage on low-RAM machines, and scale LoRA runs to the detected hardware.

## LoRA Hardware Policy

LoRA configs support `auto_scale_for_hardware=true` by default. The policy is applied by `lora/run_lora.py` and by the LoRA stage inside `train_pipeline.py`.

CUDA behavior:

```txt
>= 14GB VRAM  keep configured sequence length and let VRAM batch tuning work
8-14GB VRAM   cap global batch conservatively
< 8GB VRAM    cap max_seq_len to low_vram_max_seq_len and reduce batch pressure
```

CPU behavior:

```txt
device=cpu
precision=fp32
max_steps capped to cpu_max_steps
max_samples capped to cpu_max_samples
max_seq_len capped to cpu_max_seq_len
batch_size capped to cpu_batch_size
grad_accum raised to cpu_grad_accum
```

This prevents accidental full LoRA training on CPU-only Kaggle sessions. To intentionally run the original full CPU config, set:

```bash
SIGER_ALLOW_CPU_FULL_TRAIN=1 python lora/run_lora.py --config configs/training/general_lora.json
```

To disable the policy for one direct LoRA run:

```bash
python lora/run_lora.py --config configs/training/general_lora.json --no-hardware-policy
```

## Runtime Hardware Planner

Training uses `optimization/gpu.py` to choose a runtime strategy dynamically:

```txt
cpu            CPU-only fallback
single_gpu     one CUDA GPU
data_parallel  fallback single-process multi-GPU runtime
ddp            torchrun/DDP when WORLD_SIZE and LOCAL_RANK are present
```

The planner enables TF32 CUDA defaults, chooses conservative dataloader workers, uses pinned memory on CUDA, and unwraps `DataParallel` / `DistributedDataParallel` before saving checkpoints.

For Kaggle T4x2, `python main.py` auto-relaunches through `torchrun` and prefers `ddp`. `data_parallel` remains a fallback if `SIGER_DISABLE_AUTO_DDP=1` is set or auto-DDP is not available.

Mixed precision is selected automatically:

```txt
CUDA bf16-supported GPU  -> bf16
CUDA fp16 GPU, e.g. T4   -> fp16
CPU                      -> fp32
```

Base training can opt into conservative batch scaling and VRAM-aware batch tuning:

```json
{
  "auto_scale_batch": true,
  "max_auto_scale_factor": 2,
  "auto_tune_batch_vram": true,
  "max_global_batch_size": 128,
  "vram_safety_fraction": 0.75
}
```

This uses more available GPU capacity without allowing batch size to grow unbounded on larger machines.

Dataloader worker counts are auto-scaled with a CPU cap. In DDP/FSDP, workers are split per rank to avoid saturating Kaggle CPU. Persistent workers and prefetching are enabled when `num_workers > 0`.

Experimental cluster-scaling features now available:

- FSDP wrapping can be requested with `distributed_strategy="fsdp"` or `SIGER_DISTRIBUTED_STRATEGY=fsdp` under `torchrun`.
- Sharded checkpoint directories can be enabled with `sharded_checkpoint=true`.
- Graceful preemption handling can stop after the current optimizer step and still write final checkpoints.
- Gradient checkpointing can be enabled through `SigerConfig(gradient_checkpointing=True)`.
- Distributed validation aggregation is available through `optimization.distributed_validation.evaluate_lm_loss`.
- Conservative VRAM-aware batch suggestion is available through `auto_tune_batch_vram=true`.

These features are intentionally opt-in. The current state is best described as a **cluster-ready experimental stack**, not yet a fully managed production cluster platform.

Example FSDP launch:

```bash
SIGER_DISTRIBUTED_STRATEGY=fsdp torchrun --standalone --nproc_per_node=2 main.py
```

Example config switches:

```json
{
  "distributed_strategy": "fsdp",
  "gradient_checkpointing": true,
  "sharded_checkpoint": true,
  "elastic_recovery": true,
  "auto_tune_batch_vram": true,
  "vram_safety_fraction": 0.7
}
```

## CPU Constraints

Target baseline:

```txt
2 CPU cores
4GB RAM
CPU-only inference/training experiments
```

Recommended LoRA settings for small machines:

```txt
batch_size: 1
grad_accum: 4
max_seq_len: 256-384
```

## SSM Scan Memory Policy

`model/ssm_core.py` uses a streaming selective scan for full-sequence forward passes. The implementation intentionally avoids precomputing full `(B, L, D, N)` `dA`/`dB` tensors, because that can grow quickly with sequence length, inner dimension, and batch size. The preferred default keeps memory bounded around the recurrent state `(B, D, N)`, which matches the CPU/VPS target better.

Single-token decode/cache experiments should call `SSMCore.step(...)` with shape `(B, 1, D)`. Full prompts and training batches should use `forward(...)`.

## Quantization

Quantization modules live under:

```txt
optimization/quantization/
```

General idea:

```txt
FP32 -> INT8 or INT4
larger memory savings
possible quality loss
```

Treat current quantization numbers as experimental until benchmarked on the exact checkpoint.

## ONNX

ONNX export lives under:

```txt
optimization/onnx/export.py
```

ONNX Runtime is intended for CPU serving, but every exported checkpoint needs direct validation.

## Cache Work

Cache experiments live in:

```txt
optimization/kvcache.py
```

Because SigerLM is SSM-based, cache behavior is not identical to Transformer KV cache. Validate correctness before relying on speedups.

## Suggested Checks

```powershell
python -m py_compile optimization\hardware.py optimization\cpu\threading.py optimization\quantization\quantize.py optimization\onnx\export.py
```

## Current Priority

1. Keep training/inference stable on CPU.
2. Benchmark actual checkpoints before documenting speed claims.
3. Validate ONNX export on merged LoRA checkpoints.
4. Add repeatable benchmark scripts for general and Lampung routes.
