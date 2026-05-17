# Optimization

SigerLM targets modest hardware, especially CPU/VPS deployments. Optimization work is still experimental.

## Hardware Detection

```python
from optimization.hardware import detect_hardware, print_hardware_profile

hardware = detect_hardware(prefer_gpu=True)
print_hardware_profile(hardware)
```

`lora/run_lora.py` uses this to choose GPU when available and reduce CPU thread usage on low-RAM machines.

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
