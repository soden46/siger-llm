from __future__ import annotations

import os
from dataclasses import replace

from lora.config import LoRAConfig
from optimization.hardware import HardwareProfile


def _cap(value: int, limit: int) -> int:
    return max(1, min(int(value), int(limit)))


def _cap_optional(value: int | None, limit: int | None) -> int | None:
    if limit is None:
        return value
    if value is None:
        return int(limit)
    return _cap(value, int(limit))


def apply_lora_hardware_policy(
    config: LoRAConfig,
    hardware: HardwareProfile,
) -> tuple[LoRAConfig, list[str]]:
    """Return a LoRA config scaled to the detected hardware.

    CPU fallback is intentionally a bounded smoke/debug run. Full SFT on CPU is
    too slow for Kaggle-style sessions and usually means the GPU quota is gone.
    Set SIGER_ALLOW_CPU_FULL_TRAIN=1 to keep the original config.
    """

    if not config.auto_scale_for_hardware:
        return config, ["hardware policy disabled by config"]

    changes: list[str] = []
    scaled = replace(config)

    if scaled.device == "auto":
        scaled.device = hardware.device
        changes.append(f"device auto -> {scaled.device}")

    if hardware.device == "cpu":
        if os.getenv("SIGER_ALLOW_CPU_FULL_TRAIN") == "1":
            changes.append("CPU full train allowed by SIGER_ALLOW_CPU_FULL_TRAIN=1")
            return scaled, changes

        before = replace(scaled)
        scaled.device = "cpu"
        scaled.precision = "fp32"
        scaled.distributed_strategy = "auto"
        scaled.auto_tune_batch_vram = False
        scaled.batch_size = _cap(scaled.batch_size, scaled.cpu_batch_size)
        scaled.grad_accum = max(int(scaled.grad_accum), int(scaled.cpu_grad_accum))
        scaled.max_seq_len = _cap(scaled.max_seq_len, scaled.cpu_max_seq_len)
        scaled.max_steps = _cap(scaled.max_steps, scaled.cpu_max_steps)
        scaled.max_samples = _cap_optional(scaled.max_samples, scaled.cpu_max_samples)
        scaled.save_every = _cap(scaled.save_every, max(1, min(scaled.cpu_save_every, scaled.max_steps)))
        scaled.max_dataloader_workers = 0
        scaled.resource_target_fraction = min(float(scaled.resource_target_fraction), 0.6)

        changes.extend(
            [
                "CPU fallback active: bounded smoke/debug run",
                f"max_steps {before.max_steps} -> {scaled.max_steps}",
                f"max_samples {before.max_samples} -> {scaled.max_samples}",
                f"max_seq_len {before.max_seq_len} -> {scaled.max_seq_len}",
                f"batch_size {before.batch_size} -> {scaled.batch_size}",
                f"grad_accum {before.grad_accum} -> {scaled.grad_accum}",
            ]
        )
        return scaled, changes

    if hardware.device == "cuda":
        scaled.device = "cuda"
        gpu_memory_gb = hardware.gpu_memory_gb or 0.0
        if gpu_memory_gb and gpu_memory_gb < 8.0:
            before_seq = scaled.max_seq_len
            before_global = scaled.max_global_batch_size
            scaled.max_seq_len = _cap(scaled.max_seq_len, scaled.low_vram_max_seq_len)
            scaled.max_global_batch_size = _cap(
                scaled.max_global_batch_size,
                scaled.low_vram_max_global_batch_size,
            )
            scaled.batch_size = _cap(scaled.batch_size, 1)
            scaled.grad_accum = max(int(scaled.grad_accum), 8)
            changes.extend(
                [
                    f"low VRAM policy active ({gpu_memory_gb:.1f} GB)",
                    f"max_seq_len {before_seq} -> {scaled.max_seq_len}",
                    f"max_global_batch_size {before_global} -> {scaled.max_global_batch_size}",
                ]
            )
        elif gpu_memory_gb and gpu_memory_gb < 14.0:
            before_global = scaled.max_global_batch_size
            scaled.max_global_batch_size = _cap(scaled.max_global_batch_size, 8)
            scaled.batch_size = _cap(scaled.batch_size, 2)
            changes.extend(
                [
                    f"mid VRAM policy active ({gpu_memory_gb:.1f} GB)",
                    f"max_global_batch_size {before_global} -> {scaled.max_global_batch_size}",
                ]
            )
        else:
            changes.append("CUDA policy: keep training config, VRAM batch auto-tune enabled")

    return scaled, changes
