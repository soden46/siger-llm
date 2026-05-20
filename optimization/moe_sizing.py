from __future__ import annotations

from dataclasses import dataclass

from optimization.hardware import HardwareProfile


@dataclass(frozen=True)
class AdaptiveMoESettings:
    num_experts: int
    top_k: int
    layers_every: int
    reason: str


def resolve_adaptive_moe_settings(
    hardware: HardwareProfile,
    *,
    learning_loss: float | None = None,
    min_experts: int = 2,
    max_experts: int = 16,
) -> AdaptiveMoESettings:
    """Choose a conservative MoE shape for the current machine and checkpoint.

    The model graph still has a fixed expert count during one training stage.
    This resolver is meant to run at stage boundaries, especially when moving
    from dense SSM training to MoE expansion.
    """

    if min_experts < 1:
        raise ValueError("min_experts must be >= 1")
    if max_experts < min_experts:
        raise ValueError("max_experts must be >= min_experts")

    reasons: list[str] = []

    if hardware.device == "cuda":
        gpu_mem = hardware.gpu_memory_gb or 0.0
        if gpu_mem >= 24 or hardware.gpu_count >= 2:
            experts, top_k, layers_every = 16, 3, 1
            reasons.append("large CUDA budget")
        elif gpu_mem >= 16:
            experts, top_k, layers_every = 12, 2, 2
            reasons.append("mid/high CUDA budget")
        elif gpu_mem >= 8:
            experts, top_k, layers_every = 8, 2, 2
            reasons.append("standard CUDA budget")
        else:
            experts, top_k, layers_every = 4, 1, 2
            reasons.append("small CUDA budget")
    else:
        if hardware.ram_gb >= 16 and hardware.cpu_cores >= 8:
            experts, top_k, layers_every = 6, 1, 2
            reasons.append("strong CPU budget")
        elif hardware.ram_gb >= 8 and hardware.cpu_cores >= 4:
            experts, top_k, layers_every = 4, 1, 2
            reasons.append("moderate CPU budget")
        else:
            experts, top_k, layers_every = 2, 1, 4
            reasons.append("low CPU/RAM budget")

    if learning_loss is not None:
        if learning_loss <= 2.2:
            experts = int(round(experts * 1.5))
            top_k += 1
            layers_every = max(1, layers_every - 1)
            reasons.append("dense loss very mature")
        elif learning_loss <= 2.8:
            experts = int(round(experts * 1.25))
            reasons.append("dense loss mature")
        elif learning_loss > 4.5:
            experts = max(min_experts, experts // 2)
            top_k = 1
            layers_every = min(4, layers_every + 1)
            reasons.append("dense loss still unstable")
        else:
            reasons.append("dense loss passed expansion gate")

    experts = max(min_experts, min(max_experts, int(experts)))
    top_k = max(1, min(int(top_k), experts))
    layers_every = max(1, int(layers_every))

    return AdaptiveMoESettings(
        num_experts=experts,
        top_k=top_k,
        layers_every=layers_every,
        reason=", ".join(reasons),
    )
