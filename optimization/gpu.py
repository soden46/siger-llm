from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
try:
    from torch.distributed.fsdp import FullyShardedDataParallel
except ImportError:  # pragma: no cover - depends on torch build
    FullyShardedDataParallel = None


@dataclass(frozen=True)
class RuntimePlan:
    device: str
    device_count: int
    strategy: str
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    dataloader_workers: int = 0
    pin_memory: bool = False

    @property
    def is_distributed(self) -> bool:
        return self.strategy in {"ddp", "fsdp"}

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def configure_cuda_runtime() -> None:
    """Enable fast-but-safe CUDA defaults for modern NVIDIA GPUs."""
    if not torch.cuda.is_available():
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _distributed_env_present() -> bool:
    return _env_int("WORLD_SIZE", 1) > 1 and "LOCAL_RANK" in os.environ


def build_runtime_plan(
    *,
    prefer_gpu: bool = True,
    requested_device: str = "auto",
    cpu_cores: int = 1,
    max_workers: int | None = None,
    strategy: str = "auto",
) -> RuntimePlan:
    if cpu_cores <= 1:
        cpu_cores = os.cpu_count() or 1
    if requested_device == "auto":
        device = "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"
    else:
        device = requested_device

    if device != "cuda" or not torch.cuda.is_available():
        workers = max(0, min(max_workers if max_workers is not None else 2, max(cpu_cores - 1, 0)))
        return RuntimePlan(
            device="cpu",
            device_count=0,
            strategy="cpu",
            dataloader_workers=workers,
            pin_memory=False,
        )

    configure_cuda_runtime()
    device_count = torch.cuda.device_count()
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    world_size = _env_int("WORLD_SIZE", 1)
    requested_strategy = os.environ.get("SIGER_DISTRIBUTED_STRATEGY", strategy).lower()
    if _distributed_env_present():
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        strategy = "fsdp" if requested_strategy == "fsdp" else "ddp"
        effective_device_count = world_size
    elif device_count > 1:
        strategy = "data_parallel"
        effective_device_count = device_count
    else:
        strategy = "single_gpu"
        effective_device_count = 1

    default_workers = 2 if cpu_cores >= 4 else 1
    workers = min(max_workers if max_workers is not None else default_workers, max(cpu_cores - 1, 0))
    return RuntimePlan(
        device="cuda",
        device_count=effective_device_count,
        strategy=strategy,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        dataloader_workers=max(0, workers),
        pin_memory=True,
    )


def print_runtime_plan(plan: RuntimePlan) -> None:
    if not plan.is_main_process:
        return
    print("Runtime plan")
    print(f"  strategy  : {plan.strategy}")
    print(f"  devices   : {plan.device_count}")
    print(f"  workers   : {plan.dataloader_workers}")
    print(f"  pin memory: {plan.pin_memory}")


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    while isinstance(model, (nn.DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def wrap_model_for_runtime(model: nn.Module, plan: RuntimePlan, *, enabled: bool = True) -> nn.Module:
    if not enabled:
        return model
    if plan.strategy == "fsdp":
        if FullyShardedDataParallel is None:
            raise RuntimeError("FSDP is not available in this PyTorch build.")
        print(f"Using FSDP rank {plan.rank}/{plan.world_size} on cuda:{plan.local_rank}")
        return FullyShardedDataParallel(model)
    if plan.strategy == "ddp":
        print(f"Using DDP rank {plan.rank}/{plan.world_size} on cuda:{plan.local_rank}")
        return DistributedDataParallel(model, device_ids=[plan.local_rank], output_device=plan.local_rank)
    if plan.strategy == "data_parallel":
        device_ids = list(range(torch.cuda.device_count()))
        names = [torch.cuda.get_device_name(index) for index in device_ids]
        print(f"Using DataParallel on {len(device_ids)} GPUs: {', '.join(names)}")
        return nn.DataParallel(model, device_ids=device_ids)
    return model


def maybe_wrap_data_parallel(
    model: nn.Module,
    *,
    device: str,
    enabled: bool = True,
) -> nn.Module:
    plan = build_runtime_plan(requested_device=device)
    return wrap_model_for_runtime(model, plan, enabled=enabled)
