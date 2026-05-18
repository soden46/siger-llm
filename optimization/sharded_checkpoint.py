from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist

from optimization.gpu import is_main_process, unwrap_model


def save_sharded_checkpoint(
    *,
    model,
    optimizer=None,
    scheduler=None,
    output_dir: str | Path,
    step: int,
    loss: float,
    config: dict,
) -> None:
    """Save an experimental distributed checkpoint directory.

    Uses `torch.distributed.checkpoint` when distributed is initialized.
    Falls back to a normal single-file state inside the output directory.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "model": unwrap_model(model).state_dict(),
        "step": step,
        "loss": loss,
        "config": config,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_step"] = getattr(scheduler, "current_step", 0)

    if dist.is_available() and dist.is_initialized():
        try:
            import torch.distributed.checkpoint as dcp

            dcp.save(state, checkpoint_id=str(output))
            return
        except Exception as exc:
            if is_main_process():
                print(f"Sharded checkpoint fallback to rank-0 torch.save: {exc}")

    if is_main_process():
        torch.save(state, output / "checkpoint.pt")
