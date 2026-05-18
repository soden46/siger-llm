from __future__ import annotations

import torch
import torch.distributed as dist


@torch.no_grad()
def evaluate_lm_loss(model, dataloader, *, device: str) -> dict[str, float]:
    """Evaluate LM loss and aggregate across distributed ranks when active."""
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0.0, device=device)

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)
        _, loss = model(x, targets=y)
        if loss.dim() > 0:
            loss = loss.mean()
        loss_sum += loss.detach()
        count += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    avg = (loss_sum / count.clamp_min(1)).item()
    model.train()
    return {"loss": avg, "batches": count.item()}
