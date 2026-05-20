from __future__ import annotations

import torch


def suggest_cuda_batch_size(
    *,
    base_batch_size: int,
    max_batch_size: int,
    max_seq_len: int,
    d_model: int,
    d_inner: int | None = None,
    d_state: int = 16,
    n_layers: int = 1,
    safety_fraction: float = 0.65,
) -> int:
    """Conservative VRAM-aware batch suggestion for training/finetuning.

    This is intentionally not an OOM-search loop. It uses currently free VRAM
    on the active CUDA device and a rough training-memory estimate so notebook
    and DDP runs do not thrash the GPU.
    """
    if not torch.cuda.is_available():
        return base_batch_size

    try:
        current_device = torch.cuda.current_device()
        free_bytes, _ = torch.cuda.mem_get_info(current_device)
    except Exception:
        free_bytes, _ = torch.cuda.mem_get_info()

    safe_bytes = int(free_bytes * safety_fraction)
    d_inner = d_inner or d_model * 2

    # FP16/BF16 forward activation estimate for selective SSM. It accounts for
    # hidden states, expanded inner activations, gates, B/C/delta projections,
    # and scan state.
    hidden_bytes = max_seq_len * d_model * 2
    inner_bytes = max_seq_len * d_inner * 2 * 6
    state_bytes = d_inner * max(1, d_state) * 2 * 2
    layer_forward_bytes = hidden_bytes + inner_bytes + state_bytes
    total_forward_bytes = layer_forward_bytes * max(1, n_layers)

    # Training needs substantially more memory than inference: backward
    # activations, selective-scan intermediates/recompute, gradient tensors, and
    # AdamW state headroom. This is deliberately conservative.
    bytes_per_sample = int(total_forward_bytes * 2.5)
    if bytes_per_sample <= 0:
        return base_batch_size

    suggested = safe_bytes // bytes_per_sample
    if suggested > 4:
        suggested = (suggested // 4) * 4
    elif suggested > 2:
        suggested = (suggested // 2) * 2

    return int(max(base_batch_size, min(max_batch_size, suggested)))
