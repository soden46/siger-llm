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
    safety_fraction: float = 0.70,
) -> int:
    """Conservative VRAM-aware batch suggestion.

    This is intentionally not an OOM-search loop. It uses currently free VRAM
    and a rough activation estimate so notebook runs do not thrash the GPU.
    """
    if not torch.cuda.is_available():
        return base_batch_size

    free_bytes, _ = torch.cuda.mem_get_info()
    safe_bytes = int(free_bytes * safety_fraction)
    d_inner = d_inner or d_model * 2
    # Conservative FP16/BF16 activation estimate for selective SSM.
    # It accounts for hidden states, expanded inner activations, gates,
    # B/C/delta projections, and scan state. The multiplier intentionally
    # leaves headroom because this is not an OOM probing loop.
    hidden_bytes = max_seq_len * d_model * 2
    inner_bytes = max_seq_len * d_inner * 2 * 6
    state_bytes = d_inner * max(1, d_state) * 2 * 2
    layer_bytes = hidden_bytes + inner_bytes + state_bytes
    bytes_per_sample = int(layer_bytes * max(1, n_layers) * 1.35)
    if bytes_per_sample <= 0:
        return base_batch_size

    suggested = max(base_batch_size, safe_bytes // bytes_per_sample)
    return int(max(base_batch_size, min(max_batch_size, suggested)))
