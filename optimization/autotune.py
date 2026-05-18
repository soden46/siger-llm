from __future__ import annotations

import torch


def suggest_cuda_batch_size(
    *,
    base_batch_size: int,
    max_batch_size: int,
    max_seq_len: int,
    d_model: int,
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
    # Rough FP16 activation + optimizer overhead estimate for this small SSM.
    bytes_per_sample = max_seq_len * d_model * 2 * 96
    if bytes_per_sample <= 0:
        return base_batch_size

    suggested = max(base_batch_size, safe_bytes // bytes_per_sample)
    return int(max(base_batch_size, min(max_batch_size, suggested)))
