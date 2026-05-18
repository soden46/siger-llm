from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    RMSNorm normalizes by root-mean-square without subtracting the mean. This is
    the normalization style used by many modern LLMs because it is stable and
    cheaper than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normed = normed * self.weight
        if self.bias is not None:
            normed = normed + self.bias
        return normed


def build_norm(dim: int, *, norm_type: str = "rmsnorm", eps: float = 1e-6, bias: bool = False) -> nn.Module:
    norm_type = (norm_type or "rmsnorm").lower()
    if norm_type in {"rmsnorm", "rms"}:
        return RMSNorm(dim, eps=eps, bias=bias)
    if norm_type in {"layernorm", "layer_norm", "ln"}:
        return nn.LayerNorm(dim, eps=eps, elementwise_affine=True, bias=bias)
    raise ValueError(f"Unsupported norm_type: {norm_type}")
