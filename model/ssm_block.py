# model/ssm_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ssm_core import SSMCore
from model.norms import build_norm
from model.moe import SparseMoE


def _activation_fn(name: str):
    name = (name or "silu").lower()
    if name in {"silu", "swish"}:
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name}")


class SSMBlock(nn.Module):
    """
    Full Mamba block dengan gating mechanism.
    Analoginya ke Laravel: ini satu 'middleware' dalam pipeline.
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        d_inner = config.d_model * config.expand
        self.layer_idx = layer_idx

        self.norm = build_norm(
            config.d_model,
            norm_type=getattr(config, "norm_type", "rmsnorm"),
            eps=getattr(config, "norm_eps", 1e-6),
            bias=getattr(config, "norm_bias", False),
        )

        # Input projection: split jadi 2 branch
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)

        # Conv lokal untuk locality bias
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner  # depthwise conv
        )

        self.ssm = SSMCore(config)
        self.activation = _activation_fn(getattr(config, "activation", "silu"))

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.last_moe_loss = None

        moe_every = max(1, int(getattr(config, "moe_layers_every", 1)))
        self.use_moe = bool(getattr(config, "use_moe", False)) and ((layer_idx + 1) % moe_every == 0)
        if self.use_moe:
            self.moe_norm = build_norm(
                config.d_model,
                norm_type=getattr(config, "norm_type", "rmsnorm"),
                eps=getattr(config, "norm_eps", 1e-6),
                bias=getattr(config, "norm_bias", False),
            )
            self.moe = SparseMoE(
                config.d_model,
                num_experts=int(getattr(config, "moe_num_experts", 8)),
                top_k=int(getattr(config, "moe_top_k", 2)),
                hidden_mult=int(getattr(config, "moe_expert_hidden_mult", 2)),
                dropout=float(getattr(config, "dropout", 0.0)),
            )

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Split 2 branch: SSM branch & gate branch
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Conv lokal (B, L, D) → (B, D, L) → conv → (B, L, D)
        x_conv = self.conv1d(x_branch.transpose(1, 2))
        x_conv = x_conv[:, :, :x_branch.size(1)].transpose(1, 2)
        x_conv = self.activation(x_conv)

        # SSM
        y = self.ssm(x_conv)

        # Gating: multiply dengan sigmoid branch
        y = y * self.activation(z_gate)

        # Output projection + residual
        out = self.out_proj(y)
        out = self.dropout(out) + residual

        if self.use_moe:
            out = out + self.moe(self.moe_norm(out))
            self.last_moe_loss = self.moe.last_aux_loss
        else:
            self.last_moe_loss = None

        return out
