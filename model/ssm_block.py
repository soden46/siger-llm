# model/ssm_block.py
import torch.nn as nn

class SSMBlock(nn.Module):
    """
    Full Mamba block dengan gating mechanism.
    Analoginya ke Laravel: ini satu 'middleware' dalam pipeline.
    """
    def __init__(self, config):
        super().__init__()
        d_inner = config.d_model * config.expand

        self.norm = nn.LayerNorm(config.d_model)

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

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Split 2 branch: SSM branch & gate branch
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Conv lokal (B, L, D) → (B, D, L) → conv → (B, L, D)
        x_conv = self.conv1d(x_branch.transpose(1, 2))
        x_conv = x_conv[:, :, :x_branch.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM
        y = self.ssm(x_conv)

        # Gating: multiply dengan sigmoid branch
        y = y * F.silu(z_gate)

        # Output projection + residual
        out = self.out_proj(y)
        return self.dropout(out) + residual