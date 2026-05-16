# model/ssm_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SSMCore(nn.Module):
    """
    Implementasi discrete SSM:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state  # N
        d_inner = config.d_model * config.expand

        # Learnable matrices
        # A: state transition (d_inner, d_state)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1)
            .float().unsqueeze(0).repeat(d_inner, 1))
        )

        # D: skip connection (residual)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Projections untuk B, C, delta (input-dependent = selective!)
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    def forward(self, x):
        # x shape: (batch, seq_len, d_inner)
        B, L, D = x.shape
        d_state = self.d_state

        # Compute A (always negative untuk stability)
        A = -torch.exp(self.A_log.float())  # (D, N)

        # Compute input-dependent B, C, delta (ini yg bikin "selective")
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt_rank = self.dt_proj.in_features
        delta, B_mat, C_mat = x_proj.split([dt_rank, d_state, d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))  # (B, L, D) — step size

        # Discretize: ZOH (Zero-Order Hold)
        # A_bar = exp(delta * A)
        # B_bar = (A_bar - I) * inv(A) * B ≈ delta * B (simplified)
        dA = torch.exp(delta.unsqueeze(-1) * A)          # (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)   # (B, L, D, N)

        # Selective scan — recurrent state update
        h = torch.zeros(B, D, d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C_mat[:, t].unsqueeze(-2)).sum(-1)  # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D          # skip connection

        return y