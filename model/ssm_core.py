# model/ssm_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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
        dt_rank = config.dt_rank if isinstance(config.dt_rank, int) else max(1, d_inner // 16)
        self.dt_min = float(getattr(config, "dt_min", 0.001))
        self.dt_max = float(getattr(config, "dt_max", 0.1))
        self.dt_scale = float(getattr(config, "dt_scale", 1.0))
        self.x_proj = nn.Linear(d_inner, dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    def reset_dt_parameters(self):
        dt_rank = self.dt_proj.in_features
        dt_init_std = (dt_rank ** -0.5) * self.dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.dt_proj.out_features)
            * (math.log(self.dt_max) - math.log(self.dt_min))
            + math.log(self.dt_min)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def step(self, x_t: torch.Tensor, h: torch.Tensor | None = None):
        # x_t: (B, 1, d_inner)
        B, _, D = x_t.shape
        d_state = self.d_state

        if h is None:
            h = torch.zeros(B, D, d_state, device=x_t.device, dtype=x_t.dtype)

        A = -torch.exp(self.A_log.float()).to(x_t.dtype)

        x_proj = self.x_proj(x_t)
        dt_rank = self.dt_proj.in_features
        delta, B_mat, C_mat = x_proj.split([dt_rank, d_state, d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))

        dA = torch.exp(delta.squeeze(1).unsqueeze(-1) * A)
        dB = delta.squeeze(1).unsqueeze(-1) * B_mat.squeeze(1).unsqueeze(-2)

        x_now = x_t.squeeze(1).unsqueeze(-1)
        h = dA * h + dB * x_now

        y = (h * C_mat.squeeze(1).unsqueeze(-2)).sum(-1)
        y = y + x_t.squeeze(1) * self.D

        return y.unsqueeze(1), h

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

        if not torch.onnx.is_in_onnx_export():
            self._last_h = h.detach()

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D          # skip connection

        return y
