# model/ssm_core.py
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSMCore(nn.Module):
    """
    Implementasi discrete selective SSM:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t
    """

    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state
        d_inner = config.d_model * config.expand
        self.d_inner = d_inner

        # S4D-style diagonal state initialization: A = -diag(1, 2, ..., N).
        A_init = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.D = nn.Parameter(torch.ones(d_inner))

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

    def step(self, x_t: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-token selective scan update for cached generation.
        """
        B, L, D = x_t.shape
        if L != 1:
            raise ValueError(f"SSMCore.step expects a single token, got sequence length {L}.")

        d_state = self.d_state

        if h is None:
            h = torch.zeros(B, D, d_state, device=x_t.device, dtype=x_t.dtype)

        A = -torch.exp(self.A_log.float()).to(x_t.dtype)
        x_now = x_t[:, 0]
        x_proj = self.x_proj(x_now)
        dt_rank = self.dt_proj.in_features
        delta, B_mat, C_mat = x_proj.split([dt_rank, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        dA = torch.exp(delta.unsqueeze(-1) * A)
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)

        h = dA * h + dB * x_now.unsqueeze(-1)
        y = (h * C_mat.unsqueeze(-2)).sum(-1)
        y = y + x_now * self.D
        return y.unsqueeze(1), h

    def forward(self, x):
        B, L, D = x.shape
        d_state = self.d_state

        A = -torch.exp(self.A_log.float()).to(x.dtype)
        x_proj = self.x_proj(x)
        dt_rank = self.dt_proj.in_features
        delta, B_mat, C_mat = x_proj.split([dt_rank, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # Streaming selective scan keeps memory bounded at (B, D, N). Materializing
        # full (B, L, D, N) tensors is faster in small cases but risky on CPU/VPS.
        h = torch.zeros(B, D, d_state, device=x.device, dtype=x.dtype)
        ys = []

        for t in range(L):
            delta_t = delta[:, t]
            b_t = B_mat[:, t]
            c_t = C_mat[:, t]
            dA_t = torch.exp(delta_t.unsqueeze(-1) * A)
            dB_t = delta_t.unsqueeze(-1) * b_t.unsqueeze(-2)
            h = dA_t * h + dB_t * x[:, t].unsqueeze(-1)
            y = (h * c_t.unsqueeze(-2)).sum(-1)
            ys.append(y)

        if not torch.onnx.is_in_onnx_export():
            self._last_h = h.detach()

        y = torch.stack(ys, dim=1)
        y = y + x * self.D
        return y
