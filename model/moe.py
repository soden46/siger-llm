from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoE(nn.Module):
    """Small top-k feed-forward MoE for SigerLM blocks.

    This module is intentionally simple and local-device friendly. It keeps
    the dense SSM path intact, then adds sparse expert capacity as a residual
    branch. Only the selected top-k experts are evaluated for each token.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        router_jitter: float = 0.01,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        if top_k < 1 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.router_jitter = float(router_jitter)
        hidden_dim = int(d_model * hidden_mult)

        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, hidden_dim, bias=False),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, d_model, bias=False),
                )
                for _ in range(num_experts)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("last_aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_switch_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_importance_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("last_expert_load", torch.zeros(num_experts), persistent=False)
        self.register_buffer("last_expert_importance", torch.zeros(num_experts), persistent=False)
        self.register_buffer("last_dead_expert_fraction", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        flat_x = x.reshape(batch * seq_len, d_model)

        gate_logits = self.gate(flat_x)
        if self.training and self.router_jitter > 0:
            gate_logits = gate_logits + torch.randn_like(gate_logits) * self.router_jitter

        gate_probs = F.softmax(gate_logits, dim=-1)
        top_weights, top_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        flat_out = torch.zeros_like(flat_x)
        for expert_idx in range(self.num_experts):
            mask = top_indices == expert_idx
            if not mask.any():
                continue

            token_indices, route_slots = mask.nonzero(as_tuple=True)
            expert_input = flat_x.index_select(0, token_indices)
            expert_output = self.experts[expert_idx](expert_input)
            weights = top_weights[token_indices, route_slots].unsqueeze(-1)
            flat_out.index_add_(0, token_indices, expert_output * weights)

        self.last_aux_loss = self._load_balance_loss(gate_probs, top_indices)
        return self.dropout(flat_out.reshape(batch, seq_len, d_model))

    def _load_balance_loss(
        self,
        gate_probs: torch.Tensor,
        top_indices: torch.Tensor,
    ) -> torch.Tensor:
        importance = gate_probs.mean(dim=0)
        selected = F.one_hot(top_indices, num_classes=self.num_experts).float()
        load = selected.sum(dim=1).mean(dim=0) / float(self.top_k)

        uniform = torch.full_like(importance, 1.0 / float(self.num_experts))

        # Switch-style loss: penalizes mismatch between probability mass and
        # actual top-k assignments. Minimum is near 1.0 when routing is uniform.
        switch_loss = self.num_experts * torch.sum(importance * load.detach())

        # Direct differentiable pressure on router probabilities. This helps
        # early training before top-k assignments have become diverse.
        importance_loss = self.num_experts * torch.sum((importance - uniform).pow(2))

        self.last_switch_loss = switch_loss.detach()
        self.last_importance_loss = importance_loss.detach()
        self.last_expert_load = load.detach()
        self.last_expert_importance = importance.detach()
        self.last_dead_expert_fraction = (load == 0).float().mean().detach()

        return switch_loss + importance_loss


MoEBranch = SparseMoE
