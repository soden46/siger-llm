# lora/layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    Drop-in replacement untuk nn.Linear dengan LoRA adapter.

    Forward:
        y = x @ W.T               ← frozen original weight
          + x @ A.T @ B.T * scale ← trainable LoRA path
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int   = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank         = rank
        self.scaling      = alpha / rank

        # ── Frozen original weight ────────────────────────
        self.weight = original_linear.weight  # reference, bukan copy
        self.bias   = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # ── Trainable LoRA matrices ───────────────────────
        # A: init dengan kaiming (bukan nol — biar ada gradient dari awal)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )
        # B: init dengan nol (output LoRA = 0 di awal, sama kayak pretrained)
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )

        self.lora_dropout = nn.Dropout(dropout)

        # Init A dengan kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Flag: apakah LoRA aktif
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = F.linear(x, self.weight, self.bias)

        if not self.enabled or self.rank == 0:
            return base_out

        # LoRA path: x → dropout → A → B → scale
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA ke weight asli → hapus overhead inference.
        Panggil ini setelah training selesai.
        W_merged = W + (B @ A) * scaling
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias is not None)
        merged.weight.data = (
            self.weight.data +
            (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.2f}")