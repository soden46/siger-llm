# training/optimizer.py
import torch
import math
from torch.optim import AdamW

def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.1):
    """
    AdamW dengan weight decay selective.
    Parameter 1D (bias, layernorm) → NO weight decay.
    Parameter 2D+ (weight matrices) → weight decay.
    Ini trick standar dari GPT-2 paper.
    """
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:  # bias, layernorm weights
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model params: {total:,} total | {trainable:,} trainable")

    return optimizer


class CosineScheduler:
    """
    Cosine decay dengan warmup.
    Analoginya: panas dulu (warmup) → dingin pelan-pelan (cosine).
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,   # biasanya 10% dari max_lr
    ):
        self.optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps   = max_steps
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        step = self.current_step

        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Phase 2: Cosine decay
        if step >= self.max_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self):
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_step += 1
        return lr
    