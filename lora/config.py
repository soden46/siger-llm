# lora/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    # ── LoRA core ─────────────────────────────────────────
    rank: int           = 8       # r: makin gede makin expressive, makin berat
    alpha: float        = 16.0    # scaling: alpha/rank = scaling factor
    dropout: float      = 0.05    # regularization

    # Layer mana yang di-inject LoRA
    # Buat SSM: target projection layers
    target_modules: List[str] = field(default_factory=lambda: [
        "in_proj",     # input projection
        "out_proj",    # output projection
        "x_proj",      # SSM x projection
        "dt_proj",     # delta projection
    ])

    # ── Training ──────────────────────────────────────────
    learning_rate:  float = 2e-4   # LoRA lr bisa lebih gede dari full finetune
    max_steps:      int   = 5_000
    batch_size:     int   = 4
    grad_accum:     int   = 8      # effective batch = 32
    warmup_steps:   int   = 100
    max_seq_len:    int   = 512
    weight_decay:   float = 0.01

    # ── Dataset ───────────────────────────────────────────
    dataset_name:   str   = "HuggingFaceH4/ultrachat_200k"
    dataset_split:  str   = "train_sft"
    max_samples:    int   = 50_000  # ambil sebagian dulu

    # ── Save ──────────────────────────────────────────────
    save_dir:       str   = "./checkpoints/lora"
    save_every:     int   = 500
    log_interval:   int   = 10

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank