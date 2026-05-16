# lora/config.py
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    # ── LoRA core ─────────────────────────────────────────
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05

    # Layer target LoRA pada arsitektur Mamba/SSM
    target_modules: List[str] = field(default_factory=lambda: [
        "in_proj",
        "out_proj",
        "x_proj",
        "dt_proj",
    ])

    # ── Training ──────────────────────────────────────────
    learning_rate: float = 2e-4
    max_steps: int = 300
    batch_size: int = 2
    grad_accum: int = 4
    warmup_steps: int = 20
    max_seq_len: int = 128
    weight_decay: float = 0.01

    # ── Dataset Lokal Lampung ─────────────────────────────
    dataset_path: str = "data/lampung/final/train_instruction.jsonl"

    # Masih disediakan kalau nanti mau pakai HuggingFace lagi
    dataset_name: Optional[str] = None
    dataset_split: str = "train"
    max_samples: Optional[int] = None

    # ── Base checkpoint ───────────────────────────────────
    base_checkpoint: str = "./checkpoints/best_model.pt"

    # ── Save ──────────────────────────────────────────────
    save_dir: str = "./checkpoints/lora"
    save_every: int = 100
    log_interval: int = 10

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank