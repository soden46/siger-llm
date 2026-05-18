from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import List, Optional


@dataclass
class LoRAConfig:
    # ── LoRA core ─────────────────────────────────────────
    rank: int = 32
    alpha: float = 32.0
    dropout: float = 0.05

    # Layer target LoRA pada arsitektur Mamba/SSM
    target_modules: List[str] = field(default_factory=lambda: [
        "in_proj",
        "out_proj",
        "x_proj",
        "dt_proj",
    ])

    # ── Training ──────────────────────────────────────────
    learning_rate: float = 1e-4
    max_steps: int = 2500
    batch_size: int = 8
    grad_accum: int = 4
    warmup_steps: int = 100
    max_seq_len: int = 512
    weight_decay: float = 0.01
    device: str = "auto"  # "auto", "cpu", or "cuda"
    prefer_gpu: bool = True
    distributed_strategy: str = "auto"
    sharded_checkpoint: bool = False
    elastic_recovery: bool = True
    resource_target_fraction: float = 0.8

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
    merged_output: str = "./checkpoints/lora/model_merged.pt"

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    @classmethod
    def from_json(cls, path: str | Path) -> "LoRAConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        valid_fields = set(cls.__dataclass_fields__.keys())
        unknown = sorted(set(data.keys()) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown LoRAConfig fields in {config_path}: {unknown}")

        return cls(**data)
