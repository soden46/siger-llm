# config/model_config.py
from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class SigerConfig:
    """
    Konfigurasi model SigerLM.

    'Siger' adalah mahkota adat Lampung — simbol keanggunan dan identitas.
    Model ini dirancang untuk mendukung bahasa-bahasa lokal Indonesia,
    khususnya Bahasa Lampung Dialek O dan Nyo.
    """

    # ── Vocabulary ────────────────────────────────────────
    vocab_size: int  = 100277       # tiktoken cl100k_base
                                    # smoke test pakai 100271

    # ── Model Dimensions ──────────────────────────────────
    d_model: int     = 512          # embedding / hidden dimension
    n_layers: int    = 12           # jumlah SSM blocks
    d_state: int     = 16           # SSM state dimension (N)
    d_conv: int      = 4            # depthwise conv kernel size
    expand: int      = 2            # d_inner = d_model * expand

    # ── SSM Specific ──────────────────────────────────────
    dt_rank: str     = "auto"       # "auto" = max(1, d_inner // 16)
    dt_min: float    = 0.001
    dt_max: float    = 0.1
    dt_init: str     = "random"
    dt_scale: float  = 1.0

    # ── Regularization ────────────────────────────────────
    dropout: float   = 0.1
    bias: bool       = False

    # ── Training ──────────────────────────────────────────
    max_seq_len: int  = 2048
    pad_token_id: int = 100258      # <|pad|>

    # ── Initialization ────────────────────────────────────
    initializer_range: float = 0.02
    residual_in_fp32: bool   = True

    def __post_init__(self):
        if self.dt_rank == "auto":
            d_inner = self.d_model * self.expand
            self.dt_rank = max(1, d_inner // 16)

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand

    @property
    def model_size_approx(self) -> str:
        d_inner = self.d_inner
        dt_rank = self.dt_rank if isinstance(self.dt_rank, int) else max(1, d_inner // 16)
        per_block = (
            self.d_model * d_inner * 2 +
            d_inner * self.d_conv +
            d_inner * (dt_rank + self.d_state * 2) +
            dt_rank * d_inner +
            d_inner * self.d_state +
            d_inner * self.d_model
        )
        embedding = self.vocab_size * self.d_model
        total     = embedding + (per_block * self.n_layers)
        if total < 1e6:
            return f"{total/1e3:.1f}K"
        elif total < 1e9:
            return f"{total/1e6:.1f}M"
        return f"{total/1e9:.1f}B"

    def save(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        path = Path(save_dir) / "model_config.json"
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        print(f"💾 Config saved: {path}")

    @classmethod
    def from_json(cls, path: str) -> "SigerConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def smoke(cls) -> "SigerConfig":
        """Config sangat kecil untuk smoke test & CI."""
        return cls(vocab_size=100271, d_model=64, n_layers=2, max_seq_len=128)

    @classmethod
    def tiny(cls) -> "SigerConfig":
        return cls(d_model=128, n_layers=4, max_seq_len=512)

    @classmethod
    def small(cls) -> "SigerConfig":
        return cls(d_model=256, n_layers=8, max_seq_len=1024)

    @classmethod
    def base(cls) -> "SigerConfig":
        return cls(d_model=512, n_layers=12, max_seq_len=2048)

    @classmethod
    def medium(cls) -> "SigerConfig":
        return cls(d_model=768, n_layers=16, max_seq_len=2048)

    def __repr__(self) -> str:
        return (
            f"SigerConfig(\n"
            f"  vocab_size    = {self.vocab_size:,}\n"
            f"  d_model       = {self.d_model}\n"
            f"  n_layers      = {self.n_layers}\n"
            f"  d_state       = {self.d_state}\n"
            f"  d_conv        = {self.d_conv}\n"
            f"  d_inner       = {self.d_inner}\n"
            f"  dt_rank       = {self.dt_rank}\n"
            f"  max_seq_len   = {self.max_seq_len}\n"
            f"  approx_params = {self.model_size_approx}\n"
            f")"
        )


# Backward-compat alias
MambaConfig = SigerConfig