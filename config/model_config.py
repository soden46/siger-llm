# config/model_config.py
from dataclasses import dataclass

@dataclass
class SigerConfig:
    vocab_size: int = 32000
    d_model: int = 512        # dimension utama
    n_layers: int = 12        # jumlah SSM block
    d_state: int = 16         # ukuran state SSM (N)
    d_conv: int = 4           # kernel conv lokal
    expand: int = 2           # expansion factor
    dt_rank: str = "auto"     # rank untuk delta
    dropout: float = 0.1
    max_seq_len: int = 2048