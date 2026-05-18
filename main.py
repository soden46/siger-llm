# main.py
import torch
from pathlib import Path
from config.model_config   import SigerConfig
from model.siger_model     import SigerLM
from tokenizer.hybrid_tokenizer import build_tokenizer
from training.dataset      import TextDataset
from training.trainer      import Trainer
from model.ssm_core import SSMCore

# ── Config ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Model
    # "vocab_size":    100277,
    # "d_model":       512,
    # "n_layers":      12,
    
    # Model kecil dulu buat smoke test
    "vocab_size": 100271,
    "d_model": 256,
    "n_layers": 8,

    # Training
    # "max_steps":     100_000,
    # "batch_size":    8,
    # "max_seq_len":   1024,
    # "max_seq_len":   1024,
    # "grad_accum_steps": 4,      # effective batch = 8 * 4 = 32

    # Training kecil
    "max_steps": 1500,
    "batch_size": 256,
    "auto_scale_batch": True,
    "max_auto_scale_factor": 2,
    "max_global_batch_size": 512,
    "max_seq_len": 32 ,
    "grad_accum_steps": 1,
    "device": "auto",
    "prefer_gpu": True,
    "multi_gpu": True,
    "distributed_strategy": "auto",
    "gradient_checkpointing": True,
    "sharded_checkpoint": False,
    "elastic_recovery": True,
    "auto_tune_batch_vram": False,
    "vram_safety_fraction": 0.70,

    # Optimizer
    # "max_lr":        3e-4,
    # "min_lr":        3e-5,
    # "warmup_steps":  2_000,
    # "weight_decay":  0.1,
    # "grad_clip":     1.0,

     # Optimizer
    "max_lr": 5e-4,
    "min_lr": 5e-5,
    "warmup_steps": 100,
    "weight_decay": 0.1,
    "grad_clip": 1.0,

    # Logging & saving
    # "log_interval":  10,
    # "save_every":    500,
    # "checkpoint_dir": "./checkpoints",
    # "num_workers":   2,

    # Logging & saving
    "log_interval": 10,
    "save_every": 500,
    "checkpoint_dir": "./checkpoints",
    "num_workers": 0,
}

def main():
    # 1. Tokenizer
    tok = build_tokenizer("auto")
    print(f"Tokenizer backend: {tok.backend} | vocab_size={tok.vocab_size}")

    # 2. Dataset — ganti dengan corpus lo
    sample_texts = []

    for path in Path("data").rglob("*.txt"):
        sample_texts.append(path.read_text(encoding="utf-8"))

    if not sample_texts:
        raise RuntimeError(
            "Tidak ada file .txt di folder data/. "
            "Buat dulu data/corpus.txt lalu isi teks panjang."
        )

    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=tok,
        max_seq_len=TRAIN_CONFIG["max_seq_len"]
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset menghasilkan 0 chunks. "
            "Isi data/corpus.txt harus lebih panjang atau kecilkan max_seq_len."
        )

    # 3. Model
    model_config = SigerConfig(
        vocab_size=tok.vocab_size,
        d_model=TRAIN_CONFIG["d_model"],
        n_layers=TRAIN_CONFIG["n_layers"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        gradient_checkpointing=TRAIN_CONFIG.get("gradient_checkpointing", False),
    )
    
    model = SigerLM(model_config)

    # 4. Trainer
    trainer = Trainer(model, TRAIN_CONFIG)
    trainer.train(dataset, resume=False)

if __name__ == "__main__":
    main()
