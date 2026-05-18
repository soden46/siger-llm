# main.py
import sys
import os
import torch
from pathlib import Path
from config.model_config   import SigerConfig
from model.siger_model     import SigerLM
from tokenizer.hybrid_tokenizer import build_tokenizer
from training.dataset      import TextDataset
from training.text_sources import resolve_text_sources
from training.trainer      import Trainer
from model.ssm_core import SSMCore
from optimization.gpu import maybe_relaunch_with_torchrun


MODEL_PROFILES = {
    "small": {"d_model": 256, "n_layers": 8, "max_seq_len": 128},
    "base": {"d_model": 512, "n_layers": 12, "max_seq_len": 256},
    "reasoning_base": {"d_model": 512, "n_layers": 12, "max_seq_len": 512},
}

# ── Config ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Model
    # "vocab_size":    100277,
    # "d_model":       512,
    # "n_layers":      12,
    
    # Model kecil dulu buat smoke test
    "model_profile": "small",
    "vocab_size": 100271,
    "d_model": 256,
    "n_layers": 8,
    "d_state": 16,
    "expand": 2,
    "d_conv": 4,
    "activation": "silu",
    "norm_type": "rmsnorm",
    "norm_eps": 1e-6,
    "norm_bias": False,
    "initializer_range": 0.02,
    "residual_scale_init": True,

    # Training
    # "max_steps":     100_000,
    # "batch_size":    8,
    # "max_seq_len":   1024,
    # "max_seq_len":   1024,
    # "grad_accum_steps": 4,      # effective batch = 8 * 4 = 32

    # Training kecil
    "max_steps": 1500,
    "batch_size": 16,
    "auto_scale_batch": True,
    "max_auto_scale_factor": 2,
    "max_global_batch_size": 64,
    "max_per_device_batch_size": 16,
    "max_seq_len": 128 ,
    "grad_accum_steps": 4,
    "max_chars_per_text_file": 8_000_000,
    "max_dataset_chunks": 200_000,
    "text_sources": ["data"],
    "text_include": "*.txt",
    "text_exclude": [
        "data/tmp/*",
        "data/**/*.report.txt",
    ],
    "max_text_files": None,
    "device": "auto",
    "prefer_gpu": True,
    "multi_gpu": True,
    "precision": "auto",
    "resource_target_fraction": 0.8,
    "distributed_strategy": "auto",
    "gradient_checkpointing": False,
    "sharded_checkpoint": False,
    "elastic_recovery": True,
    "auto_tune_batch_vram": True,
    "vram_safety_fraction": 0.75,
    "torch_compile": False,
    "torch_compile_mode": "reduce-overhead",

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
    "save_every": 250,
    "checkpoint_dir": "./checkpoints",
    "num_workers": "auto",
    "max_dataloader_workers": 2,
    "prefetch_factor": 2,
}

def main():
    profile_name = os.environ.get("SIGER_MODEL_PROFILE", TRAIN_CONFIG.get("model_profile", "small")).lower()
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unknown SIGER_MODEL_PROFILE={profile_name}. Choose: {', '.join(MODEL_PROFILES)}")
    TRAIN_CONFIG.update(MODEL_PROFILES[profile_name])
    print(f"Model profile: {profile_name}")

    maybe_relaunch_with_torchrun(
        script_path=Path(__file__).resolve(),
        argv=sys.argv[1:],
        strategy=TRAIN_CONFIG.get("distributed_strategy", "auto"),
        enabled=TRAIN_CONFIG.get("multi_gpu", True),
    )

    # 1. Tokenizer
    tok = build_tokenizer("auto")
    print(f"Tokenizer backend: {tok.backend} | vocab_size={tok.vocab_size}")
    TRAIN_CONFIG["vocab_size"] = tok.vocab_size

    # 2. Dataset sources
    text_paths = resolve_text_sources(TRAIN_CONFIG)

    if not text_paths:
        raise RuntimeError(
            "Tidak ada text source untuk base training. "
            "Set SIGER_TEXT_SOURCES ke file/folder/glob, atau isi TRAIN_CONFIG['text_sources']."
        )

    print(f"Text files: {len(text_paths)}")
    for path in text_paths[:10]:
        print(f"  - {path}")
    if len(text_paths) > 10:
        print(f"  ... +{len(text_paths) - 10} more")
    dataset = TextDataset.from_text_files(
        paths=text_paths,
        tokenizer=tok,
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        max_chars_per_file=TRAIN_CONFIG.get("max_chars_per_text_file"),
        max_chunks=TRAIN_CONFIG.get("max_dataset_chunks"),
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Dataset menghasilkan 0 chunks. "
            "Isi data/corpus.txt harus lebih panjang atau kecilkan max_seq_len."
        )

    # 3. Model
    model_config = SigerConfig(
        vocab_size=TRAIN_CONFIG["vocab_size"],
        d_model=TRAIN_CONFIG["d_model"],
        n_layers=TRAIN_CONFIG["n_layers"],
        d_state=TRAIN_CONFIG.get("d_state", 16),
        d_conv=TRAIN_CONFIG.get("d_conv", 4),
        expand=TRAIN_CONFIG.get("expand", 2),
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
        activation=TRAIN_CONFIG.get("activation", "silu"),
        norm_type=TRAIN_CONFIG.get("norm_type", "rmsnorm"),
        norm_eps=TRAIN_CONFIG.get("norm_eps", 1e-6),
        norm_bias=TRAIN_CONFIG.get("norm_bias", False),
        initializer_range=TRAIN_CONFIG.get("initializer_range", 0.02),
        residual_scale_init=TRAIN_CONFIG.get("residual_scale_init", True),
        gradient_checkpointing=TRAIN_CONFIG.get("gradient_checkpointing", False),
    )
    TRAIN_CONFIG["d_inner"] = model_config.d_inner
    TRAIN_CONFIG["d_state"] = model_config.d_state
    TRAIN_CONFIG["expand"] = model_config.expand
    
    model = SigerLM(model_config)
    compile_requested = (
        os.environ.get("SIGER_TORCH_COMPILE", "0") == "1"
        or bool(TRAIN_CONFIG.get("torch_compile", False))
    )
    if compile_requested and hasattr(torch, "compile"):
        print(f"Compiling model with torch.compile(mode={TRAIN_CONFIG.get('torch_compile_mode', 'reduce-overhead')})")
        model = torch.compile(model, mode=TRAIN_CONFIG.get("torch_compile_mode", "reduce-overhead"))

    # 4. Trainer
    trainer = Trainer(model, TRAIN_CONFIG)
    trainer.train(dataset, resume=False)

if __name__ == "__main__":
    main()
