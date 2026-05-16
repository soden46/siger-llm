# lora/run_lora.py
from pathlib import Path
import re

import torch

from optimization.cpu.threading import configure_cpu
from config.model_config import SigerConfig
from model.siger_model import SigerLM
from tokenizer.tokenizer import MultilingualTokenizer
from lora.config import LoRAConfig
from lora.model import LoRAModel
from lora.dataset import InstructionDataset
from lora.trainer import LoRATrainer


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint_state(checkpoint_path: str) -> dict:
    """
    Support dua format:
    1. best_model.pt -> raw state_dict
    2. step_*.pt     -> dict dengan key 'model_state'
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Base checkpoint tidak ditemukan: {checkpoint_path}\n"
            "Pastikan lu sudah menjalankan base training dan punya checkpoints/best_model.pt"
        )

    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]

    return checkpoint


def infer_model_config_from_state_dict(state_dict: dict) -> SigerConfig:
    """
    Auto-infer konfigurasi model dari checkpoint.
    Ini penting karena smoke checkpoint lu bisa:
    - vocab_size=100271
    - d_model=64
    - n_layers=2

    Sementara target model besar nanti bisa berbeda.
    """
    if "embedding.weight" not in state_dict:
        raise RuntimeError(
            "Checkpoint tidak punya key embedding.weight, "
            "tidak bisa infer SigerConfig."
        )

    vocab_size, d_model = state_dict["embedding.weight"].shape

    layer_indices = set()

    for key in state_dict.keys():
        match = re.match(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

    n_layers = max(layer_indices) + 1 if layer_indices else 0

    if n_layers == 0:
        raise RuntimeError(
            "Tidak bisa infer n_layers dari checkpoint."
        )

    d_state = 16
    expand = 2
    d_conv = 4

    a_log_key = "layers.0.ssm.A_log"
    if a_log_key in state_dict:
        d_inner, d_state = state_dict[a_log_key].shape
        expand = max(1, d_inner // d_model)

    conv_key = "layers.0.conv1d.weight"
    if conv_key in state_dict:
        d_conv = state_dict[conv_key].shape[-1]

    config = SigerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        max_seq_len=128,
    )

    return config


def load_base_model(checkpoint_path: str) -> SigerLM:
    """
    Load base model dari best_model.pt / step checkpoint,
    lalu auto-reconstruct config dari shape checkpoint.
    """
    print("📦 Loading base checkpoint...")
    state_dict = load_checkpoint_state(checkpoint_path)

    model_config = infer_model_config_from_state_dict(state_dict)

    print("🧠 Inferred base model config:")
    print(f"   vocab_size : {model_config.vocab_size}")
    print(f"   d_model    : {model_config.d_model}")
    print(f"   n_layers   : {model_config.n_layers}")
    print(f"   d_state    : {model_config.d_state}")
    print(f"   d_conv     : {model_config.d_conv}")
    print(f"   expand     : {model_config.expand}")

    model = SigerLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("✅ Base model loaded successfully.")
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    configure_cpu(n_cores=2)

    # ── 1. Config LoRA Lampung ────────────────────────────
    lora_config = LoRAConfig(
        rank=8,
        alpha=16.0,
        dropout=0.05,

        target_modules=[
            "in_proj",
            "out_proj",
            "x_proj",
            "dt_proj",
        ],

        # Training awal CPU-safe
        learning_rate=2e-4,
        max_steps=300,
        batch_size=2,
        grad_accum=4,
        warmup_steps=20,
        max_seq_len=128,
        weight_decay=0.01,

        # Dataset Lampung hasil pipeline kita
        dataset_path="data/lampung/final/train_instruction.jsonl",
        max_samples=None,

        # Base checkpoint hasil training main.py
        base_checkpoint="./checkpoints/best_model.pt",

        # Save LoRA
        save_dir="./checkpoints/lora",
        save_every=100,
        log_interval=10,
    )

    # ── 2. Load base model ────────────────────────────────
    base_model = load_base_model(lora_config.base_checkpoint)

    # ── 3. Inject LoRA ────────────────────────────────────
    print("\n🧩 Injecting LoRA adapters...")
    lora_model = LoRAModel(base_model, lora_config)

    # ── 4. Load tokenizer + dataset Lampung ───────────────
    tokenizer = MultilingualTokenizer()

    dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=lora_config.dataset_path,
        max_seq_len=lora_config.max_seq_len,
        max_samples=lora_config.max_samples,
    )

    # ── 5. Train LoRA ─────────────────────────────────────
    trainer = LoRATrainer(
        lora_model=lora_model,
        config=lora_config,
        tokenizer=tokenizer,
    )

    trainer.train(dataset)

    # ── 6. Merge adapter ke base model ────────────────────
    print("\n🔀 Merging LoRA into base model...")

    merged_path = "./checkpoints/lora/model_lampung_merged.pt"
    lora_model.merge_and_export(merged_path)

    print("🎉 Done!")
    print(f"   LoRA adapters saved in : {lora_config.save_dir}")
    print(f"   Merged model saved at  : {merged_path}")


if __name__ == "__main__":
    main()