import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.model_config import SigerConfig
from lora.config import LoRAConfig
from lora.dataset import InstructionDataset
from lora.model import LoRAModel
from lora.trainer import LoRATrainer
from model.siger_model import SigerLM
from optimization.cpu.threading import configure_cpu
from optimization.gpu import barrier, cleanup_distributed, is_main_process, maybe_relaunch_with_torchrun
from optimization.hardware import detect_hardware, print_hardware_profile
from tokenizer.hybrid_tokenizer import build_tokenizer


def load_checkpoint_state(checkpoint_path: str) -> dict:
    path = Path(checkpoint_path)
    if not path.exists():
        latest_meta = path.parent / "latest.json"
        if path.name == "best_model.pt" and latest_meta.exists():
            with latest_meta.open("r", encoding="utf-8") as f:
                latest = json.load(f)
            fallback = path.parent / str(latest["latest"])
            if fallback.exists():
                print(
                    f"Base checkpoint {checkpoint_path} tidak ada; "
                    f"fallback ke latest checkpoint: {fallback}"
                )
                path = fallback

    if not path.exists():
        raise FileNotFoundError(
            f"Base checkpoint tidak ditemukan: {checkpoint_path}\n"
            "Pastikan base training sudah menghasilkan checkpoints/best_model.pt "
            "atau checkpoints/latest.json."
        )

    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    return checkpoint


def infer_model_config_from_state_dict(state_dict: dict) -> SigerConfig:
    if "embedding.weight" not in state_dict:
        raise RuntimeError("Checkpoint tidak punya key embedding.weight.")

    vocab_size, d_model = state_dict["embedding.weight"].shape
    layer_indices = set()

    for key in state_dict.keys():
        match = re.match(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

    n_layers = max(layer_indices) + 1 if layer_indices else 0
    if n_layers == 0:
        raise RuntimeError("Tidak bisa infer n_layers dari checkpoint.")

    d_state = 16
    expand = 2
    d_conv = 4

    if "layers.0.ssm.A_log" in state_dict:
        d_inner, d_state = state_dict["layers.0.ssm.A_log"].shape
        expand = max(1, d_inner // d_model)

    if "layers.0.conv1d.weight" in state_dict:
        d_conv = state_dict["layers.0.conv1d.weight"].shape[-1]

    return SigerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        max_seq_len=512,
    )


def load_base_model(checkpoint_path: str) -> SigerLM:
    print("Loading base checkpoint...")
    state_dict = load_checkpoint_state(checkpoint_path)
    model_config = infer_model_config_from_state_dict(state_dict)

    print("Inferred base model config:")
    print(f"   vocab_size : {model_config.vocab_size}")
    print(f"   d_model    : {model_config.d_model}")
    print(f"   n_layers   : {model_config.n_layers}")
    print(f"   d_state    : {model_config.d_state}")
    print(f"   d_conv     : {model_config.d_conv}")
    print(f"   expand     : {model_config.expand}")

    model = SigerLM(model_config)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Base model loaded successfully.")
    return model


def default_lora_config(device: str) -> LoRAConfig:
    return LoRAConfig(
        rank=16,
        alpha=16.0,
        dropout=0.05,
        target_modules=[
            "in_proj",
            "out_proj",
            "x_proj",
            "dt_proj",
        ],
        learning_rate=5e-5,
        max_steps=1500,
        batch_size=256,
        grad_accum=1,
        warmup_steps=100,
        max_seq_len=32,
        weight_decay=0.01,
        device=device,
        prefer_gpu=True,
        distributed_strategy="auto",
        precision="auto",
        max_dataloader_workers=2,
        auto_tune_batch_vram=True,
        max_global_batch_size=16,
        vram_safety_fraction=0.75,
        resource_target_fraction=0.8,
        dataset_path="data/lampung/final/train_augmented_instruction.jsonl",
        max_samples=None,
        base_checkpoint="./checkpoints/best_model.pt",
        save_dir="./checkpoints/lora/lampung",
        save_every=250,
        log_interval=10,
        merged_output="./checkpoints/lora/model_lampung_merged.pt",
    )


def load_lora_config(config_path: str | None, device: str) -> LoRAConfig:
    if not config_path:
        return default_lora_config(device)

    config = LoRAConfig.from_json(config_path)
    if config.device == "auto":
        config.device = device
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a SigerLM LoRA adapter.")
    parser.add_argument(
        "--config",
        help="Optional LoRA training config JSON, e.g. configs/training/general_lora.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hardware = detect_hardware(prefer_gpu=True)
    print_hardware_profile(hardware)

    if hardware.device == "cpu":
        n_cores = 1 if hardware.ram_gb < 3.0 else min(2, hardware.cpu_cores)
        configure_cpu(n_cores=n_cores)

    lora_config = load_lora_config(args.config, hardware.device)
    maybe_relaunch_with_torchrun(
        script_path=Path(__file__).resolve(),
        argv=sys.argv[1:],
        strategy=lora_config.distributed_strategy,
        enabled=(lora_config.device == "cuda" and lora_config.prefer_gpu),
    )
    print(f"Training dataset: {lora_config.dataset_path or lora_config.dataset_name}")
    print(f"Save dir: {lora_config.save_dir}")

    base_model = load_base_model(lora_config.base_checkpoint)

    print("\nInjecting LoRA adapters...")
    lora_model = LoRAModel(base_model, lora_config)

    tokenizer = build_tokenizer("auto")
    print(f"Tokenizer backend: {tokenizer.backend} | vocab_size={tokenizer.vocab_size}")

    if tokenizer.vocab_size != base_model.config.vocab_size:
        raise RuntimeError(
            "Tokenizer vocab_size tidak cocok dengan base checkpoint. "
            f"tokenizer={tokenizer.vocab_size}, model={base_model.config.vocab_size}. "
            "Pakai tokenizer yang sama dengan saat base model dilatih, atau retrain base model."
        )

    dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=lora_config.dataset_path,
        dataset_name=lora_config.dataset_name,
        split=lora_config.dataset_split,
        max_seq_len=lora_config.max_seq_len,
        max_samples=lora_config.max_samples,
    )

    trainer = LoRATrainer(
        lora_model=lora_model,
        config=lora_config,
        tokenizer=tokenizer,
    )
    trainer.train(dataset)

    barrier()
    if is_main_process():
        print("\nMerging LoRA into base model...")
        lora_model.merge_and_export(lora_config.merged_output)

        print("Done.")
        print(f"   LoRA adapters saved in : {lora_config.save_dir}")
        print(f"   Merged model saved at  : {lora_config.merged_output}")

    barrier()
    cleanup_distributed()


if __name__ == "__main__":
    main()
