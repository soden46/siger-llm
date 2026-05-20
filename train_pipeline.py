from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from config.model_config import SigerConfig
from lora.config import LoRAConfig
from lora.dataset import InstructionDataset
from lora.model import LoRAModel
from lora.run_lora import infer_model_config_from_state_dict, load_checkpoint_state
from lora.trainer import LoRATrainer
from main import MODEL_PROFILES, TRAIN_CONFIG
from model.siger_model import SigerLM
from optimization.cpu.threading import configure_cpu
from optimization.gpu import barrier, cleanup_distributed, is_main_process, maybe_relaunch_with_torchrun
from optimization.hardware import detect_hardware, print_hardware_profile
from optimization.moe_sizing import resolve_adaptive_moe_settings
from tokenizer.hybrid_tokenizer import build_tokenizer
from training.dataset import TextDataset
from training.text_sources import resolve_text_sources
from training.trainer import Trainer


@dataclass
class PipelineConfig:
    dense_loss_threshold: float = 3.5
    dense_min_steps: int = 1500
    dense_max_steps: int = 3000
    moe_max_steps: int = 5000
    moe_plateau_delta: float = 0.005
    moe_min_checkpoints: int = 2
    dense_checkpoint_dir: str = "./checkpoints/auto/dense"
    moe_checkpoint_dir: str = "./checkpoints/auto/moe"
    state_path: str = "./checkpoints/auto/pipeline_state.json"
    lora_config: str = "./configs/training/general_lora.json"
    force_stage: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic SigerLM training pipeline: dense -> MoE -> LoRA."
    )
    parser.add_argument("--dense-loss-threshold", type=float, default=3.5)
    parser.add_argument("--dense-min-steps", type=int, default=1500)
    parser.add_argument("--dense-max-steps", type=int, default=3000)
    parser.add_argument("--moe-max-steps", type=int, default=5000)
    parser.add_argument("--moe-plateau-delta", type=float, default=0.005)
    parser.add_argument("--moe-min-checkpoints", type=int, default=2)
    parser.add_argument("--dense-checkpoint-dir", default="./checkpoints/auto/dense")
    parser.add_argument("--moe-checkpoint-dir", default="./checkpoints/auto/moe")
    parser.add_argument("--state-path", default="./checkpoints/auto/pipeline_state.json")
    parser.add_argument("--lora-config", default="./configs/training/general_lora.json")
    parser.add_argument(
        "--force-stage",
        choices=["dense", "moe", "lora"],
        help="Run one stage regardless of metric gates.",
    )
    return parser.parse_args()


def load_latest_meta(checkpoint_dir: str | Path) -> dict[str, Any] | None:
    meta_path = Path(checkpoint_dir) / "latest.json"
    if not meta_path.exists():
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def latest_checkpoint_path(checkpoint_dir: str | Path) -> Path | None:
    meta = load_latest_meta(checkpoint_dir)
    if not meta:
        return None
    path = Path(checkpoint_dir) / str(meta["latest"])
    return path if path.exists() else None


def write_state(config: PipelineConfig, stage: str, status: str, **extra: Any) -> None:
    path = Path(config.state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": stage,
        "status": status,
        "config": asdict(config),
        **extra,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_base_train_config(
    *,
    profile_name: str,
    max_steps: int,
    checkpoint_dir: str,
    tokenizer_vocab_size: int,
) -> dict[str, Any]:
    if profile_name not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {profile_name}")
    config = dict(TRAIN_CONFIG)
    config.update(MODEL_PROFILES[profile_name])
    config["model_profile"] = profile_name
    config["max_steps"] = max_steps
    config["checkpoint_dir"] = checkpoint_dir
    config["vocab_size"] = tokenizer_vocab_size
    return config


def build_text_dataset(config: dict[str, Any], tokenizer) -> TextDataset:
    text_paths = resolve_text_sources(config)
    if not text_paths:
        raise RuntimeError(
            "Tidak ada text source untuk base training. "
            "Set SIGER_TEXT_SOURCES atau isi TRAIN_CONFIG['text_sources']."
        )

    print(f"Text files: {len(text_paths)}")
    for path in text_paths[:10]:
        print(f"  - {path}")
    if len(text_paths) > 10:
        print(f"  ... +{len(text_paths) - 10} more")

    dataset = TextDataset.from_text_files(
        paths=text_paths,
        tokenizer=tokenizer,
        max_seq_len=config["max_seq_len"],
        max_chars_per_file=config.get("max_chars_per_text_file"),
        max_chunks=config.get("max_dataset_chunks"),
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset menghasilkan 0 chunks.")
    return dataset


def build_model(config: dict[str, Any]) -> SigerLM:
    model_config = SigerConfig(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        d_state=config.get("d_state", 16),
        d_conv=config.get("d_conv", 4),
        expand=config.get("expand", 2),
        max_seq_len=config["max_seq_len"],
        activation=config.get("activation", "silu"),
        norm_type=config.get("norm_type", "rmsnorm"),
        norm_eps=config.get("norm_eps", 1e-6),
        norm_bias=config.get("norm_bias", False),
        initializer_range=config.get("initializer_range", 0.02),
        residual_scale_init=config.get("residual_scale_init", True),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        use_moe=config.get("use_moe", False),
        moe_num_experts=config.get("moe_num_experts", 8),
        moe_top_k=config.get("moe_top_k", 2),
        moe_expert_hidden_mult=config.get("moe_expert_hidden_mult", 2),
        moe_layers_every=config.get("moe_layers_every", 2),
        moe_aux_loss_weight=config.get("moe_aux_loss_weight", 0.01),
        moe_router_jitter=config.get("moe_router_jitter", 0.01),
    )
    config["d_inner"] = model_config.d_inner
    config["d_state"] = model_config.d_state
    config["expand"] = model_config.expand
    return SigerLM(model_config)


def run_base_stage(
    *,
    profile_name: str,
    max_steps: int,
    checkpoint_dir: str,
    resume: bool,
    warm_start_checkpoint: Path | None = None,
    learning_loss: float | None = None,
) -> dict[str, Any]:
    tokenizer = build_tokenizer("auto")
    print(f"Tokenizer backend: {tokenizer.backend} | vocab_size={tokenizer.vocab_size}")
    train_config = build_base_train_config(
        profile_name=profile_name,
        max_steps=max_steps,
        checkpoint_dir=checkpoint_dir,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )
    if train_config.get("use_moe", False):
        hardware = detect_hardware(prefer_gpu=train_config.get("prefer_gpu", True))
        moe_settings = resolve_adaptive_moe_settings(
            hardware,
            learning_loss=learning_loss,
            max_experts=int(train_config.get("moe_max_experts", 16)),
        )
        train_config["moe_num_experts"] = moe_settings.num_experts
        train_config["moe_top_k"] = moe_settings.top_k
        train_config["moe_layers_every"] = moe_settings.layers_every
        print(
            "Adaptive MoE settings: "
            f"experts={moe_settings.num_experts}, "
            f"top_k={moe_settings.top_k}, "
            f"layers_every={moe_settings.layers_every} "
            f"({moe_settings.reason})"
        )
    dataset = build_text_dataset(train_config, tokenizer)
    model = build_model(train_config)

    if warm_start_checkpoint:
        print(f"Warm-starting {profile_name} from: {warm_start_checkpoint}")
        checkpoint = torch.load(warm_start_checkpoint, map_location="cpu")
        state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
        missing, unexpected = model.load_state_dict(state, strict=False)
        moe_missing = [key for key in missing if ".moe" in key or ".moe_norm" in key]
        real_missing = [key for key in missing if key not in moe_missing]
        if real_missing or unexpected:
            raise RuntimeError(
                "Warm-start checkpoint tidak cocok. "
                f"missing={real_missing[:10]}, unexpected={unexpected[:10]}"
            )
        print(f"MoE added with {len(moe_missing)} new randomly initialized tensors.")

    trainer = Trainer(model, train_config)
    trainer.train(dataset, resume=resume)
    return load_latest_meta(checkpoint_dir) or {}


def checkpoint_losses(checkpoint_dir: str | Path) -> list[tuple[int, float]]:
    rows: list[tuple[int, float]] = []
    for path in sorted(Path(checkpoint_dir).glob("step_*.pt")):
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception as exc:
            print(f"Skip unreadable checkpoint {path}: {exc}")
            continue
        if "step" in ckpt and "loss" in ckpt:
            rows.append((int(ckpt["step"]), float(ckpt["loss"])))
    rows.sort(key=lambda item: item[0])
    return rows


def is_dense_ready(config: PipelineConfig, meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    return (
        int(meta.get("step", 0)) >= config.dense_min_steps
        and float(meta.get("loss", float("inf"))) <= config.dense_loss_threshold
    )


def is_moe_plateau(config: PipelineConfig) -> bool:
    losses = checkpoint_losses(config.moe_checkpoint_dir)
    if len(losses) < config.moe_min_checkpoints:
        return False
    _, prev_loss = losses[-2]
    _, last_loss = losses[-1]
    return abs(prev_loss - last_loss) <= config.moe_plateau_delta


def run_lora_stage(config: PipelineConfig, base_checkpoint: Path) -> None:
    hardware = detect_hardware(prefer_gpu=True)
    print_hardware_profile(hardware)
    if hardware.device == "cpu":
        n_cores = 1 if hardware.ram_gb < 3.0 else min(2, hardware.cpu_cores)
        configure_cpu(n_cores=n_cores)

    lora_config = LoRAConfig.from_json(config.lora_config)
    if lora_config.device == "auto":
        lora_config.device = hardware.device
    lora_config.base_checkpoint = str(base_checkpoint)

    print(f"Training dataset: {lora_config.dataset_path or lora_config.dataset_name}")
    print(f"Base checkpoint : {lora_config.base_checkpoint}")
    print(f"Save dir        : {lora_config.save_dir}")

    state_dict = load_checkpoint_state(lora_config.base_checkpoint)
    model_config = infer_model_config_from_state_dict(state_dict)
    base_model = SigerLM(model_config)
    base_model.load_state_dict(state_dict, strict=True)
    base_model.eval()

    print("\nInjecting LoRA adapters...")
    lora_model = LoRAModel(base_model, lora_config)

    tokenizer = build_tokenizer("auto")
    print(f"Tokenizer backend: {tokenizer.backend} | vocab_size={tokenizer.vocab_size}")
    if tokenizer.vocab_size != base_model.config.vocab_size:
        raise RuntimeError(
            "Tokenizer vocab_size tidak cocok dengan base checkpoint. "
            f"tokenizer={tokenizer.vocab_size}, model={base_model.config.vocab_size}."
        )

    dataset = InstructionDataset(
        tokenizer=tokenizer,
        dataset_path=lora_config.dataset_path,
        dataset_name=lora_config.dataset_name,
        split=lora_config.dataset_split,
        max_seq_len=lora_config.max_seq_len,
        max_samples=lora_config.max_samples,
    )

    trainer = LoRATrainer(lora_model=lora_model, config=lora_config, tokenizer=tokenizer)
    trainer.train(dataset)

    barrier()
    if is_main_process():
        print("\nMerging LoRA into base model...")
        lora_model.merge_and_export(lora_config.merged_output)
    barrier()
    cleanup_distributed()


def main() -> None:
    args = parse_args()
    config = PipelineConfig(**vars(args))

    maybe_relaunch_with_torchrun(
        script_path=Path(__file__).resolve(),
        argv=sys.argv[1:],
        strategy=TRAIN_CONFIG.get("distributed_strategy", "auto"),
        enabled=TRAIN_CONFIG.get("multi_gpu", True),
    )

    Path(config.dense_checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.moe_checkpoint_dir).mkdir(parents=True, exist_ok=True)

    dense_meta = load_latest_meta(config.dense_checkpoint_dir)
    if config.force_stage in {None, "dense"} and not is_dense_ready(config, dense_meta):
        write_state(config, "dense", "running", latest=dense_meta)
        dense_meta = run_base_stage(
            profile_name="small",
            max_steps=config.dense_max_steps,
            checkpoint_dir=config.dense_checkpoint_dir,
            resume=True,
        )
        write_state(config, "dense", "complete", latest=dense_meta)

    if config.force_stage is None and not is_dense_ready(config, dense_meta):
        print(
            "Dense gate belum lolos: "
            f"step={dense_meta.get('step') if dense_meta else None}, "
            f"loss={dense_meta.get('loss') if dense_meta else None}. "
            "Pipeline berhenti sebelum MoE."
        )
        write_state(config, "dense", "gate_failed", latest=dense_meta)
        return

    dense_checkpoint = latest_checkpoint_path(config.dense_checkpoint_dir)
    moe_checkpoint = latest_checkpoint_path(config.moe_checkpoint_dir)
    if config.force_stage != "lora" and not dense_checkpoint:
        raise RuntimeError("Dense checkpoint belum tersedia untuk transisi MoE.")

    moe_meta = load_latest_meta(config.moe_checkpoint_dir)
    if config.force_stage in {None, "moe"} and not is_moe_plateau(config):
        write_state(config, "moe", "running", latest=moe_meta)
        moe_meta = run_base_stage(
            profile_name="small_moe",
            max_steps=config.moe_max_steps,
            checkpoint_dir=config.moe_checkpoint_dir,
            resume=(moe_meta is not None),
            warm_start_checkpoint=None if moe_meta else dense_checkpoint,
            learning_loss=float(dense_meta["loss"]) if dense_meta and "loss" in dense_meta else None,
        )
        write_state(config, "moe", "complete", latest=moe_meta)

    if config.force_stage is None and not is_moe_plateau(config):
        print("MoE plateau gate belum lolos. Pipeline berhenti sebelum LoRA.")
        write_state(
            config,
            "moe",
            "gate_failed",
            latest=load_latest_meta(config.moe_checkpoint_dir),
            recent_losses=checkpoint_losses(config.moe_checkpoint_dir)[-3:],
        )
        return

    moe_checkpoint = latest_checkpoint_path(config.moe_checkpoint_dir) or dense_checkpoint
    if not moe_checkpoint:
        raise RuntimeError("Tidak ada checkpoint base untuk LoRA. Jalankan dense/MoE dulu.")
    if config.force_stage in {None, "lora"}:
        write_state(config, "lora", "running", base_checkpoint=str(moe_checkpoint))
        run_lora_stage(config, moe_checkpoint)
        write_state(config, "lora", "complete", base_checkpoint=str(moe_checkpoint))


if __name__ == "__main__":
    main()
