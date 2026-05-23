"""DPO training for SigerLM LoRA adapters.

This module keeps DPO independent from the normal SFT LoRA trainer because the
batch shape and loss are different: every sample contains a prompt, a chosen
answer, and a rejected answer.
"""

from __future__ import annotations

import argparse
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from lora.config import LoRAConfig
from lora.model import LoRAModel
from lora.run_lora import load_base_model
from tokenizer.hybrid_tokenizer import build_tokenizer


logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class DPOConfig:
    """DPO-specific training configuration."""

    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"
    average_log_prob: bool = True

    prompt_field: str = "prompt"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    system_prompt: str = (
        "Kamu adalah SigerLM, asisten AI umum yang cerdas, ringkas, kritis, "
        "akurat, dan menjawab dengan bahasa yang lembut."
    )

    log_preference_accuracy: bool = True
    log_margin: bool = True


def _tokenizer_pad_id(tokenizer) -> int:
    pad_id = getattr(tokenizer, "pad_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        raise RuntimeError("Tokenizer tidak punya pad_id/pad_token_id.")
    return int(pad_id)


def _build_chat_text(system_prompt: str, prompt: str, answer: str) -> str:
    return (
        f"<|system|>{system_prompt}<|end_turn|>\n"
        f"<|user|>{prompt}<|end_turn|>\n"
        f"<|assistant|>{answer}<|end_turn|>"
    )


def _build_labels(input_ids: list[int], tokenizer) -> list[int]:
    labels = [IGNORE_INDEX] * len(input_ids)
    assistant_id = tokenizer.special_tokens.get("<|assistant|>")
    end_turn_id = tokenizer.special_tokens.get("<|end_turn|>")
    if assistant_id is None or end_turn_id is None:
        raise RuntimeError("Tokenizer tidak punya <|assistant|> atau <|end_turn|>.")

    in_assistant = False
    for idx, token_id in enumerate(input_ids):
        if token_id == assistant_id:
            in_assistant = True
            continue
        if token_id == end_turn_id and in_assistant:
            labels[idx] = token_id
            in_assistant = False
            continue
        if in_assistant:
            labels[idx] = token_id
    return labels


class PreferenceDataset(Dataset):
    """Preference-pair JSONL dataset for DPO.

    Expected row:
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    """

    def __init__(
        self,
        data_path: str | Path,
        tokenizer,
        config: DPOConfig,
        max_length: int = 512,
        max_samples: Optional[int] = None,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.pad_id = _tokenizer_pad_id(tokenizer)
        self.rows = self._load_data(max_samples=max_samples)

        if not self.rows:
            raise RuntimeError(f"PreferenceDataset kosong: {self.data_path}")

        logger.info("Loaded %s preference pairs from %s", len(self.rows), self.data_path)

    def _load_data(self, max_samples: Optional[int]) -> list[dict]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Preference dataset tidak ditemukan: {self.data_path}")

        rows: list[dict] = []
        with self.data_path.open("r", encoding="utf-8-sig") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping invalid JSON at line %s: %s", line_num, exc)
                    continue

                if all(
                    str(row.get(field, "")).strip()
                    for field in (
                        self.config.prompt_field,
                        self.config.chosen_field,
                        self.config.rejected_field,
                    )
                ):
                    rows.append(row)
                    if max_samples is not None and len(rows) >= max_samples:
                        break

        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def _encode_answer(self, prompt: str, answer: str) -> dict[str, torch.Tensor]:
        text = _build_chat_text(self.config.system_prompt, prompt, answer)
        input_ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        input_ids = input_ids[: self.max_length]
        labels = _build_labels(input_ids, self.tokenizer)

        if all(label == IGNORE_INDEX for label in labels):
            raise RuntimeError("Preference row tidak punya token assistant untuk loss.")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt = str(row[self.config.prompt_field]).strip()
        chosen = str(row[self.config.chosen_field]).strip()
        rejected = str(row[self.config.rejected_field]).strip()

        chosen_encoded = self._encode_answer(prompt, chosen)
        rejected_encoded = self._encode_answer(prompt, rejected)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_labels": chosen_encoded["labels"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_labels": rejected_encoded["labels"],
        }


def preference_collate_fn(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    def pad_stack(key: str, pad_value: int) -> torch.Tensor:
        max_len = max(item[key].numel() for item in batch)
        padded = []
        for item in batch:
            value = item[key]
            pad_len = max_len - value.numel()
            if pad_len:
                value = torch.cat(
                    [value, torch.full((pad_len,), pad_value, dtype=value.dtype)]
                )
            padded.append(value)
        return torch.stack(padded)

    return {
        "chosen_input_ids": pad_stack("chosen_input_ids", pad_id),
        "chosen_labels": pad_stack("chosen_labels", IGNORE_INDEX),
        "rejected_input_ids": pad_stack("rejected_input_ids", pad_id),
        "rejected_labels": pad_stack("rejected_labels", IGNORE_INDEX),
    }


class DPOTrainer:
    """Small, SigerLM-native DPO trainer for LoRA adapters."""

    def __init__(
        self,
        model: LoRAModel,
        tokenizer,
        train_dataset: PreferenceDataset,
        dpo_config: DPOConfig,
        lora_config: LoRAConfig,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.dpo_config = dpo_config
        self.lora_config = lora_config
        self.device = torch.device(
            "cuda" if lora_config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise RuntimeError(
                "Tidak ada parameter LoRA yang trainable. "
                "Cek target_modules; untuk SigerLM pakai in_proj/out_proj/x_proj/dt_proj."
            )

        self.optimizer = torch.optim.AdamW(
            params,
            lr=lora_config.learning_rate,
            weight_decay=lora_config.weight_decay,
        )

    @contextmanager
    def _lora_disabled(self):
        previous_states = {}
        for name, layer in self.model.lora_layers.items():
            previous_states[name] = layer.enabled
            layer.enabled = False
        try:
            yield
        finally:
            for name, enabled in previous_states.items():
                self.model.lora_layers[name].enabled = enabled

    def _sequence_log_probs(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits, _ = self.model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_mask = shift_labels.ne(IGNORE_INDEX)
        safe_labels = shift_labels.masked_fill(~loss_mask, 0)

        token_log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = token_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * loss_mask
        seq_log_probs = token_log_probs.sum(dim=-1)

        if self.dpo_config.average_log_prob:
            seq_log_probs = seq_log_probs / loss_mask.sum(dim=-1).clamp_min(1)

        return seq_log_probs

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        policy_chosen = self._sequence_log_probs(chosen_ids, chosen_labels)
        policy_rejected = self._sequence_log_probs(rejected_ids, rejected_labels)

        with torch.no_grad(), self._lora_disabled():
            ref_chosen = self._sequence_log_probs(chosen_ids, chosen_labels)
            ref_rejected = self._sequence_log_probs(rejected_ids, rejected_labels)

        policy_log_ratio = policy_chosen - policy_rejected
        ref_log_ratio = ref_chosen - ref_rejected
        logits = policy_log_ratio - ref_log_ratio

        if self.dpo_config.dpo_loss_type == "sigmoid":
            loss = -F.logsigmoid(self.dpo_config.dpo_beta * logits).mean()
        elif self.dpo_config.dpo_loss_type == "hinge":
            loss = torch.relu(1.0 - self.dpo_config.dpo_beta * logits).mean()
        elif self.dpo_config.dpo_loss_type == "ipo":
            target = 1.0 / (2.0 * self.dpo_config.dpo_beta)
            loss = (logits - target).pow(2).mean()
        else:
            raise ValueError(f"Unknown DPO loss type: {self.dpo_config.dpo_loss_type}")

        reward_chosen = self.dpo_config.dpo_beta * (policy_chosen - ref_chosen)
        reward_rejected = self.dpo_config.dpo_beta * (policy_rejected - ref_rejected)
        metrics = {
            "loss": float(loss.detach().cpu()),
            "reward_chosen": float(reward_chosen.mean().detach().cpu()),
            "reward_rejected": float(reward_rejected.mean().detach().cpu()),
        }
        if self.dpo_config.log_preference_accuracy:
            metrics["preference_accuracy"] = float(
                (reward_chosen > reward_rejected).float().mean().detach().cpu()
            )
        if self.dpo_config.log_margin:
            metrics["margin"] = float((reward_chosen - reward_rejected).mean().detach().cpu())

        return loss, metrics

    def train(self) -> str:
        save_dir = Path(self.lora_config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        pad_id = _tokenizer_pad_id(self.tokenizer)
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.lora_config.batch_size,
            shuffle=True,
            collate_fn=lambda batch: preference_collate_fn(batch, pad_id=pad_id),
            num_workers=0,
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        max_steps = int(self.lora_config.max_steps)
        grad_accum = max(1, int(self.lora_config.grad_accum))
        step = 0
        micro_step = 0
        last_metrics: dict[str, float] = {}

        print("\nDPO Training")
        print(f"   Dataset   : {len(self.train_dataset):,} pairs")
        print(f"   Max steps : {max_steps:,}")
        print(f"   Batch size: {self.lora_config.batch_size} x {grad_accum} accum")
        print(f"   Device    : {self.device}\n")

        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break

                loss, metrics = self.compute_loss(batch)
                (loss / grad_accum).backward()
                micro_step += 1
                last_metrics = metrics

                if micro_step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    step += 1

                    if step % self.lora_config.log_interval == 0:
                        print(
                            f"step={step:06d} loss={metrics['loss']:.4f} "
                            f"pref_acc={metrics.get('preference_accuracy', 0.0):.3f} "
                            f"margin={metrics.get('margin', 0.0):.4f}"
                        )

                    if step % self.lora_config.save_every == 0:
                        self._save(step, metrics["loss"])

        return self._save(step, last_metrics.get("loss", 0.0))

    def _save(self, step: int, loss: float) -> str:
        path = Path(self.lora_config.save_dir) / f"dpo_lora_step_{step:06d}.pt"
        self.model.save_lora(str(path))
        print(f"Saved DPO LoRA: {path} | loss={loss:.4f}")
        return str(path)


def load_dpo_config(path: str | Path) -> tuple[DPOConfig, LoRAConfig, str]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    dataset_path = raw.get("dataset", {}).get("path")
    if not dataset_path:
        raise ValueError(f"Config {config_path} tidak punya dataset.path")

    training = raw.get("training", {})
    lora = raw.get("lora", {})
    output = raw.get("output", {})
    dpo = raw.get("dpo", {})
    hardware = raw.get("hardware", {})

    dpo_config = DPOConfig(
        dpo_beta=float(dpo.get("beta", dpo.get("dpo_beta", 0.1))),
        dpo_loss_type=str(dpo.get("loss_type", "sigmoid")),
        average_log_prob=bool(dpo.get("average_log_prob", True)),
        prompt_field=str(raw.get("dataset", {}).get("prompt_field", "prompt")),
        chosen_field=str(raw.get("dataset", {}).get("chosen_field", "chosen")),
        rejected_field=str(raw.get("dataset", {}).get("rejected_field", "rejected")),
    )

    lora_config = LoRAConfig(
        rank=int(lora.get("rank", 8)),
        alpha=float(lora.get("alpha", 16.0)),
        dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["in_proj", "out_proj", "x_proj", "dt_proj"])),
        learning_rate=float(training.get("learning_rate", 1e-4)),
        max_steps=int(training.get("max_steps", 500)),
        batch_size=int(training.get("batch_size", 4)),
        grad_accum=int(training.get("gradient_accumulation_steps", 1)),
        warmup_steps=0,
        max_seq_len=int(raw.get("dataset", {}).get("max_length", 512)),
        weight_decay=float(training.get("weight_decay", 0.0)),
        device=str(hardware.get("device", "auto")),
        precision=str(training.get("mixed_precision", "fp32")),
        max_dataloader_workers=int(hardware.get("num_workers", 0)),
        auto_tune_batch_vram=False,
        base_checkpoint=str(raw.get("base_checkpoint", "./checkpoints/best_model.pt")),
        save_dir=str(output.get("output_dir", "./checkpoints/lora/dpo")),
        save_every=int(training.get("save_steps", 100)),
        log_interval=int(training.get("log_steps", 10)),
    )

    return dpo_config, lora_config, str(dataset_path)


def train_dpo(
    model_checkpoint: str | Path,
    preference_dataset: str | Path,
    output_dir: str | Path,
    dpo_config: DPOConfig,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    learning_rate: float = 1e-4,
    max_steps: int = 500,
    batch_size: int = 4,
    grad_accum: int = 1,
    max_seq_len: int = 512,
    target_modules: Optional[Iterable[str]] = None,
    device: str = "auto",
) -> str:
    base_model = load_base_model(str(model_checkpoint), max_seq_len=max_seq_len)
    tokenizer = build_tokenizer("auto")
    if tokenizer.vocab_size != base_model.config.vocab_size:
        raise RuntimeError(
            "Tokenizer vocab_size tidak cocok dengan base checkpoint. "
            f"tokenizer={tokenizer.vocab_size}, model={base_model.config.vocab_size}."
        )

    lora_config = LoRAConfig(
        rank=lora_rank,
        alpha=float(lora_alpha),
        dropout=0.05,
        target_modules=list(target_modules or ["in_proj", "out_proj", "x_proj", "dt_proj"]),
        learning_rate=learning_rate,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_seq_len=max_seq_len,
        weight_decay=0.0,
        device=("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)),
        save_dir=str(output_dir),
        save_every=100,
        log_interval=10,
    )

    lora_model = LoRAModel(base_model, lora_config)
    dataset = PreferenceDataset(
        preference_dataset,
        tokenizer,
        dpo_config,
        max_length=max_seq_len,
    )
    trainer = DPOTrainer(lora_model, tokenizer, dataset, dpo_config, lora_config)
    return trainer.train()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SigerLM with DPO LoRA.")
    parser.add_argument("--config", help="Optional DPO config JSON.")
    parser.add_argument("--model-checkpoint", help="Base model checkpoint.")
    parser.add_argument("--preference-dataset", help="Preference dataset JSONL.")
    parser.add_argument("--output-dir", help="Output directory.")
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--dpo-loss-type", choices=["sigmoid", "hinge", "ipo"], default="sigmoid")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None, help="Accepted for compatibility; max-steps controls training.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    if args.config:
        dpo_config, lora_config, dataset_path = load_dpo_config(args.config)
        if lora_config.device == "auto":
            lora_config.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model = load_base_model(lora_config.base_checkpoint, max_seq_len=lora_config.max_seq_len)
        tokenizer = build_tokenizer("auto")
        if tokenizer.vocab_size != base_model.config.vocab_size:
            raise RuntimeError(
                "Tokenizer vocab_size tidak cocok dengan base checkpoint. "
                f"tokenizer={tokenizer.vocab_size}, model={base_model.config.vocab_size}."
            )
        lora_model = LoRAModel(base_model, lora_config)
        dataset = PreferenceDataset(dataset_path, tokenizer, dpo_config, max_length=lora_config.max_seq_len)
        trainer = DPOTrainer(lora_model, tokenizer, dataset, dpo_config, lora_config)
        trainer.train()
        return

    missing = [
        name
        for name, value in {
            "--model-checkpoint": args.model_checkpoint,
            "--preference-dataset": args.preference_dataset,
            "--output-dir": args.output_dir,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required args without --config: {', '.join(missing)}")

    dpo_config = DPOConfig(
        dpo_beta=args.dpo_beta,
        dpo_loss_type=args.dpo_loss_type,
    )
    train_dpo(
        model_checkpoint=args.model_checkpoint,
        preference_dataset=args.preference_dataset,
        output_dir=args.output_dir,
        dpo_config=dpo_config,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        device=args.device,
    )


if __name__ == "__main__":
    main()
