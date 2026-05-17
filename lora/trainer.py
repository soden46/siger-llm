# lora/trainer.py
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from .config  import LoRAConfig
from .model   import LoRAModel
from .dataset import InstructionDataset, collate_fn
from training.optimizer import CosineScheduler
from training.logger    import TrainingLogger
from optimization.hardware import detect_hardware, print_hardware_profile


class LoRATrainer:
    def __init__(
        self,
        lora_model: LoRAModel,
        config: LoRAConfig,
        tokenizer,
    ):
        self.model     = lora_model
        self.config    = config
        self.tokenizer = tokenizer
        if config.device == "auto":
            hardware = detect_hardware(prefer_gpu=config.prefer_gpu)
            print_hardware_profile(hardware)
            self.device = hardware.device
        else:
            self.device = config.device
        self.model.to(self.device)

        # Hanya optimize LoRA params
        lora_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate / 10,
        )
        self.logger = TrainingLogger(log_interval=config.log_interval)

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def train(self, dataset: InstructionDataset):
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.pad_id),
            num_workers=1,
            pin_memory=(self.device == "cuda"),
        )

        print(f"\n🚀 LoRA Training")
        print(f"   Dataset   : {len(dataset):,} examples")
        print(f"   Max steps : {self.config.max_steps:,}")
        print(f"   Batch size: {self.config.batch_size} × {self.config.grad_accum} accum")
        print(f"   Eff. batch: {self.config.batch_size * self.config.grad_accum}\n")

        self.model.train()
        step = 0
        tokens_since_step = 0
        last_step_time = time.time()
        self.optimizer.zero_grad()

        while step < self.config.max_steps:
            for batch in loader:
                if step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    tokens_since_step += int(attention_mask.sum().item())
                else:
                    tokens_since_step += int((input_ids != self.tokenizer.pad_id).sum().item())

                # Forward
                logits, _ = self.model(input_ids)

                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()

                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=0.0,
                )
                loss = loss / self.config.grad_accum
                loss.backward()

                if (step + 1) % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr = self.scheduler.step()

                    now = time.time()
                    elapsed = max(now - last_step_time, 1e-9)
                    tokens_per_sec = tokens_since_step / elapsed
                    last_step_time = now
                    tokens_since_step = 0

                    self.logger.log(
                        step,
                        loss.item() * self.config.grad_accum,
                        lr,
                        tokens_per_sec=tokens_per_sec,
                    )

                    if step > 0 and step % self.config.save_every == 0:
                        self._save(step, loss.item())

                step += 1

        # Final save
        self._save(step, loss.item())
        print("\n✅ LoRA training complete!")

    def _save(self, step: int, loss: float):
        path = f"{self.config.save_dir}/lora_step_{step:06d}.pt"
        self.model.save_lora(path)
        print(f"💾 Saved: {path} | loss={loss:.4f}")
