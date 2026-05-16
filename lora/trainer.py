# lora/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from .config  import LoRAConfig
from .model   import LoRAModel
from .dataset import InstructionDataset, collate_fn
from training.optimizer import CosineScheduler
from training.logger    import TrainingLogger


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
        self.device    = "cpu"   # VPS lo CPU only
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
            pin_memory=False,
        )

        print(f"\n🚀 LoRA Training")
        print(f"   Dataset   : {len(dataset):,} examples")
        print(f"   Max steps : {self.config.max_steps:,}")
        print(f"   Batch size: {self.config.batch_size} × {self.config.grad_accum} accum")
        print(f"   Eff. batch: {self.config.batch_size * self.config.grad_accum}\n")

        self.model.train()
        step = 0
        self.optimizer.zero_grad()

        while step < self.config.max_steps:
            for batch in loader:
                if step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                # Forward
                logits, _ = self.model(input_ids)

                # Loss — hanya di posisi yang bukan -100
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,    # skip system/user tokens
                    label_smoothing=0.1,  # regularization ringan
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

                    self.logger.log(step, loss.item() * self.config.grad_accum, lr)

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