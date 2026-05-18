# lora/trainer.py
import torch
import torch.nn as nn
import time
import psutil
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from pathlib import Path

from .config  import LoRAConfig
from .model   import LoRAModel
from .dataset import InstructionDataset, collate_fn
from training.optimizer import CosineScheduler
from training.logger    import TrainingLogger
from optimization.gpu import (
    barrier,
    build_runtime_plan,
    print_runtime_plan,
    unwrap_model,
    wrap_model_for_runtime,
)
from optimization.elastic import install_signal_handlers
from optimization.sharded_checkpoint import save_sharded_checkpoint
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
        else:
            hardware = detect_hardware(prefer_gpu=(config.device == "cuda"))
        self.runtime = build_runtime_plan(
            prefer_gpu=config.prefer_gpu,
            requested_device=config.device,
            cpu_cores=hardware.cpu_cores,
            strategy=config.distributed_strategy,
            resource_target_fraction=getattr(config, "resource_target_fraction", 1.0),
        )
        self.device = self.runtime.device
        if getattr(config, "resource_target_fraction", 1.0) < 1.0:
            max_threads = max(1, int((psutil.cpu_count(logical=True) or 1) * config.resource_target_fraction))
            torch.set_num_threads(max_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            if self.runtime.is_main_process:
                print(f"Resource throttle: target={config.resource_target_fraction:.0%}, torch_threads={max_threads}")
        self.model.to(self.device)
        self.model = wrap_model_for_runtime(
            self.model,
            self.runtime,
            enabled=True,
        )
        print_runtime_plan(self.runtime)

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
        elastic_state = install_signal_handlers() if self.config.elastic_recovery else None
        sampler = DistributedSampler(dataset, shuffle=True) if self.runtime.is_distributed else None
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.pad_id),
            num_workers=self.runtime.dataloader_workers,
            pin_memory=self.runtime.pin_memory,
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
            if self.runtime.is_distributed and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(step)
            for batch in loader:
                if step >= self.config.max_steps or (elastic_state and elastic_state.should_stop):
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

                    if self.runtime.is_main_process:
                        self.logger.log(
                            step,
                            loss.item() * self.config.grad_accum,
                            lr,
                            tokens_per_sec=tokens_per_sec,
                        )

                    if self.runtime.is_main_process and step > 0 and step % self.config.save_every == 0:
                        self._save(step, loss.item())
                        if self.config.sharded_checkpoint:
                            save_sharded_checkpoint(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                output_dir=Path(self.config.save_dir) / f"sharded_step_{step:06d}",
                                step=step,
                                loss=loss.item(),
                                config=self.config.__dict__,
                            )

                step += 1

            if elastic_state and elastic_state.should_stop:
                break

        # Final save
        if self.runtime.is_distributed:
            barrier()
        if self.runtime.is_main_process:
            self._save(step, loss.item())
            if self.config.sharded_checkpoint:
                save_sharded_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    output_dir=Path(self.config.save_dir) / "sharded_final",
                    step=step,
                    loss=loss.item(),
                    config=self.config.__dict__,
                )
        print("\n✅ LoRA training complete!")

    def _save(self, step: int, loss: float):
        path = f"{self.config.save_dir}/lora_step_{step:06d}.pt"
        unwrap_model(self.model).save_lora(path)
        print(f"💾 Saved: {path} | loss={loss:.4f}")
