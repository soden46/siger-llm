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
from optimization.autotune import suggest_cuda_batch_size
from optimization.gpu import (
    amp_dtype_from_plan,
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
            max_workers=config.max_dataloader_workers,
            strategy=config.distributed_strategy,
            resource_target_fraction=getattr(config, "resource_target_fraction", 1.0),
            precision=config.precision,
        )
        self.device = self.runtime.device
        self.amp_dtype = amp_dtype_from_plan(self.runtime)
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

        if config.auto_tune_batch_vram:
            base_batch = int(config.batch_size)
            base_config = getattr(lora_model.base_model, "config", None)
            d_model = int(getattr(base_config, "d_model", 256))
            d_inner = int(getattr(base_config, "d_inner", d_model * int(getattr(base_config, "expand", 2))))
            d_state = int(getattr(base_config, "d_state", 16))
            n_layers = int(getattr(base_config, "n_layers", 1))
            tuned_batch = suggest_cuda_batch_size(
                base_batch_size=base_batch,
                max_batch_size=int(config.max_global_batch_size),
                max_seq_len=int(config.max_seq_len),
                d_model=d_model,
                d_inner=d_inner,
                d_state=d_state,
                n_layers=n_layers,
                safety_fraction=float(config.vram_safety_fraction),
            )
            if tuned_batch > base_batch:
                config.batch_size = tuned_batch
                if self.runtime.is_main_process:
                    scope = "per-rank" if self.runtime.is_distributed else "global"
                    print(f"VRAM-tuned LoRA {scope} batch_size: {base_batch} -> {tuned_batch}")

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
        loader_kwargs = {}
        if self.runtime.dataloader_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.pad_id),
            num_workers=self.runtime.dataloader_workers,
            pin_memory=self.runtime.pin_memory,
            **loader_kwargs,
        )

        print(f"\n🚀 LoRA Training")
        print(f"   Dataset   : {len(dataset):,} examples")
        print(f"   Max steps : {self.config.max_steps:,}")
        print(f"   Batch size: {self.config.batch_size} × {self.config.grad_accum} accum")
        print(f"   World size: {self.runtime.world_size}")
        print(f"   Eff. batch: {self.config.batch_size * self.config.grad_accum * max(1, self.runtime.world_size)}\n")

        self.model.train()
        step = 0
        tokens_since_step = 0
        last_step_time = time.time()
        self.optimizer.zero_grad(set_to_none=True)
        device_str = str(self.device)
        device_type = "cuda" if "cuda" in device_str else "cpu"
        scaler = torch.amp.GradScaler(
            device_type,
            enabled=(device_type == "cuda" and self.runtime.precision == "fp16"),
        )
        last_loss = float("inf")

        while step < self.config.max_steps:
            if self.runtime.is_distributed and hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(step)
            for batch in loader:
                if step >= self.config.max_steps or (elastic_state and elastic_state.should_stop):
                    break

                input_ids = batch["input_ids"].to(self.device, non_blocking=self.runtime.pin_memory)
                labels = batch["labels"].to(self.device, non_blocking=self.runtime.pin_memory)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    tokens_since_step += int(attention_mask.sum().item())
                else:
                    tokens_since_step += int((input_ids != self.tokenizer.pad_id).sum().item())

                # Forward
                with torch.autocast(
                    device_type=device_type,
                    dtype=self.amp_dtype,
                    enabled=(device_type == "cuda" and self.runtime.precision != "fp32"),
                ):
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
                scaler.scale(loss).backward()
                last_loss = loss.item() * self.config.grad_accum

                if (step + 1) % self.config.grad_accum == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr = self.scheduler.step()

                    now = time.time()
                    elapsed = max(now - last_step_time, 1e-9)
                    tokens_per_sec = tokens_since_step / elapsed
                    last_step_time = now
                    tokens_since_step = 0

                    if self.runtime.is_main_process:
                        self.logger.log(
                            step,
                            last_loss,
                            lr,
                            tokens_per_sec=tokens_per_sec,
                        )

                    if self.runtime.is_main_process and step > 0 and step % self.config.save_every == 0:
                        self._save(step, last_loss)
                        if self.config.sharded_checkpoint:
                            save_sharded_checkpoint(
                                model=self.model,
                                optimizer=self.optimizer,
                                scheduler=self.scheduler,
                                output_dir=Path(self.config.save_dir) / f"sharded_step_{step:06d}",
                                step=step,
                                loss=last_loss,
                                config=self.config.__dict__,
                            )

                step += 1

            if elastic_state and elastic_state.should_stop:
                break

        # Final save
        if self.runtime.is_distributed:
            barrier()
        if self.runtime.is_main_process:
            self._save(step, last_loss)
            if self.config.sharded_checkpoint:
                save_sharded_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    output_dir=Path(self.config.save_dir) / "sharded_final",
                    step=step,
                    loss=last_loss,
                    config=self.config.__dict__,
                )
        print("\n✅ LoRA training complete!")

    def _save(self, step: int, loss: float):
        path = f"{self.config.save_dir}/lora_step_{step:06d}.pt"
        unwrap_model(self.model).save_lora(path)
        print(f"💾 Saved: {path} | loss={loss:.4f}")
