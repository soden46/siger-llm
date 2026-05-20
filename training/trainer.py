# training/trainer.py
import torch
import torch.nn as nn
import psutil
import time
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Optional

from .optimizer   import build_optimizer, CosineScheduler
from .checkpoint  import CheckpointManager
from .logger      import TrainingLogger
from .dataset     import TextDataset
from tokenizer.tokenizer import MultilingualTokenizer
from optimization.gpu import (
    amp_dtype_from_plan,
    barrier,
    build_runtime_plan,
    cleanup_distributed,
    print_runtime_plan,
    unwrap_model,
    wrap_model_for_runtime,
)
from optimization.autotune import suggest_cuda_batch_size
from optimization.elastic import install_signal_handlers
from optimization.sharded_checkpoint import save_sharded_checkpoint


class Trainer:
    def __init__(self, model, config: dict, device: str = None):
        self.model  = model
        self.config = config
        self.runtime = build_runtime_plan(
            prefer_gpu=config.get("prefer_gpu", True),
            requested_device=device or config.get("device", "auto"),
            cpu_cores=config.get("cpu_cores", 1),
            max_workers=config.get("max_dataloader_workers"),
            strategy=config.get("distributed_strategy", "auto"),
            resource_target_fraction=config.get("resource_target_fraction", 1.0),
            precision=config.get("precision", "auto"),
        )
        self.device = self.runtime.device
        self.amp_dtype = amp_dtype_from_plan(self.runtime)
        if self.config.get("resource_target_fraction", 1.0) < 1.0:
            max_threads = max(1, int((psutil.cpu_count(logical=True) or 1) * self.config["resource_target_fraction"]))
            torch.set_num_threads(max_threads)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            if self.runtime.is_main_process:
                print(f"Resource throttle: target={self.config['resource_target_fraction']:.0%}, torch_threads={max_threads}")

        self.model.to(self.device)
        self.model = wrap_model_for_runtime(
            self.model,
            self.runtime,
            enabled=config.get("multi_gpu", True),
        )
        if self.runtime.is_main_process:
            print(f"🖥️  Device: {self.device}")
            print_runtime_plan(self.runtime)

        import os
        env_batch = os.environ.get("SIGER_BATCH_SIZE")
        if env_batch:
            old_batch = int(config["batch_size"])
            config["batch_size"] = int(env_batch)
            if self.runtime.is_main_process:
                print(f"Env override batch_size: {old_batch} -> {config['batch_size']}")

        if (
            config.get("auto_scale_batch", False)
            and self.runtime.device_count > 1
            and not self.runtime.is_distributed
        ):
            base_batch = int(config["batch_size"])
            factor = min(self.runtime.device_count, int(config.get("max_auto_scale_factor", 2)))
            if config.get("resource_target_fraction", 1.0) < 1.0:
                factor = min(factor, 1)
            max_batch = int(config.get("max_global_batch_size", base_batch))
            scaled_batch = min(base_batch * factor, max_batch)
            if scaled_batch > base_batch:
                config["batch_size"] = scaled_batch
                if self.runtime.is_main_process:
                    print(f"Auto-scaled global batch_size: {base_batch} -> {scaled_batch}")

        if config.get("auto_tune_batch_vram", False):
            base_batch = int(config["batch_size"])
            per_device_cap = int(config.get("max_per_device_batch_size", base_batch))
            global_cap = int(config.get("max_global_batch_size", base_batch))
            if self.runtime.is_distributed:
                global_cap = max(1, global_cap // max(1, self.runtime.world_size))
            max_tuned_batch = max(base_batch, min(per_device_cap, global_cap))
            tuned_batch = suggest_cuda_batch_size(
                base_batch_size=base_batch,
                max_batch_size=max_tuned_batch,
                max_seq_len=int(config["max_seq_len"]),
                d_model=int(config.get("d_model", 256)),
                d_inner=int(config.get("d_inner", int(config.get("d_model", 256)) * int(config.get("expand", 2)))),
                d_state=int(config.get("d_state", 16)),
                n_layers=int(config.get("n_layers", 1)),
                safety_fraction=float(config.get("vram_safety_fraction", 0.70)),
            )
            if tuned_batch > base_batch:
                config["batch_size"] = tuned_batch
                if self.runtime.is_main_process:
                    scope = "per-rank" if self.runtime.is_distributed else "global"
                    print(f"VRAM-tuned {scope} batch_size: {base_batch} -> {tuned_batch}")

        # Komponen training
        self.optimizer = build_optimizer(
            self.model,
            lr=config["max_lr"],
            weight_decay=config.get("weight_decay", 0.1)
        )
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            max_lr=config["max_lr"],
            min_lr=config.get("min_lr", config["max_lr"] / 10),
        )
        self.ckpt_manager = CheckpointManager(
            save_dir=config["checkpoint_dir"],
            keep_last=config.get("keep_last_checkpoints", 3),
        )
        self.logger = TrainingLogger(
            log_interval=config.get("log_interval", 10)
        )

        self.grad_clip   = config.get("grad_clip", 1.0)
        self.accum_steps = config.get("grad_accum_steps", 1)  # gradient accumulation

    def _build_dataloader(self, dataset: TextDataset) -> DataLoader:
        sampler = DistributedSampler(dataset, shuffle=True) if self.runtime.is_distributed else None
        configured_workers = self.config.get("num_workers", "auto")
        num_workers = self.runtime.dataloader_workers if configured_workers in {None, "auto"} else int(configured_workers)
        loader_kwargs = {}
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = int(self.config.get("prefetch_factor", 2))
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=self.runtime.pin_memory,
            drop_last=True,
            **loader_kwargs,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single forward + backward step."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Mixed precision (otomatis kalau CUDA)
        device_str = str(self.device)
        device_type = "cuda" if "cuda" in device_str else "cpu"
        with torch.autocast(
            device_type=device_type,
            dtype=self.amp_dtype,
            enabled=(device_type == "cuda" and self.runtime.precision != "fp32"),
        ):
            _, loss = self.model(x, targets=y)
            if loss.dim() > 0:
                loss = loss.mean()
            loss = loss / self.accum_steps  # scale buat grad accum

        loss.backward()
        return loss.item() * self.accum_steps  # return unscaled loss

    def train(
        self,
        dataset: TextDataset,
        resume: bool = True,
    ):
        """Main training loop."""
        elastic_state = install_signal_handlers() if self.config.get("elastic_recovery", True) else None
        dataloader = self._build_dataloader(dataset)

        device_str = str(self.device)
        device_type = "cuda" if "cuda" in device_str else "cpu"
        scaler = torch.amp.GradScaler(
            device_type,
            enabled=(device_type == "cuda" and self.runtime.precision == "fp16"),
        )
        
        global_step = 0
        best_loss = float("inf")

        if resume:
            global_step, best_loss = self.ckpt_manager.load(
                self.model, self.optimizer, self.scheduler
            )

        self.model.train()
        max_steps = self.config["max_steps"]
        if self.runtime.is_main_process:
            print(f"\n🚀 Training starts | max_steps={max_steps:,}")
            effective_batch = self.config["batch_size"] * self.accum_steps * max(1, self.runtime.world_size)
            print(f"   batch_size={self.config['batch_size']} | "
                  f"grad_accum={self.accum_steps} | "
                  f"world_size={self.runtime.world_size} | "
                  f"effective_batch={effective_batch}\n")

        self.optimizer.zero_grad(set_to_none=True)
        epoch = 0
        last_loss = 0.0
        loss_accum = 0.0
        micro_batches = 0
        last_step_time = time.time()
        steps_per_epoch = len(dataloader)

        while global_step < max_steps:
            epoch += 1
            if self.runtime.is_distributed and hasattr(dataloader.sampler, "set_epoch"):
                dataloader.sampler.set_epoch(epoch)
            for batch_idx, (x, y) in enumerate(dataloader):
                if global_step >= max_steps or (elastic_state and elastic_state.should_stop):
                    break

                # Forward + Backward menggunakan AMP Scaler yang benar
                x = x.to(self.device, non_blocking=self.runtime.pin_memory)
                y = y.to(self.device, non_blocking=self.runtime.pin_memory)
                with torch.autocast(
                    device_type=device_type,
                    dtype=self.amp_dtype,
                    enabled=(device_type == "cuda" and self.runtime.precision != "fp32"),
                ):
                    _, loss = self.model(x, targets=y)
                    if loss.dim() > 0:
                        loss = loss.mean()
                    loss = loss / self.accum_steps

                # Skala loss untuk mencegah underflow gradien pada FP16
                scaler.scale(loss).backward()
                loss_accum += loss.item() * self.accum_steps
                micro_batches += 1

                # Update weights setiap accum_steps
                is_accum_step = micro_batches >= self.accum_steps
                is_epoch_end = (batch_idx + 1) == steps_per_epoch
                if is_accum_step or is_epoch_end:
                    # Unscale gradien sebelum clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    # Optimizer step via Scaler
                    scale_before_step = scaler.get_scale()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    step_was_skipped = (
                        scaler.is_enabled()
                        and scaler.get_scale() < scale_before_step
                    )

                    actual_loss = loss_accum / max(1, micro_batches)
                    loss_accum = 0.0
                    micro_batches = 0

                    if step_was_skipped:
                        continue

                    if self.runtime.is_distributed:
                        loss_tensor = torch.tensor([actual_loss], device=self.device)
                        torch.distributed.all_reduce(
                            loss_tensor,
                            op=torch.distributed.ReduceOp.AVG,
                        )
                        actual_loss = loss_tensor.item()

                    # LR scheduler step
                    lr = self.scheduler.step()
                    global_step += 1
                    last_loss = actual_loss

                    # Hitung throughput tokens
                    tokens_per_step = (
                        self.config["batch_size"]
                        * self.config["max_seq_len"]
                        * self.accum_steps
                        * max(1, self.runtime.world_size)
                    )
                    now = time.time()
                    elapsed = max(now - last_step_time, 1e-9)
                    tokens_per_sec = tokens_per_step / elapsed
                    last_step_time = now

                    # Logging
                    if self.runtime.is_main_process:
                        extra_metrics = {}
                        raw_model = unwrap_model(self.model)
                        moe_aux = getattr(raw_model, "last_moe_aux_loss", None)
                        if moe_aux is None:
                            moe_aux = getattr(raw_model, "last_aux_loss", None)
                        dead_experts = getattr(raw_model, "last_moe_dead_expert_fraction", None)
                        if dead_experts is None:
                            dead_experts = getattr(raw_model, "last_dead_expert_fraction", None)
                        if moe_aux is not None:
                            extra_metrics["moe_aux"] = float(moe_aux.detach().cpu().item())
                        if dead_experts is not None:
                            extra_metrics["moe_dead"] = float(dead_experts.detach().cpu().item())
                        self.logger.log(
                            global_step,
                            actual_loss,
                            lr,
                            tokens_per_sec,
                            extra_metrics=extra_metrics or None,
                        )

                    # 2. PERBAIKAN BUG: Simpan checkpoint & best model HANYA pada interval tertentu
                    save_every = self.config.get("save_every", 500)
                    if global_step > 0 and global_step % save_every == 0:
                        if self.runtime.is_distributed:
                            barrier()

                        if self.runtime.is_main_process:
                            self.ckpt_manager.save(
                                self.model, self.optimizer, self.scheduler,
                                step=global_step, loss=actual_loss, config=self.config,
                            )
                            if self.config.get("sharded_checkpoint", False):
                                save_sharded_checkpoint(
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    output_dir=self.ckpt_manager.save_dir / f"sharded_step_{global_step:07d}",
                                    step=global_step,
                                    loss=actual_loss,
                                    config=self.config,
                                )

                            # Cek dan simpan model terbaik di interval ini saja untuk menghemat I/O disk
                            if actual_loss < best_loss:
                                best_loss = actual_loss
                                self._save_best()

                        if self.runtime.is_distributed:
                            barrier()

            if elastic_state and elastic_state.should_stop:
                break

        if self.runtime.is_distributed:
            barrier()

        if self.runtime.is_main_process:
            self.logger.summary(global_step)
            # Final save
            self.ckpt_manager.save(
                self.model, self.optimizer, self.scheduler,
                step=global_step, loss=last_loss, config=self.config,
            )
            if self.config.get("sharded_checkpoint", False):
                save_sharded_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    output_dir=self.ckpt_manager.save_dir / "sharded_final",
                    step=global_step,
                    loss=last_loss,
                    config=self.config,
                )
            if last_loss < best_loss or not self._best_path().exists():
                self._save_best()

        if self.runtime.is_distributed:
            barrier()
            cleanup_distributed()

    def _best_path(self):
        return self.ckpt_manager.save_dir / "best_model.pt"

    def _save_best(self):
        """Simpan model terbaik terpisah."""
        best_path = self._best_path()
        torch.save(unwrap_model(self.model).state_dict(), best_path)
        print(f"🏆 Best model saved!")
