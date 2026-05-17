# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .optimizer   import build_optimizer, CosineScheduler
from .checkpoint  import CheckpointManager
from .logger      import TrainingLogger
from .dataset     import TextDataset
from tokenizer.tokenizer import MultilingualTokenizer


class Trainer:
    def __init__(self, model, config: dict, device: str = None):
        self.model  = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        print(f"🖥️  Device: {self.device}")

        # Komponen training
        self.optimizer = build_optimizer(
            model,
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
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=(self.device == "cuda"),
            drop_last=True,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single forward + backward step."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Mixed precision (otomatis kalau CUDA)
        with torch.autocast(device_type=self.device, dtype=torch.float16,
                            enabled=(self.device == "cuda")):
            _, loss = self.model(x, targets=y)
            loss = loss / self.accum_steps  # scale buat grad accum

        loss.backward()
        return loss.item() * self.accum_steps  # return unscaled loss

    def train(
        self,
        dataset: TextDataset,
        resume: bool = True,
    ):
        """Main training loop."""
        dataloader = self._build_dataloader(dataset)
        
        # 1. Pastikan device_type string untuk GradScaler
        device_type = "cuda" if "cuda" in self.device else "cpu"
        scaler = torch.amp.GradScaler(device_type, enabled=(device_type == "cuda"))
        
        global_step = 0
        best_loss = float("inf")

        if resume:
            global_step, best_loss = self.ckpt_manager.load(
                self.model, self.optimizer, self.scheduler
            )

        self.model.train()
        max_steps = self.config["max_steps"]
        print(f"\n🚀 Training starts | max_steps={max_steps:,}")
        print(f"   batch_size={self.config['batch_size']} | "
              f"grad_accum={self.accum_steps} | "
              f"effective_batch={self.config['batch_size'] * self.accum_steps}\n")

        self.optimizer.zero_grad()
        epoch = 0
        
        while global_step < max_steps:
            epoch += 1
            for batch_idx, (x, y) in enumerate(dataloader):
                if global_step >= max_steps:
                    break

                # Forward + Backward menggunakan AMP Scaler yang benar
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=(device_type == "cuda")):
                    _, loss = self.model(x, targets=y)
                    loss = loss / self.accum_steps

                # Skala loss untuk mencegah underflow gradien pada FP16
                scaler.scale(loss).backward()
                unscaled_loss = loss.item() * self.accum_steps

                # Update weights setiap accum_steps
                if (batch_idx + 1) % self.accum_steps == 0:
                    # Unscale gradien sebelum clipping
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                    # Optimizer step via Scaler
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                    # LR scheduler step
                    lr = self.scheduler.step()

                    # Hitung throughput tokens
                    tokens_per_step = (
                        self.config["batch_size"] * self.config["max_seq_len"] * self.accum_steps
                    )

                    # Logging
                    self.logger.log(global_step, unscaled_loss, lr, tokens_per_step)

                    # 2. PERBAIKAN BUG: Simpan checkpoint & best model HANYA pada interval tertentu
                    save_every = self.config.get("save_every", 500)
                    if global_step > 0 and global_step % save_every == 0:
                        self.ckpt_manager.save(
                            self.model, self.optimizer, self.scheduler,
                            step=global_step, loss=unscaled_loss, config=self.config,
                        )
                        
                        # Cek dan simpan model terbaik di interval ini saja untuk menghemat I/O disk
                        if unscaled_loss < best_loss:
                            best_loss = unscaled_loss
                            self._save_best()

                    global_step += 1

        self.logger.summary(global_step)
        # Final save
        self.ckpt_manager.save(
            self.model, self.optimizer, self.scheduler,
            step=global_step, loss=unscaled_loss, config=self.config,
        )

    def _save_best(self):
        """Simpan model terbaik terpisah."""
        best_path = f"{self.config['checkpoint_dir']}/best_model.pt"
        torch.save(self.model.state_dict(), best_path)
        print(f"🏆 Best model saved!")