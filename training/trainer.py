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
        """
        Main training loop.
        Analoginya ke Laravel queue worker:
        while ada job → proses → simpan state.
        """
        dataloader = self._build_dataloader(dataset)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda"))

        global_step = 0
        best_loss   = float("inf")

        # Resume dari checkpoint kalau ada
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

        # ── Training Loop ──────────────────────────────────────
        self.optimizer.zero_grad()
        epoch = 0

        while global_step < max_steps:
            epoch += 1

            for batch_idx, (x, y) in enumerate(dataloader):
                if global_step >= max_steps:
                    break

                # Forward + backward
                loss = self.train_step(x, y)

                # Update weights setiap accum_steps
                if (batch_idx + 1) % self.accum_steps == 0:

                    # Gradient clipping — cegah exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # LR scheduler step
                    lr = self.scheduler.step()

                    # Hitung throughput
                    tokens_per_step = (
                        self.config["batch_size"]
                        * self.config["max_seq_len"]
                        * self.accum_steps
                    )

                    # Logging
                    self.logger.log(global_step, loss, lr, tokens_per_step)

                    # Checkpoint
                    save_every = self.config.get("save_every", 500)
                    if global_step > 0 and global_step % save_every == 0:
                        self.ckpt_manager.save(
                            self.model, self.optimizer, self.scheduler,
                            step=global_step, loss=loss,
                            config=self.config,
                        )
                        if loss < best_loss:
                            best_loss = loss
                            self._save_best()

                    global_step += 1

        self.logger.summary(global_step)
        # Final save
        self.ckpt_manager.save(
            self.model, self.optimizer, self.scheduler,
            step=global_step, loss=loss, config=self.config,
        )

    def _save_best(self):
        """Simpan model terbaik terpisah."""
        best_path = f"{self.config['checkpoint_dir']}/best_model.pt"
        torch.save(self.model.state_dict(), best_path)
        print(f"🏆 Best model saved!")