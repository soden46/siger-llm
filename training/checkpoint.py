# training/checkpoint.py
import torch
import os
import json
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    Save & load model checkpoint.
    Kayak git commit — tiap N step lo save state.
    """
    def __init__(self, save_dir: str, keep_last: int = 3):
        self.save_dir  = Path(save_dir)
        self.keep_last = keep_last
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[str] = []

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        loss: float,
        config: dict,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"step_{step:07d}_{timestamp}.pt"
        ckpt_path = self.save_dir / ckpt_name

        torch.save({
            "step":            step,
            "loss":            loss,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step":  scheduler.current_step,
            "config":          config,
        }, ckpt_path)

        # Simpan metadata
        meta_path = self.save_dir / "latest.json"
        with open(meta_path, "w") as f:
            json.dump({"latest": ckpt_name, "step": step, "loss": loss}, f)

        self.history.append(str(ckpt_path))
        print(f"💾 Saved checkpoint: {ckpt_name} | loss={loss:.4f}")

        # Hapus checkpoint lama
        self._cleanup()

        return str(ckpt_path)

    def load(self, model, optimizer=None, scheduler=None, path: str = None):
        """Load checkpoint. Kalau path=None, load yang paling baru."""
        if path is None:
            meta_path = self.save_dir / "latest.json"
            if not meta_path.exists():
                print("⚠️  No checkpoint found, starting fresh.")
                return 0, float("inf")
            with open(meta_path) as f:
                meta = json.load(f)
            path = self.save_dir / meta["latest"]

        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler and "scheduler_step" in ckpt:
            scheduler.current_step = ckpt["scheduler_step"]

        step = ckpt.get("step", 0)
        loss = ckpt.get("loss", float("inf"))
        print(f"✅ Loaded checkpoint: step={step} | loss={loss:.4f}")
        return step, loss

    def _cleanup(self):
        if len(self.history) > self.keep_last:
            old = self.history.pop(0)
            if os.path.exists(old):
                os.remove(old)