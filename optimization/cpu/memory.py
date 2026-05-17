from contextlib import contextmanager
import gc

import psutil
import torch


class MemoryManager:
    """Utilities for checking and clearing RAM during CPU-heavy runs."""

    RAM_LIMIT_GB = 3.0

    @staticmethod
    def current_usage_gb() -> float:
        return psutil.Process().memory_info().rss / 1e9

    @staticmethod
    def available_gb() -> float:
        return psutil.virtual_memory().available / 1e9

    @classmethod
    def check(cls, label: str = "") -> None:
        used = cls.current_usage_gb()
        avail = cls.available_gb()
        print(f"RAM [{label}]: used={used:.2f} GB | avail={avail:.2f} GB")
        if used > cls.RAM_LIMIT_GB:
            print("WARNING: RAM usage is high, clearing cache...")
            cls.clear()

    @staticmethod
    def clear() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def track(self, label: str):
        before = self.current_usage_gb()
        yield
        after = self.current_usage_gb()
        delta = after - before
        print(f"[{label}] RAM delta: {delta:+.2f} GB (total: {after:.2f} GB)")


def load_model_efficient(model_class, config, checkpoint_path: str):
    """Load a checkpoint with lower peak RAM usage on CPU."""
    mem = MemoryManager()

    with mem.track("model_load"):
        with torch.device("meta"):
            model = model_class(config)

        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict, assign=True)

    model.eval()
    mem.check("after_load")
    return model

