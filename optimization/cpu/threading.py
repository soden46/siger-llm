# optimization/cpu/threading.py
import torch
import os
import psutil


def configure_cpu(n_cores: int = 2):
    """
    Setup optimal CPU config buat VPS 2 core.
    Harus dipanggil SEBELUM import model.
    """

    # PyTorch threads
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(1)

    # OpenMP (used by PyTorch internals)
    os.environ["OMP_NUM_THREADS"]        = str(n_cores)
    os.environ["MKL_NUM_THREADS"]        = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(n_cores)

    # Disable unnecessary torch features di CPU
    torch.backends.cudnn.enabled = False

    # Enable CPU optimizations
    # AVX2/AVX512 auto-detected, tapi bisa di-force:
    os.environ["PYTORCH_JIT"] = "1"

    print(f"✅ CPU configured: {n_cores} threads")
    print(f"   Available cores: {psutil.cpu_count()}")
    print(f"   Available RAM  : {psutil.virtual_memory().available / 1e9:.1f}GB")


# optimization/cpu/memory.py
import torch
import gc
import psutil
from contextlib import contextmanager


class MemoryManager:
    """
    Kelola RAM ketat buat VPS 4GB.
    """

    RAM_LIMIT_GB = 3.0   # batas safety sebelum OOM

    @staticmethod
    def current_usage_gb() -> float:
        return psutil.Process().memory_info().rss / 1e9

    @staticmethod
    def available_gb() -> float:
        return psutil.virtual_memory().available / 1e9

    @classmethod
    def check(cls, label: str = ""):
        used = cls.current_usage_gb()
        avail = cls.available_gb()
        print(f"🧠 RAM [{label}]: used={used:.2f}GB | avail={avail:.2f}GB")
        if used > cls.RAM_LIMIT_GB:
            print("⚠️  WARNING: RAM usage tinggi, clearing cache...")
            cls.clear()

    @staticmethod
    def clear():
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    @contextmanager
    def track(self, label: str):
        """Context manager buat track RAM usage per operasi."""
        before = self.current_usage_gb()
        yield
        after = self.current_usage_gb()
        delta = after - before
        print(f"📊 [{label}] RAM delta: {delta:+.2f}GB (total: {after:.2f}GB)")


def load_model_efficient(model_class, config, checkpoint_path: str):
    """
    Load model dengan RAM footprint minimal.
    Pakai map_location='cpu' + lazy loading.
    """
    mem = MemoryManager()

    with mem.track("model_load"):
        # Init model dengan empty weights dulu (hemat RAM saat load)
        with torch.device("meta"):
            model = model_class(config)

        # Load weights langsung ke CPU tanpa double memory
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,    # security + hemat RAM
        )
        model.load_state_dict(state_dict, assign=True)

    model.eval()
    mem.check("after_load")
    return model