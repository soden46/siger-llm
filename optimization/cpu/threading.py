import os

import psutil
import torch


def configure_cpu(n_cores: int = 2) -> None:
    """Configure PyTorch and BLAS threading for CPU-only runs."""
    n_cores = max(1, min(n_cores, psutil.cpu_count(logical=True) or 1))

    torch.set_num_threads(n_cores)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    os.environ["OMP_NUM_THREADS"] = str(n_cores)
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
    os.environ["PYTORCH_JIT"] = "1"

    torch.backends.cudnn.enabled = False

    print(f"CPU configured: {n_cores} threads")
    print(f"  available cores: {psutil.cpu_count(logical=True)}")
    print(f"  available RAM  : {psutil.virtual_memory().available / 1e9:.1f} GB")

