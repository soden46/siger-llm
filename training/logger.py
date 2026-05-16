# training/logger.py
import time
from collections import deque


class TrainingLogger:
    """
    Simple logger buat track metrics.
    Bisa diextend ke WandB/TensorBoard nanti.
    """
    def __init__(self, log_interval: int = 10, window: int = 100):
        self.log_interval = log_interval
        self.loss_window  = deque(maxlen=window)  # rolling average
        self.start_time   = time.time()
        self.step_times   = deque(maxlen=50)
        self._last_time   = time.time()

    def log(self, step: int, loss: float, lr: float, tokens_per_sec: float = 0):
        self.loss_window.append(loss)
        now = time.time()
        self.step_times.append(now - self._last_time)
        self._last_time = now

        if step % self.log_interval == 0:
            avg_loss   = sum(self.loss_window) / len(self.loss_window)
            avg_step_t = sum(self.step_times) / len(self.step_times)
            elapsed    = now - self.start_time
            perplexity = min(2 ** avg_loss, 99999)  # PPL = 2^loss

            print(
                f"step={step:>7,} | "
                f"loss={loss:.4f} | "
                f"avg_loss={avg_loss:.4f} | "
                f"ppl={perplexity:.1f} | "
                f"lr={lr:.2e} | "
                f"tok/s={tokens_per_sec:,.0f} | "
                f"elapsed={elapsed/60:.1f}m"
            )

    def summary(self, total_steps: int):
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.loss_window) / max(len(self.loss_window), 1)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Steps   : {total_steps:,}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  PPL     : {2**avg_loss:.2f}")
        print(f"  Time    : {elapsed/3600:.2f}h")
        print(f"{'='*60}")