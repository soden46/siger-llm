# optimization/benchmark.py
import torch
import time
import psutil
from dataclasses import dataclass


@dataclass
class BenchResult:
    tokens_per_sec: float
    latency_ms:     float
    ram_mb:         float
    model_size_mb:  float


def benchmark(generator, prompt: str = "Halo, apa kabar?",
              n_tokens: int = 50, n_runs: int = 3) -> BenchResult:
    """Ukur kecepatan dan RAM usage."""

    ram_before = psutil.Process().memory_info().rss / 1e6

    latencies = []
    for i in range(n_runs):
        start = time.perf_counter()
        out   = generator.generate(prompt, max_new_tokens=n_tokens)
        end   = time.perf_counter()
        latencies.append(end - start)
        print(f"Run {i+1}: {n_tokens/(end-start):.1f} tok/s | '{out[:40]}...'")

    ram_after = psutil.Process().memory_info().rss / 1e6
    avg_lat   = sum(latencies) / n_runs

    result = BenchResult(
        tokens_per_sec = n_tokens / avg_lat,
        latency_ms     = avg_lat * 1000,
        ram_mb         = ram_after,
        model_size_mb  = ram_after - ram_before,
    )

    print(f"\n{'='*40}")
    print(f"⚡ {result.tokens_per_sec:.1f} tokens/sec")
    print(f"⏱️  {result.latency_ms:.0f}ms per {n_tokens} tokens")
    print(f"🧠 RAM: {result.ram_mb:.0f}MB")
    print(f"{'='*40}")
    return result