# optimization/trainer.py
"""
Quantization-Aware Training (QAT) Trainer.

Berbeda dengan post-training quantization (PTQ) yang dilakukan setelah training,
QAT mensimulasikan efek quantization SELAMA training supaya model lebih robust
terhadap error quantization saat inference.

Kapan pakai QAT vs PTQ?
  PTQ (quantize.py) : model sudah jadi, tinggal quantize → cepat, cukup untuk banyak kasus
  QAT (file ini)    : kalau PTQ drop akurasi terlalu besar → fine-tune lagi dengan fake quant

Untuk VPS CPU lo, PTQ INT8 dynamic biasanya sudah cukup.
Pakai QAT kalau PPL naik > 10% setelah quantization.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional

from training.optimizer  import build_optimizer, CosineScheduler
from training.logger     import TrainingLogger
from training.checkpoint import CheckpointManager


class QATTrainer:
    """
    Quantization-Aware Training trainer.

    Cara kerja:
    1. Load model yang sudah ditraining
    2. Insert fake quantization nodes (simulasi INT8 error)
    3. Fine-tune beberapa steps supaya model adapt
    4. Export ke quantized model

    Fake quantization = simulasi rounding error INT8 tapi
    masih pakai FP32 untuk gradients (supaya bisa backprop).
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = "cpu",
    ):
        self.model  = model
        self.config = config
        self.device = device

        self.model.to(self.device)
        self._prepare_qat()

        self.optimizer = build_optimizer(
            model,
            lr=config.get("qat_lr", 1e-5),       # LR kecil untuk fine-tune
            weight_decay=config.get("weight_decay", 0.01),
        )
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps = config.get("warmup_steps", 50),
            max_steps    = config.get("qat_steps", 500),
            max_lr       = config.get("qat_lr", 1e-5),
            min_lr       = config.get("qat_lr", 1e-5) / 10,
        )
        self.logger = TrainingLogger(
            log_interval=config.get("log_interval", 10)
        )
        self.ckpt = CheckpointManager(
            save_dir  = config.get("checkpoint_dir", "./checkpoints/qat"),
            keep_last = 2,
        )

    def _prepare_qat(self):
        """
        Insert fake quantization observers ke model.
        Harus dipanggil sebelum training QAT dimulai.
        """
        self.model.train()

        # Set qconfig untuk CPU (fbgemm = AVX2 optimized x86)
        self.model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")

        # Prepare model untuk QAT — insert observer & fake quant nodes
        torch.quantization.prepare_qat(self.model, inplace=True)

        print("✅ QAT observers inserted")
        print("   qconfig : fbgemm (CPU x86)")

    def train(self, dataloader: DataLoader, n_steps: int = 500):
        """
        Jalankan QAT fine-tuning.

        Args:
            dataloader : DataLoader dari training dataset
            n_steps    : jumlah steps QAT (biasanya 200-1000)
        """
        self.model.train()
        step = 0
        self.optimizer.zero_grad()

        print(f"\n🔧 QAT Fine-tuning | steps={n_steps}")
        print(f"   LR : {self.config.get('qat_lr', 1e-5)}")
        print(f"   Tujuan: adapt model ke INT8 quantization error\n")

        while step < n_steps:
            for x, y in dataloader:
                if step >= n_steps:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                _, loss = self.model(x, targets=y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                lr = self.scheduler.step()
                self.logger.log(step, loss.item(), lr)

                step += 1

        print("✅ QAT fine-tuning selesai")

    def convert_to_quantized(self, save_path: Optional[str] = None) -> nn.Module:
        """
        Convert model dari QAT (fake quant) ke model INT8 yang sesungguhnya.
        Harus dipanggil SETELAH training QAT selesai.

        Args:
            save_path : opsional, path untuk save quantized model

        Returns:
            INT8 quantized model
        """
        self.model.eval()

        # Convert fake quant → real INT8 quantized weights
        quantized_model = torch.quantization.convert(self.model, inplace=False)

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(quantized_model.state_dict(), save_path)

            size_mb = sum(
                p.nelement() * p.element_size()
                for p in quantized_model.parameters()
            ) / 1e6
            print(f"💾 QAT model saved: {save_path} ({size_mb:.1f}MB)")

        return quantized_model


class BenchmarkComparator:
    """
    Utility untuk compare kualitas model sebelum vs sesudah quantization.
    Pastikan quantization tidak merusak output terlalu banyak.
    """

    @staticmethod
    def compare_outputs(
        model_fp32: nn.Module,
        model_int8: nn.Module,
        sample_input: torch.Tensor,
        device: str = "cpu",
    ) -> dict:
        """
        Bandingkan output FP32 vs INT8 untuk input yang sama.

        Returns:
            dict berisi statistik perbedaan output
        """
        model_fp32.eval()
        model_int8.eval()

        with torch.no_grad():
            out_fp32, _ = model_fp32(sample_input.to(device))
            out_int8, _ = model_int8(sample_input.to(device))

        # Hitung perbedaan
        diff     = (out_fp32 - out_int8).abs()
        max_diff = diff.max().item()
        avg_diff = diff.mean().item()

        # Cosine similarity per posisi
        cos_sim = nn.functional.cosine_similarity(
            out_fp32.view(-1, out_fp32.size(-1)),
            out_int8.view(-1, out_int8.size(-1)),
            dim=-1,
        ).mean().item()

        result = {
            "max_diff":   round(max_diff, 6),
            "avg_diff":   round(avg_diff, 6),
            "cos_sim":    round(cos_sim, 4),
            "grade":      "🟢 OK" if cos_sim > 0.99 else
                          "🟡 Acceptable" if cos_sim > 0.97 else
                          "🔴 Degraded",
        }

        print(f"\n📊 FP32 vs INT8 Output Comparison:")
        print(f"   Max diff   : {result['max_diff']}")
        print(f"   Avg diff   : {result['avg_diff']}")
        print(f"   Cosine sim : {result['cos_sim']}  {result['grade']}")

        return result

    @staticmethod
    def compare_speed(
        model_fp32: nn.Module,
        model_int8: nn.Module,
        sample_input: torch.Tensor,
        n_runs: int = 10,
        device: str = "cpu",
    ) -> dict:
        """
        Bandingkan kecepatan inference FP32 vs INT8.
        """
        import time

        model_fp32.eval()
        model_int8.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model_fp32(sample_input)
                model_int8(sample_input)

        # Benchmark FP32
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model_fp32(sample_input)
        fp32_time = (time.perf_counter() - t0) / n_runs

        # Benchmark INT8
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_runs):
                model_int8(sample_input)
        int8_time = (time.perf_counter() - t0) / n_runs

        speedup = fp32_time / max(int8_time, 1e-9)

        result = {
            "fp32_ms": round(fp32_time * 1000, 2),
            "int8_ms": round(int8_time * 1000, 2),
            "speedup": round(speedup, 2),
        }

        print(f"\n⚡ Speed Comparison (avg over {n_runs} runs):")
        print(f"   FP32 : {result['fp32_ms']}ms")
        print(f"   INT8 : {result['int8_ms']}ms")
        print(f"   Speedup: {result['speedup']}×")

        return result