# optimization/quantization/quantize.py
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from pathlib import Path
import os


class ModelQuantizer:
    """
    3 level quantization, pilih sesuai kebutuhan:
    
    INT8 Dynamic  → 4x lebih kecil, ~95% quality, paling gampang
    INT8 Static   → lebih cepat dari dynamic, perlu calibration data
    INT4 GPTQ     → 8x lebih kecil, ~90% quality, paling kecil
    """

    def __init__(self, model, save_dir: str = "./checkpoints/quantized"):
        self.model    = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── INT8 Dynamic (recommended buat lo) ────────────────
    def quantize_int8_dynamic(self):
        """
        Dynamic quantization: weight di-quantize, activation runtime.
        Zero calibration needed — langsung jalan.
        Cocok banget buat CPU inference.
        """
        print("⚙️  Applying INT8 Dynamic Quantization...")

        model_fp32 = self.model.cpu()
        model_fp32.eval()

        # Quantize semua layer Linear & Embedding
        quantized = quantize_dynamic(
            model_fp32,
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )

        # Hitung size reduction
        original_size = self._model_size_mb(self.model)
        quantized_size = self._model_size_mb(quantized)

        print(f"✅ INT8 Dynamic done!")
        print(f"   Original : {original_size:.1f} MB")
        print(f"   Quantized: {quantized_size:.1f} MB")
        print(f"   Reduction: {original_size/quantized_size:.1f}x smaller")

        # Save
        path = self.save_dir / "model_int8_dynamic.pt"
        torch.save(quantized.state_dict(), path)
        print(f"💾 Saved: {path}")

        return quantized

    # ── INT8 Static (lebih cepat, perlu calibration) ──────
    def quantize_int8_static(self, calibration_loader):
        """
        Static quantization: observer jalan dulu di calibration data,
        lalu scale factor di-freeze. Lebih cepat saat inference.
        """
        print("⚙️  Applying INT8 Static Quantization...")

        model = self.model.cpu().eval()

        # Setup qconfig
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # CPU x86

        # Fuse layers yang bisa digabung (Linear + ReLU, dsb)
        # Ini bikin inference lebih cepat karena kurang kernel call
        torch.quantization.fuse_modules(
            model,
            [["norm", "ssm"]],   # sesuaikan dengan layer lo
            inplace=True
        )

        # Prepare: insert observer
        torch.quantization.prepare(model, inplace=True)

        # Calibration: jalanin beberapa batch tanpa gradient
        print("📊 Running calibration...")
        with torch.no_grad():
            for i, (x, _) in enumerate(calibration_loader):
                model(x)
                if i >= 100:   # 100 batch cukup
                    break

        # Convert ke quantized
        torch.quantization.convert(model, inplace=True)

        original_size  = self._model_size_mb(self.model)
        quantized_size = self._model_size_mb(model)
        print(f"✅ INT8 Static done! {original_size:.1f}MB → {quantized_size:.1f}MB")

        path = self.save_dir / "model_int8_static.pt"
        torch.save(model.state_dict(), path)
        return model

    # ── INT4 via bitsandbytes (paling kecil) ──────────────
    def quantize_int4_bnb(self):
        """
        4-bit quantization via bitsandbytes.
        Butuh: pip install bitsandbytes
        Paling agresif — model bisa 8x lebih kecil.
        """
        try:
            import bitsandbytes as bnb
        except ImportError:
            print("❌ pip install bitsandbytes")
            return None

        print("⚙️  Applying INT4 Quantization (bitsandbytes)...")

        # Replace semua nn.Linear dengan 4-bit version
        def replace_linear_4bit(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float32,   # compute dalam fp32
                        compress_statistics=True,
                        quant_type="nf4",              # NormalFloat4 — paling bagus
                    )
                    new_layer.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type="nf4"
                    )
                    setattr(module, name, new_layer)
                else:
                    replace_linear_4bit(child)

        model = self.model.cpu()
        replace_linear_4bit(model)

        size = self._model_size_mb(model)
        print(f"✅ INT4 NF4 done! Model size: {size:.1f}MB")

        path = self.save_dir / "model_int4.pt"
        torch.save(model.state_dict(), path)
        return model

    @staticmethod
    def _model_size_mb(model) -> float:
        total = sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
        )
        return total / (1024 ** 2)