# lora/merge.py
"""
Utility functions untuk merge LoRA weights ke base model.
Setelah merge, tidak ada lagi LoRA overhead saat inference.

Formula merge:
  W_merged = W_original + (lora_B @ lora_A) * scaling
           = W_original + (d_out×r @ r×d_in) * (alpha/r)
"""
import torch
import torch.nn as nn
import copy
from pathlib import Path
from typing import Optional

from .layer  import LoRALinear
from .model  import LoRAModel
from .config import LoRAConfig


def merge_lora_to_base(
    lora_model: LoRAModel,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Merge semua LoRA adapters ke dalam base model weights.

    Args:
        lora_model : LoRAModel yang sudah ditraining
        save_path  : kalau diisi, save merged model ke path ini
        verbose    : print info merge

    Returns:
        nn.Module — base model dengan LoRA ter-merge (pure nn.Linear)
    """
    merged_model = copy.deepcopy(lora_model.base_model)
    n_merged     = 0

    def _merge_recursive(merged_module: nn.Module,
                         lora_module: nn.Module,
                         prefix: str = ""):
        nonlocal n_merged

        for name, child in lora_module.named_children():
            full_name    = f"{prefix}.{name}" if prefix else name
            merged_child = getattr(merged_module, name)

            if isinstance(child, LoRALinear):
                # Merge: W_new = W + (B @ A) * scaling
                delta_w = (child.lora_B @ child.lora_A) * child.scaling

                merged_linear = nn.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                )
                merged_linear.weight.data = child.weight.data + delta_w
                if child.bias is not None:
                    merged_linear.bias.data = child.bias.data

                setattr(merged_module, name, merged_linear)
                n_merged += 1

                if verbose:
                    delta_norm = delta_w.norm().item()
                    print(f"  ✅ Merged: {full_name:<45} | ΔW norm: {delta_norm:.4f}")

            else:
                # Rekursif ke child modules
                _merge_recursive(merged_child, child, full_name)

    _merge_recursive(merged_model, lora_model.base_model)

    if verbose:
        print(f"\n🔀 Total layers merged: {n_merged}")

    # Pastikan semua params trainable di merged model
    for param in merged_model.parameters():
        param.requires_grad = True

    # Save kalau diminta
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged_model.state_dict(), save_path)
        size_mb = sum(
            p.nelement() * p.element_size()
            for p in merged_model.parameters()
        ) / 1e6
        print(f"💾 Merged model saved: {save_path} ({size_mb:.1f}MB)")

    return merged_model


def load_and_merge(
    base_model: nn.Module,
    lora_config: LoRAConfig,
    lora_checkpoint: str,
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> nn.Module:
    """
    Shortcut: load LoRA checkpoint lalu langsung merge.

    Args:
        base_model       : model pretrained yang sudah diload
        lora_config      : LoRAConfig yang dipakai saat training
        lora_checkpoint  : path ke file lora_*.pt
        save_path        : opsional, simpan hasil merge
        verbose          : print info

    Returns:
        Merged model siap deploy
    """
    from .model import LoRAModel

    print(f"📦 Loading LoRA checkpoint: {lora_checkpoint}")
    lora_model = LoRAModel(base_model, lora_config)
    lora_model.load_lora(lora_checkpoint)

    print("🔀 Merging LoRA into base model...")
    merged = merge_lora_to_base(lora_model, save_path, verbose)
    print("✅ Merge complete!")

    return merged


def compare_weights(
    original_model: nn.Module,
    merged_model: nn.Module,
    n_layers: int = 3,
):
    """
    Debug utility: bandingkan weight original vs merged.
    Berguna untuk verifikasi merge berhasil.
    """
    print("\n📐 Weight comparison (original vs merged):\n")

    orig_params   = dict(original_model.named_parameters())
    merged_params = dict(merged_model.named_parameters())

    count = 0
    for name, orig_p in orig_params.items():
        if name not in merged_params:
            continue
        merged_p = merged_params[name]
        delta    = (merged_p - orig_p).norm().item()
        if delta > 1e-6:  # hanya tampilkan yang berubah
            print(f"  {name:<50} | Δnorm: {delta:.6f}")
            count += 1
            if count >= n_layers:
                remaining = sum(
                    1 for n, p in orig_params.items()
                    if n in merged_params and
                    (merged_params[n] - p).norm().item() > 1e-6
                ) - n_layers
                if remaining > 0:
                    print(f"  ... dan {remaining} layer lainnya")
                break

    if count == 0:
        print("  ⚠️  Tidak ada perbedaan weight terdeteksi!")
    else:
        print(f"\n  Total layer yang berubah: ~{count}+")