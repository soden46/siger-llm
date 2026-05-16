# lora/model.py
import torch
import torch.nn as nn
from typing import Dict, List
from .layer  import LoRALinear
from .config import LoRAConfig


class LoRAModel(nn.Module):
    """
    Wrapper yang inject LoRA ke SigerLM.

    Cara kerja:
    1. Freeze SEMUA weight base model
    2. Replace target Linear layers dengan LoRALinear
    3. Hanya LoRA params yang di-train
    """

    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config     = config
        self.lora_layers: Dict[str, LoRALinear] = {}

        # Step 1: Freeze semua params
        self._freeze_base()

        # Step 2: Inject LoRA
        self._inject_lora()

        # Step 3: Print summary
        self._print_summary()

    def _freeze_base(self):
        """Freeze semua parameter base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _inject_lora(self):
        """
        Ganti target Linear layers dengan LoRALinear.
        Traversal rekursif kayak tree traversal.
        """
        def _replace(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Cek apakah nama layer masuk target
                is_target = any(
                    t in name for t in self.config.target_modules
                )

                if is_target and isinstance(child, nn.Linear):
                    # Replace dengan LoRALinear
                    lora_layer = LoRALinear(
                        child,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    )
                    setattr(module, name, lora_layer)
                    self.lora_layers[full_name] = lora_layer
                else:
                    _replace(child, full_name)

        _replace(self.base_model)

    def _print_summary(self):
        total_params    = sum(p.numel() for p in self.parameters())
        trainable       = sum(p.numel() for p in self.parameters()
                              if p.requires_grad)
        pct             = 100 * trainable / total_params

        print(f"\n{'='*50}")
        print(f"LoRA Model Summary")
        print(f"{'='*50}")
        print(f"Base model params : {total_params - trainable:>12,}")
        print(f"LoRA params       : {trainable:>12,}  ({pct:.2f}%)")
        print(f"Total params      : {total_params:>12,}")
        print(f"LoRA layers       : {len(self.lora_layers)}")
        print(f"Rank / Alpha      : {self.config.rank} / {self.config.alpha}")
        print(f"{'='*50}\n")
        for name in self.lora_layers:
            print(f"  ✅ {name}")
        print()

    def forward(self, input_ids, targets=None):
        return self.base_model(input_ids, targets)

    # ── Save & Load ───────────────────────────────────────

    def save_lora(self, path: str):
        """Simpan HANYA LoRA weights — kecil banget (< 50MB)."""
        lora_state = {
            name: {
                "lora_A": layer.lora_A.data,
                "lora_B": layer.lora_B.data,
            }
            for name, layer in self.lora_layers.items()
        }
        torch.save({"lora_state": lora_state, "config": self.config}, path)
        size_mb = sum(
            v["lora_A"].nelement() * v["lora_A"].element_size() +
            v["lora_B"].nelement() * v["lora_B"].element_size()
            for v in lora_state.values()
        ) / 1e6
        print(f"💾 LoRA saved: {path} ({size_mb:.1f}MB)")

    def load_lora(self, path: str):
        """Load LoRA weights ke model yang udah ada."""
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        lora_state = ckpt["lora_state"]

        for name, weights in lora_state.items():
            if name in self.lora_layers:
                self.lora_layers[name].lora_A.data = weights["lora_A"]
                self.lora_layers[name].lora_B.data = weights["lora_B"]
            else:
                print(f"⚠️  Layer not found: {name}")

        print(f"✅ LoRA loaded from {path}")

    def merge_and_export(self, save_path: str):
        """
        Merge LoRA → base model, export sebagai model biasa.
        Hasilnya bisa di-deploy tanpa LoRA overhead.
        """
        import copy
        merged_model = copy.deepcopy(self.base_model)

        def _merge(module, ref_module, prefix=""):
            for name, child in ref_module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in self.lora_layers:
                    merged_linear = self.lora_layers[full_name].merge_weights()
                    setattr(module, name, merged_linear)
                else:
                    _merge(
                        getattr(module, name),
                        child, full_name
                    )

        _merge(merged_model, self.base_model)
        torch.save(merged_model.state_dict(), save_path)
        print(f"✅ Merged model saved: {save_path}")
        return merged_model