# model/siger_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.checkpoint import checkpoint

from model.ssm_block import SSMBlock
from model.norms import RMSNorm, build_norm


class SigerLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_name = getattr(config, "model_name", "SIGER")

        # Token embedding: token_id → hidden representation
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Stack N SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])

        # Final normalization sebelum LM head
        self.norm_f = build_norm(
            config.d_model,
            norm_type=getattr(config, "norm_type", "rmsnorm"),
            eps=getattr(config, "norm_eps", 1e-6),
            bias=getattr(config, "norm_bias", False),
        )

        # LM Head: hidden state → vocab logits
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)
        self._init_selective_dt()
        self._scale_residual_projections()

        # Weight tying:
        # embedding weight dan output projection weight dibagi
        # supaya parameter lebih hemat dan representasi lebih konsisten.
        self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            pad_token_id = getattr(self.config, "pad_token_id", None)
            if pad_token_id is not None and 0 <= pad_token_id < module.num_embeddings:
                with torch.no_grad():
                    module.weight[pad_token_id].zero_()
        elif isinstance(module, (nn.LayerNorm, RMSNorm)):
            nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_projections(self):
        if not getattr(self.config, "residual_scale_init", True):
            return
        scale = 1.0 / math.sqrt(2.0 * max(1, int(getattr(self.config, "n_layers", 1))))
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name.endswith("out_proj.weight"):
                    param.mul_(scale)

    def _init_selective_dt(self):
        for module in self.modules():
            reset = getattr(module, "reset_dt_parameters", None)
            if callable(reset):
                reset()

    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: Tensor shape (B, L)
            targets:   Tensor shape (B, L), optional

        Returns:
            logits: Tensor shape (B, L, vocab_size)
            loss:   Cross entropy loss jika targets diberikan, else None
        """

        # (B, L) → (B, L, d_model)
        x = self.embedding(input_ids)

        # Lewat semua SSM blocks
        for layer in self.layers:
            if self.training and getattr(self.config, "gradient_checkpointing", False) and x.requires_grad:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # Final normalization
        x = self.norm_f(x)

        # (B, L, d_model) → (B, L, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            moe_aux_weight = float(getattr(self.config, "moe_aux_loss_weight", 0.0))
            if moe_aux_weight > 0:
                aux_losses = [
                    layer.last_moe_loss
                    for layer in self.layers
                    if getattr(layer, "last_moe_loss", None) is not None
                ]
                if aux_losses:
                    loss = loss + moe_aux_weight * torch.stack(aux_losses).mean()

        return logits, loss
