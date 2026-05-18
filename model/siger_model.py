# model/siger_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.ssm_block import SSMBlock


class SigerLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token embedding: token_id → hidden representation
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Stack N SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(config) for _ in range(config.n_layers)
        ])

        # Final normalization sebelum LM head
        self.norm_f = nn.LayerNorm(config.d_model)

        # LM Head: hidden state → vocab logits
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
        )

        self.apply(self._init_weights)

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
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

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

        return logits, loss
