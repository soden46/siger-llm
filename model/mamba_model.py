# model/mamba_model.py
import torch.nn as nn

class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack N SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(config) for _ in range(config.n_layers)
        ])

        self.norm_f = nn.LayerNorm(config.d_model)

        # LM Head: project ke vocab (weight tying optional)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying (hemat parameter, trik umum)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)  # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss