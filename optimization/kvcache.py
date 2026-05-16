# optimization/kvcache.py
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SSMCache:
    """
    Cache SSM hidden state antar token generation.
    
    Tanpa cache: tiap token, lo recompute SELURUH sequence dari awal.
    Dengan cache: lo cuma compute 1 token baru, state-nya dilanjutin.
    
    Ini yang bikin generation 10-50x lebih cepat!
    """
    # Hidden state SSM per layer: (n_layers, batch, d_inner, d_state)
    h: Optional[torch.Tensor] = None
    # Conv state per layer: (n_layers, batch, d_inner, d_conv)
    conv: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        return self.h is None


class CachedSSMBlock(nn.Module):
    """
    SSM Block yang support incremental decode dengan cache.
    Saat prefill (prompt): jalanin full sequence sekali.
    Saat decode (generate): jalanin 1 token, update cache.
    """

    def forward(
        self,
        x: torch.Tensor,              # (B, L, d_model)
        cache: Optional[SSMCache] = None,
        use_cache: bool = False,
    ):
        is_decoding = use_cache and cache and not cache.is_empty()

        if is_decoding:
            # Decode mode: x = (B, 1, d_model) — 1 token aja
            return self._decode_step(x, cache)
        else:
            # Prefill mode: proses full prompt
            return self._prefill(x, cache, use_cache)

    def _prefill(self, x, cache, use_cache):
        """Full forward pass untuk prompt."""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Conv — simpan state akhir untuk decode
        x_conv_t = x_branch.transpose(1, 2)   # (B, D, L)
        x_conv   = self.conv1d(x_conv_t)
        x_conv   = x_conv[:, :, :x_branch.size(1)].transpose(1, 2)
        x_conv   = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z_gate)
        out = self.out_proj(y) + residual

        # Simpan cache
        if use_cache and cache is not None:
            # Simpan conv state (D, d_conv terakhir)
            cache.conv = x_conv_t[:, :, -self.conv1d.kernel_size[0]:]
            # Simpan SSM hidden state terakhir
            cache.h = self.ssm._last_h   # perlu diexpose dari SSMCore

        return out

    def _decode_step(self, x, cache):
        """Single token decode — cuma hitung 1 step."""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)   # (B, 1, D)

        # Conv dengan cache: geser window, append token baru
        # cache.conv: (B, D, d_conv-1)
        x_t   = x_branch.transpose(1, 2)          # (B, D, 1)
        conv_input = torch.cat([cache.conv, x_t], dim=-1)  # (B, D, d_conv)
        x_conv = (conv_input * self.conv1d.weight.squeeze(1)).sum(-1, keepdim=True)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias.unsqueeze(-1)
        x_conv = F.silu(x_conv.transpose(1, 2))   # (B, 1, D)

        # Update conv cache
        cache.conv = conv_input[:, :, 1:]          # geser 1

        # SSM single step
        y, new_h  = self.ssm.step(x_conv, cache.h)
        cache.h   = new_h

        y   = y * F.silu(z_gate)
        out = self.out_proj(y) + residual
        return out