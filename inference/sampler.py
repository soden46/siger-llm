# inference/sampler.py
import torch
import torch.nn.functional as F
from typing import Optional


class Sampler:
    """
    Kumpulan strategi sampling dari logits → next token.
    
    Analoginya ke Laravel:
    - Greedy  = ambil value tertinggi langsung (array_key_first)
    - Top-K   = filter 5 kandidat terbaik, random dari situ
    - Top-P   = filter sampai cumulative prob > threshold
    - Temp    = "seberapa random" outputnya
    """

    @staticmethod
    def greedy(logits: torch.Tensor) -> int:
        """Selalu ambil token dengan prob tertinggi. Deterministic."""
        return logits.argmax(dim=-1).item()

    @staticmethod
    def temperature(logits: torch.Tensor, temp: float) -> torch.Tensor:
        """
        Scale logits sebelum softmax.
        temp < 1.0 → lebih fokus / repetitif
        temp > 1.0 → lebih random / kreatif
        temp = 1.0 → normal
        """
        if temp <= 0:
            raise ValueError("Temperature harus > 0")
        return logits / temp

    @staticmethod
    def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Buang semua token kecuali K terbesar.
        Set token di luar top-K ke -inf (jadi prob=0 setelah softmax).
        """
        if k <= 0:
            return logits
        k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[..., -1, None]  # nilai terkecil dari top-K
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Nucleus sampling: ambil token sampai cumulative prob >= p.
        Lebih adaptif dari top-K karena jumlah token bisa beda tiap step.
        """
        if p >= 1.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Tandai token yang bikin cumulative prob > p
        remove_mask = cumulative - sorted_probs > p
        sorted_probs[remove_mask] = 0.0

        # Kembalikan ke urutan asli
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_indices, sorted_probs)

        # Convert balik ke logits space
        filtered = filtered.log().clamp(min=-1e9)
        return filtered

    @staticmethod
    def repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float = 1.3
    ) -> torch.Tensor:
        """
        Kurangi probabilitas token yang udah sering muncul.
        Cegah model ngulang-ngulang kata yang sama.
        penalty > 1.0 = makin dihukum kalau udah pernah muncul.
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    @classmethod
    def sample(
        cls,
        logits: torch.Tensor,       # (vocab_size,)
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        penalty: float = 1.0,
        generated_ids: list = None,
        do_sample: bool = True,
    ) -> int:
        """
        Pipeline lengkap: logits → 1 token ID.
        Urutan: temp → rep penalty → top-k → top-p → sample
        """
        logits = logits.clone().float()

        if not do_sample:
            return cls.greedy(logits)

        # 1. Repetition penalty
        if penalty != 1.0 and generated_ids:
            logits = cls.repetition_penalty(logits, generated_ids, penalty)

        # 2. Temperature
        logits = cls.temperature(logits, temperature)

        # 3. Top-K filter
        logits = cls.top_k(logits, top_k)

        # 4. Top-P (nucleus) filter
        logits = cls.top_p(logits, top_p)

        # 5. Sample dari distribusi
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()