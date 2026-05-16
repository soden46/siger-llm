# evaluation/perplexity.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import math


class PerplexityEvaluator:
    """
    Perplexity = exp(average negative log likelihood)
    
    Intuisi:
    PPL = 1     → model 100% yakin (impossible)
    PPL = 10    → model punya 10 pilihan yang equally likely → bagus
    PPL = 100   → model bingung → jelek
    PPL = 1000+ → model random banget → ga belajar apa-apa

    Referensi:
    GPT-2 small  : PPL ~29 (WikiText-103)
    GPT-3        : PPL ~20
    Target lo    : PPL < 50 (acceptable buat model kecil)
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def compute(
        self,
        texts: List[str],
        max_seq_len: int  = 512,
        stride: int       = 256,   # sliding window — lebih akurat
        batch_size: int   = 4,
    ) -> dict:
        """
        Hitung PPL dengan sliding window.
        Sliding window penting buat sequence panjang:
        daripada truncate, lo overlap tiap window
        dan hanya hitung loss di bagian non-overlap.
        """
        total_nll  = 0.0   # negative log likelihood
        total_toks = 0

        for text in texts:
            input_ids = self.tokenizer.encode(text, add_bos=True)
            seq_len   = len(input_ids)

            if seq_len < 2:
                continue

            ids_tensor = torch.tensor(input_ids, dtype=torch.long)

            # Sliding window
            prev_end = 0
            for begin in range(0, seq_len - 1, stride):
                end        = min(begin + max_seq_len, seq_len)
                window     = ids_tensor[begin:end].unsqueeze(0).to(self.device)
                target_len = end - max(begin, prev_end)   # hanya hitung bagian baru

                logits, _ = self.model(window)

                # Shift: predict token ke-i dari token ke-(i-1)
                shift_logits = logits[0, :-1, :]   # (L-1, vocab)
                shift_labels = window[0, 1:]        # (L-1,)

                # Hanya hitung loss di bagian baru (bukan overlap)
                target_start = max(0, prev_end - begin - 1)
                if target_start >= shift_logits.size(0):
                    continue

                sl = shift_logits[target_start:]
                tl = shift_labels[target_start:]

                nll = F.cross_entropy(sl, tl, reduction="sum")
                total_nll  += nll.item()
                total_toks += tl.size(0)

                prev_end = end
                if end == seq_len:
                    break

        if total_toks == 0:
            return {"ppl": float("inf"), "nll": float("inf"), "tokens": 0}

        avg_nll = total_nll / total_toks
        ppl     = math.exp(avg_nll)

        result = {
            "ppl":    round(ppl, 2),
            "nll":    round(avg_nll, 4),
            "tokens": total_toks,
            "grade":  self._grade(ppl),
        }

        print(f"📊 Perplexity: {ppl:.2f} | NLL: {avg_nll:.4f} "
              f"| Tokens: {total_toks:,} | Grade: {result['grade']}")
        return result

    @staticmethod
    def _grade(ppl: float) -> str:
        if ppl < 15:   return "🟢 Excellent"
        if ppl < 30:   return "🟢 Good"
        if ppl < 50:   return "🟡 Acceptable"
        if ppl < 100:  return "🟠 Poor"
        return             "🔴 Very Poor"

    def compare(
        self,
        texts: List[str],
        model_names: List[str],
        models: list,
    ) -> dict:
        """Bandingkan PPL beberapa model sekaligus."""
        results = {}
        for name, model in zip(model_names, models):
            self.model = model.eval().to(self.device)
            print(f"\nEvaluating: {name}")
            results[name] = self.compute(texts)
        return results