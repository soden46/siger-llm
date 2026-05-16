# evaluation/generation.py
import torch
from typing import List, Dict
from collections import Counter
import math
import re


class GenerationEvaluator:
    """
    Evaluasi kualitas teks yang di-generate.
    
    BLEU   → overlap n-gram vs referensi (precision-focused)
    ROUGE  → overlap n-gram vs referensi (recall-focused)
    Diversity → seberapa variatif output model
    """

    # ── BLEU ──────────────────────────────────────────────

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    @classmethod
    def bleu(
        cls,
        hypotheses: List[str],
        references: List[str],
        max_n: int = 4,
    ) -> Dict:
        """
        Corpus-level BLEU score.
        Makin tinggi makin mirip referensi (max 100).
        """
        clipped_counts = Counter()
        total_counts   = Counter()
        hyp_len = ref_len = 0

        for hyp, ref in zip(hypotheses, references):
            hyp_tok = hyp.lower().split()
            ref_tok = ref.lower().split()
            hyp_len += len(hyp_tok)
            ref_len += len(ref_tok)

            for n in range(1, max_n + 1):
                hyp_ngrams = cls._ngrams(hyp_tok, n)
                ref_ngrams = cls._ngrams(ref_tok, n)
                clipped    = {k: min(v, ref_ngrams[k]) for k, v in hyp_ngrams.items()}
                clipped_counts[n] += sum(clipped.values())
                total_counts[n]   += sum(hyp_ngrams.values())

        # Precision per n-gram
        precisions = []
        for n in range(1, max_n + 1):
            if total_counts[n] == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped_counts[n] / total_counts[n])

        # Brevity penalty
        bp   = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
        logsum = sum(math.log(p) if p > 0 else float("-inf") for p in precisions)
        bleu   = bp * math.exp(logsum / max_n)

        return {
            "bleu":       round(bleu * 100, 2),
            "precisions": [round(p * 100, 2) for p in precisions],
            "bp":         round(bp, 4),
            "grade":      "🟢 Good" if bleu > 0.3 else
                          "🟡 Fair" if bleu > 0.15 else "🔴 Poor",
        }

    # ── ROUGE ─────────────────────────────────────────────

    @classmethod
    def rouge(
        cls,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """ROUGE-1, ROUGE-2, ROUGE-L scores."""

        def rouge_n(hyp, ref, n):
            hyp_ng = cls._ngrams(hyp.lower().split(), n)
            ref_ng = cls._ngrams(ref.lower().split(), n)
            match  = sum((hyp_ng & ref_ng).values())
            prec   = match / max(sum(hyp_ng.values()), 1)
            rec    = match / max(sum(ref_ng.values()), 1)
            f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            return f1

        def rouge_l(hyp, ref):
            hyp_tok = hyp.lower().split()
            ref_tok = ref.lower().split()
            # LCS dynamic programming
            m, n = len(hyp_tok), len(ref_tok)
            dp   = [[0] * (n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if hyp_tok[i-1] == ref_tok[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs  = dp[m][n]
            prec = lcs / max(m, 1)
            rec  = lcs / max(n, 1)
            return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

        r1 = sum(rouge_n(h, r, 1) for h, r in zip(hypotheses, references))
        r2 = sum(rouge_n(h, r, 2) for h, r in zip(hypotheses, references))
        rl = sum(rouge_l(h, r) for h, r in zip(hypotheses, references))
        n  = len(hypotheses)

        return {
            "rouge1": round(r1 / n * 100, 2),
            "rouge2": round(r2 / n * 100, 2),
            "rougeL": round(rl / n * 100, 2),
        }

    # ── Diversity ─────────────────────────────────────────

    @staticmethod
    def diversity(texts: List[str]) -> Dict:
        """
        Distinct-1 dan Distinct-2:
        Seberapa beragam unigram/bigram di semua output.
        Makin tinggi = makin variatif, ga repetitif.
        """
        all_unigrams, all_bigrams = [], []
        for text in texts:
            tokens = text.lower().split()
            all_unigrams.extend(tokens)
            all_bigrams.extend(zip(tokens, tokens[1:]))

        d1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
        d2 = len(set(all_bigrams))  / max(len(all_bigrams),  1)

        return {
            "distinct1": round(d1 * 100, 2),
            "distinct2": round(d2 * 100, 2),
            "grade":     "🟢 Diverse"    if d2 > 0.7 else
                         "🟡 Moderate"   if d2 > 0.4 else
                         "🔴 Repetitive",
        }