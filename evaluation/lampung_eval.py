# evaluation/lampung_eval.py
"""
Evaluasi khusus untuk model translasi Bahasa Lampung SigerLM.

Mencakup:
- Translation accuracy (Lampung O → Indonesia)
- Translation accuracy (Indonesia → Lampung O)
- BLEU score untuk translasi
- Coverage kosakata Lampung
"""
import json
import torch
from pathlib import Path
from typing import List, Dict, Optional
from .generation import GenerationEvaluator


# ── Test set translasi Lampung O ─────────────────────────────
LAMPUNG_O_TEST_PAIRS = [
    # (lampung_o, indonesian)
    ("nyak haga mengan", "saya mau makan"),
    ("api kabar niku", "apa kabar kamu"),
    ("jak lapah temon", "kita pergi juga"),
    ("niku haga lapah pok", "kamu mau pergi ke mana"),
    ("nyak kona sai", "saya yang punya"),
    ("lapah majeu", "pergi ke sana"),
    ("sanak-sanak lapah sekulah", "anak-anak pergi sekolah"),
    ("mengan munih minom", "makan juga minum"),
    ("temon niku", "kamu juga"),
    ("mit Jakarta", "ke Jakarta"),
]

# ── Prompt templates ─────────────────────────────────────────
TRANSLATE_LO_TO_ID = "Terjemahkan Lampung O ke Bahasa Indonesia: {text}"
TRANSLATE_ID_TO_LO = "Terjemahkan Bahasa Indonesia ke Lampung O: {text}"
TRANSLATE_LO_TO_EN = "Translate Lampung O to English: {text}"


class LampungEvaluator:
    """
    Evaluasi kemampuan translasi Bahasa Lampung model SigerLM.

    Gunakan setelah LoRA fine-tuning dengan dataset Lampung.
    Model yang belum di-fine-tune akan dapat skor mendekati 0.
    """

    def __init__(self, generator, tokenizer, device: str = "cpu"):
        self.generator = generator
        self.tokenizer = tokenizer
        self.device    = device
        self.gen_eval  = GenerationEvaluator()

    def evaluate_lo_to_id(
        self,
        test_pairs: Optional[List[tuple]] = None,
        max_new_tokens: int = 30,
        temperature: float  = 0.1,   # low temp untuk eval translasi
    ) -> Dict:
        """
        Evaluasi translasi Lampung O → Bahasa Indonesia.

        Args:
            test_pairs     : list of (lampung_o, indonesian) tuples
            max_new_tokens : max token yang di-generate
            temperature    : sampling temperature (rendah = deterministik)

        Returns:
            dict berisi BLEU, ROUGE, accuracy, per-pair results
        """
        pairs = test_pairs or LAMPUNG_O_TEST_PAIRS
        hypotheses = []
        references = []
        per_pair   = []

        print(f"\n🌴 Evaluating Lampung O → Indonesia ({len(pairs)} pairs)...")

        for lampung, indo in pairs:
            prompt = TRANSLATE_LO_TO_ID.format(text=lampung)

            # Format sebagai instruction
            full_prompt = (
                f"<|system|>Kamu adalah asisten penerjemah Bahasa Lampung.<|end_turn|>\n"
                f"<|user|>{prompt}<|end_turn|>\n"
                f"<|assistant|>"
            )

            output = self.generator.generate(
                full_prompt,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = 10,
                top_p          = 0.9,
            ).strip()

            hypotheses.append(output)
            references.append(indo)

            # Exact match (case-insensitive)
            exact = output.lower().strip() == indo.lower().strip()

            per_pair.append({
                "lampung":    lampung,
                "reference":  indo,
                "hypothesis": output,
                "exact_match": exact,
            })

            status = "✅" if exact else "❌"
            print(f"  {status} '{lampung}' → '{output}' (ref: '{indo}')")

        # Hitung metrics
        bleu  = self.gen_eval.bleu(hypotheses, references)
        rouge = self.gen_eval.rouge(hypotheses, references)
        exact_acc = sum(p["exact_match"] for p in per_pair) / len(per_pair)

        result = {
            "direction":   "Lampung_O → Indonesia",
            "n_pairs":     len(pairs),
            "exact_match": round(exact_acc * 100, 2),
            "bleu":        bleu["bleu"],
            "rouge_l":     rouge["rougeL"],
            "per_pair":    per_pair,
            "grade":       self._grade_translation(bleu["bleu"]),
        }

        print(f"\n📊 Lampung O → Indonesia:")
        print(f"   Exact match : {result['exact_match']}%")
        print(f"   BLEU        : {result['bleu']}")
        print(f"   ROUGE-L     : {result['rouge_l']}")
        print(f"   Grade       : {result['grade']}")

        return result

    def evaluate_id_to_lo(
        self,
        test_pairs: Optional[List[tuple]] = None,
        max_new_tokens: int = 30,
        temperature: float  = 0.1,
    ) -> Dict:
        """
        Evaluasi translasi Bahasa Indonesia → Lampung O.
        Arah kebalikan — lebih sulit karena generatif.
        """
        pairs = test_pairs or LAMPUNG_O_TEST_PAIRS
        # Swap: referensi sekarang Lampung O, input Indonesia
        hypotheses = []
        references = []
        per_pair   = []

        print(f"\n🌴 Evaluating Indonesia → Lampung O ({len(pairs)} pairs)...")

        for lampung, indo in pairs:
            prompt = TRANSLATE_ID_TO_LO.format(text=indo)

            full_prompt = (
                f"<|system|>Kamu adalah asisten penerjemah Bahasa Lampung.<|end_turn|>\n"
                f"<|user|>{prompt}<|end_turn|>\n"
                f"<|assistant|>"
            )

            output = self.generator.generate(
                full_prompt,
                max_new_tokens = max_new_tokens,
                temperature    = temperature,
                top_k          = 10,
                top_p          = 0.9,
            ).strip()

            hypotheses.append(output)
            references.append(lampung)

            exact = output.lower().strip() == lampung.lower().strip()
            per_pair.append({
                "indonesian": indo,
                "reference":  lampung,
                "hypothesis": output,
                "exact_match": exact,
            })

            status = "✅" if exact else "❌"
            print(f"  {status} '{indo}' → '{output}' (ref: '{lampung}')")

        bleu  = self.gen_eval.bleu(hypotheses, references)
        rouge = self.gen_eval.rouge(hypotheses, references)
        exact_acc = sum(p["exact_match"] for p in per_pair) / len(per_pair)

        result = {
            "direction":   "Indonesia → Lampung_O",
            "n_pairs":     len(pairs),
            "exact_match": round(exact_acc * 100, 2),
            "bleu":        bleu["bleu"],
            "rouge_l":     rouge["rougeL"],
            "per_pair":    per_pair,
            "grade":       self._grade_translation(bleu["bleu"]),
        }

        print(f"\n📊 Indonesia → Lampung O:")
        print(f"   Exact match : {result['exact_match']}%")
        print(f"   BLEU        : {result['bleu']}")
        print(f"   ROUGE-L     : {result['rouge_l']}")
        print(f"   Grade       : {result['grade']}")

        return result

    def evaluate_vocabulary_coverage(
        self,
        lampung_words: Optional[List[str]] = None,
    ) -> Dict:
        """
        Hitung berapa persen kosakata Lampung yang dikenali tokenizer
        sebagai single token (bukan dipecah jadi banyak subword).

        Kosakata yang di-split banyak = kurang efisien untuk model.
        """
        default_words = [
            "nyak", "niku", "jak", "api", "haga", "lapah", "mengan",
            "minom", "temon", "munih", "kona", "sai", "majeu", "pok",
            "mit", "sanak", "sekulah", "pasar", "pekon", "gawoh",
        ]
        words = lampung_words or default_words

        single_token = 0
        multi_token  = 0
        details      = []

        for word in words:
            ids     = self.tokenizer.encode(word)
            n_toks  = len(ids)
            is_single = n_toks == 1

            if is_single:
                single_token += 1
            else:
                multi_token += 1

            details.append({
                "word":    word,
                "n_tokens": n_toks,
                "efficient": is_single,
            })

        coverage = single_token / len(words) * 100

        result = {
            "total_words":   len(words),
            "single_token":  single_token,
            "multi_token":   multi_token,
            "coverage_pct":  round(coverage, 1),
            "details":       details,
        }

        print(f"\n📚 Vocabulary Coverage (Lampung O):")
        print(f"   Single token : {single_token}/{len(words)} ({coverage:.1f}%)")
        print(f"   Multi token  : {multi_token}/{len(words)}")
        inefficient = [d for d in details if not d["efficient"]]
        if inefficient:
            print(f"\n   Kata yang di-split (perlu vocab extension):")
            for d in inefficient[:10]:
                print(f"   '{d['word']}' → {d['n_tokens']} tokens")

        return result

    def run_full_eval(
        self,
        output_path: Optional[str] = None,
    ) -> Dict:
        """
        Jalankan semua evaluasi Lampung sekaligus.

        Args:
            output_path : opsional, simpan hasil ke JSON

        Returns:
            dict lengkap hasil semua evaluasi
        """
        print(f"\n{'='*55}")
        print(f"  🌴 SigerLM Lampung Evaluation Suite")
        print(f"{'='*55}")

        results = {}
        results["lo_to_id"]    = self.evaluate_lo_to_id()
        results["id_to_lo"]    = self.evaluate_id_to_lo()
        results["vocab"]       = self.evaluate_vocabulary_coverage()

        # Summary
        print(f"\n{'='*55}")
        print(f"  📊 LAMPUNG EVAL SUMMARY")
        print(f"{'='*55}")
        print(f"  LO→ID Exact : {results['lo_to_id']['exact_match']}%   BLEU: {results['lo_to_id']['bleu']}")
        print(f"  ID→LO Exact : {results['id_to_lo']['exact_match']}%   BLEU: {results['id_to_lo']['bleu']}")
        print(f"  Vocab 1-tok : {results['vocab']['coverage_pct']}%")
        print(f"{'='*55}")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                # Hapus per_pair dari JSON kalau terlalu besar
                save_results = {
                    k: {kk: vv for kk, vv in v.items() if kk != "per_pair"}
                    for k, v in results.items()
                }
                json.dump(save_results, f, indent=2, ensure_ascii=False)
            print(f"💾 Lampung eval saved: {output_path}")

        return results

    @staticmethod
    def _grade_translation(bleu: float) -> str:
        if bleu >= 40:  return "🟢 Good"
        if bleu >= 20:  return "🟡 Fair"
        if bleu >= 5:   return "🟠 Poor (butuh lebih banyak data)"
        return                 "🔴 Very Poor (model belum belajar)"


def run_lampung_eval(generator, tokenizer, output_dir: str = "./evaluation/results"):
    """
    Shortcut function untuk menjalankan Lampung eval.

    Usage:
        from evaluation.lampung_eval import run_lampung_eval
        run_lampung_eval(generator, tokenizer)
    """
    evaluator = LampungEvaluator(generator, tokenizer)
    return evaluator.run_full_eval(
        output_path=f"{output_dir}/eval_lampung.json"
    )