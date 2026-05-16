# evaluation/runner.py
import json
import time
from pathlib import Path
from typing import List, Optional

from .perplexity   import PerplexityEvaluator
from .benchmarks   import MultiplChoiceBenchmark
from .generation   import GenerationEvaluator
from .indo_eval    import IndoEvaluator
from .lampung_eval import LampungEvaluator
from .report       import EvalReport


class EvaluationRunner:
    """
    Jalanin semua evaluasi dalam 1 call.
    Kayak `php artisan test` — run semua test suite sekaligus.

    Include evaluasi khusus Bahasa Lampung untuk SigerLM.
    """

    PPL_TEXTS = [
        # Indonesia
        "Pemerintah Indonesia mengumumkan kebijakan baru terkait pendidikan nasional.",
        "Teknologi kecerdasan buatan berkembang pesat dalam beberapa tahun terakhir.",
        "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa.",
        # English
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data.",
        # Lampung O
        "Nyak haga mengan di pasar. Api kabar niku sekeluarga?",
    ]

    INSTRUCTION_TESTS = [
        {
            "prompt":    "<|user|>Apa ibu kota Indonesia?<|end_turn|>\n<|assistant|>",
            "reference": "Jakarta adalah ibu kota Indonesia.",
        },
        {
            "prompt":    "<|user|>Jelaskan apa itu machine learning dalam 1 kalimat.<|end_turn|>\n<|assistant|>",
            "reference": "Machine learning adalah cabang AI yang memungkinkan komputer belajar dari data.",
        },
        {
            "prompt":    "<|user|>Sebutkan 3 bahasa pemrograman populer.<|end_turn|>\n<|assistant|>",
            "reference": "Python, JavaScript, dan Java adalah bahasa pemrograman populer.",
        },
    ]

    def __init__(
        self,
        model,
        tokenizer,
        generator=None,
        device: str    = "cpu",
        output_dir: str = "./evaluation/results",
    ):
        self.model      = model
        self.tokenizer  = tokenizer
        self.generator  = generator
        self.device     = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report     = EvalReport(output_dir)

    def run(
        self,
        run_ppl:     bool = True,
        run_mmlu:    bool = True,
        run_arc:     bool = True,
        run_indo:    bool = True,
        run_gen:     bool = True,
        run_lampung: bool = True,   # evaluasi Bahasa Lampung
        n_samples:   int  = 200,
        tag:         str  = "eval",
    ) -> dict:

        results = {}
        start   = time.time()

        print(f"\n{'='*55}")
        print(f"  🔬 SigerLM Evaluation Suite")
        print(f"{'='*55}\n")

        # ── 1. Perplexity ──────────────────────────────────
        if run_ppl:
            print("📐 [1/6] Perplexity...")
            ppl_eval = PerplexityEvaluator(self.model, self.tokenizer, self.device)
            results["perplexity"] = ppl_eval.compute(self.PPL_TEXTS)

        # ── 2. MMLU ────────────────────────────────────────
        if run_mmlu:
            print("\n📚 [2/6] MMLU Benchmark...")
            mc_eval = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["mmlu"] = mc_eval.evaluate_mmlu(n_samples=n_samples)

        # ── 3. ARC ─────────────────────────────────────────
        if run_arc:
            print("\n🔭 [3/6] ARC-Challenge...")
            mc_eval = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["arc"] = mc_eval.evaluate_arc(n_samples=n_samples)

        # ── 4. Indo Benchmarks ─────────────────────────────
        if run_indo:
            print("\n🇮🇩 [4/6] Indo Benchmarks...")
            indo_eval = IndoEvaluator(self.model, self.tokenizer, self.device)
            results["indo"] = {
                "sentiment": indo_eval.evaluate_sentiment(n_samples),
                "nli":       indo_eval.evaluate_nli(n_samples),
                "qa":        indo_eval.evaluate_qa(n_samples // 2),
            }

        # ── 5. Generation Quality ──────────────────────────
        if run_gen and self.generator:
            print("\n✍️  [5/6] Generation Quality...")
            gen_eval   = GenerationEvaluator()
            hypotheses = []
            references = []

            for test in self.INSTRUCTION_TESTS:
                out = self.generator.generate(
                    test["prompt"],
                    max_new_tokens=50,
                    temperature=0.1,
                )
                hypotheses.append(out)
                references.append(test["reference"])
                print(f"  Q: ...{test['prompt'][-40:]}")
                print(f"  A: {out[:60]}")

            results["generation"] = {
                "bleu":      gen_eval.bleu(hypotheses, references),
                "rouge":     gen_eval.rouge(hypotheses, references),
                "diversity": gen_eval.diversity(hypotheses),
            }

        # ── 6. Lampung Eval ────────────────────────────────
        if run_lampung and self.generator:
            print("\n🌴 [6/6] Lampung Translation Eval...")
            lampung_eval = LampungEvaluator(
                self.generator, self.tokenizer, self.device
            )
            results["lampung"] = {
                "lo_to_id": lampung_eval.evaluate_lo_to_id(temperature=0.1),
                "id_to_lo": lampung_eval.evaluate_id_to_lo(temperature=0.1),
                "vocab":    lampung_eval.evaluate_vocabulary_coverage(),
            }

        # ── Summary & Save ─────────────────────────────────
        elapsed = time.time() - start
        results["meta"] = {"elapsed_sec": round(elapsed, 1), "tag": tag}

        self._print_summary(results)
        self._save(results, tag)

        # Generate markdown report
        self.report.generate_markdown(results, tag)

        return results

    def _print_summary(self, results: dict):
        print(f"\n{'='*55}")
        print(f"  📊 EVALUATION SUMMARY")
        print(f"{'='*55}")

        if "perplexity" in results:
            r = results["perplexity"]
            print(f"  Perplexity    : {r['ppl']:<8} {r.get('grade', '')}")

        if "mmlu" in results:
            r = results["mmlu"]
            print(f"  MMLU          : {r['accuracy']}%    {r.get('grade', '')}")

        if "arc" in results:
            r = results["arc"]
            print(f"  ARC-Challenge : {r['accuracy']}%    {r.get('grade', '')}")

        if "indo" in results:
            r = results["indo"]
            if r.get("sentiment"):
                print(f"  Indo Sentiment: {r['sentiment'].get('accuracy', 'N/A')}%")
            if r.get("nli"):
                print(f"  Indo NLI      : {r['nli'].get('accuracy', 'N/A')}%")
            if r.get("qa"):
                print(f"  Indo QA F1    : {r['qa'].get('f1', 'N/A')}%")

        if "generation" in results:
            r = results["generation"]
            print(f"  BLEU          : {r['bleu']['bleu']}")
            print(f"  ROUGE-L       : {r['rouge']['rougeL']}")
            print(f"  Diversity-2   : {r['diversity']['distinct2']}%  {r['diversity']['grade']}")

        if "lampung" in results:
            r = results["lampung"]
            if r.get("lo_to_id"):
                lo = r["lo_to_id"]
                print(f"  LO→ID Exact   : {lo.get('exact_match', 'N/A')}%  BLEU: {lo.get('bleu', 'N/A')}")
            if r.get("id_to_lo"):
                il = r["id_to_lo"]
                print(f"  ID→LO Exact   : {il.get('exact_match', 'N/A')}%  BLEU: {il.get('bleu', 'N/A')}")
            if r.get("vocab"):
                print(f"  Vocab 1-token : {r['vocab'].get('coverage_pct', 'N/A')}%")

        elapsed = results.get("meta", {}).get("elapsed_sec", 0)
        print(f"\n  ⏱️  Total time : {elapsed:.1f}s")
        print(f"{'='*55}\n")

    def _save(self, results: dict, tag: str):
        path = self.output_dir / f"eval_{tag}.json"
        # Hapus per_pair dari JSON supaya tidak terlalu besar
        save_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                save_results[k] = {
                    kk: vv for kk, vv in v.items() if kk != "per_pair"
                }
            else:
                save_results[k] = v

        with open(path, "w", encoding="utf-8") as f:
            json.dump(save_results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved: {path}")