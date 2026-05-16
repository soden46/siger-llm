# evaluation/runner.py
import json
import time
from pathlib import Path
from typing import List, Optional

from .perplexity import PerplexityEvaluator
from .benchmarks import MultiplChoiceBenchmark
from .generation import GenerationEvaluator
from .indo_eval  import IndoEvaluator
from .report     import EvalReport


class EvaluationRunner:
    """
    Jalanin semua evaluasi dalam 1 call.
    Kayak `php artisan test` — run semua test suite sekaligus.
    """

    # Teks Indo + EN buat PPL evaluation
    PPL_TEXTS = [
        "Pemerintah Indonesia mengumumkan kebijakan baru terkait pendidikan nasional.",
        "Teknologi kecerdasan buatan berkembang pesat dalam beberapa tahun terakhir.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data.",
        "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa.",
    ]

    # Instruction following test cases
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
        device: str = "cpu",
        output_dir: str = "./evaluation/results",
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.device    = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        run_ppl:   bool = True,
        run_mmlu:  bool = True,
        run_arc:   bool = True,
        run_indo:  bool = True,
        run_gen:   bool = True,
        n_samples: int  = 200,
        tag:       str  = "eval",
    ) -> dict:

        results  = {}
        start    = time.time()

        print(f"\n{'='*55}")
        print(f"  🔬 LLM Evaluation Suite")
        print(f"{'='*55}\n")

        # ── 1. Perplexity ──────────────────────────────────
        if run_ppl:
            print("📐 [1/5] Perplexity...")
            ppl_eval     = PerplexityEvaluator(self.model, self.tokenizer, self.device)
            results["perplexity"] = ppl_eval.compute(self.PPL_TEXTS)

        # ── 2. MMLU ────────────────────────────────────────
        if run_mmlu:
            print("\n📚 [2/5] MMLU Benchmark...")
            mc_eval = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["mmlu"] = mc_eval.evaluate_mmlu(n_samples=n_samples)

        # ── 3. ARC ─────────────────────────────────────────
        if run_arc:
            print("\n🔭 [3/5] ARC-Challenge...")
            mc_eval       = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["arc"] = mc_eval.evaluate_arc(n_samples=n_samples)

        # ── 4. Indo Benchmarks ─────────────────────────────
        if run_indo:
            print("\n🇮🇩 [4/5] Indo Benchmarks...")
            indo_eval = IndoEvaluator(self.model, self.tokenizer, self.device)
            results["indo"] = {
                "sentiment": indo_eval.evaluate_sentiment(n_samples),
                "nli":       indo_eval.evaluate_nli(n_samples),
                "qa":        indo_eval.evaluate_qa(n_samples // 2),
            }

        # ── 5. Generation Quality ──────────────────────────
        if run_gen and self.generator:
            print("\n✍️  [5/5] Generation Quality...")
            gen_eval = GenerationEvaluator()
            hypotheses = []
            references = []

            for test in self.INSTRUCTION_TESTS:
                out = self.generator.generate(
                    test["prompt"],
                    max_new_tokens=50,
                    temperature=0.1,   # low temp buat eval
                )
                hypotheses.append(out)
                references.append(test["reference"])
                print(f"  Q: {test['prompt'][-40:]}")
                print(f"  A: {out[:60]}")

            results["generation"] = {
                "bleu":      gen_eval.bleu(hypotheses, references),
                "rouge":     gen_eval.rouge(hypotheses, references),
                "diversity": gen_eval.diversity(hypotheses),
            }

        # ── Summary ────────────────────────────────────────
        elapsed = time.time() - start
        results["meta"] = {"elapsed_sec": round(elapsed, 1), "tag": tag}

        self._print_summary(results)
        self._save(results, tag)
        return results

    def _print_summary(self, results: dict):
        print(f"\n{'='*55}")
        print(f"  📊 EVALUATION SUMMARY")
        print(f"{'='*55}")

        if "perplexity" in results:
            r = results["perplexity"]
            print(f"  Perplexity   : {r['ppl']:<8} {r.get('grade','')}")

        if "mmlu" in results:
            r = results["mmlu"]
            print(f"  MMLU         : {r['accuracy']}%    {r.get('grade','')}")

        if "arc" in results:
            r = results["arc"]
            print(f"  ARC-Challenge: {r['accuracy']}%    {r.get('grade','')}")

        if "indo" in results:
            r = results["indo"]
            if r.get("sentiment"):
                print(f"  Indo Sentiment: {r['sentiment'].get('accuracy','N/A')}%")
            if r.get("nli"):
                print(f"  Indo NLI      : {r['nli'].get('accuracy','N/A')}%")
            if r.get("qa"):
                print(f"  Indo QA F1    : {r['qa'].get('f1','N/A')}%")

        if "generation" in results:
            r = results["generation"]
            print(f"  BLEU         : {r['bleu']['bleu']}")
            print(f"  ROUGE-L      : {r['rouge']['rougeL']}")
            print(f"  Diversity-2  : {r['diversity']['distinct2']}%  "
                  f"{r['diversity']['grade']}")

        elapsed = results.get("meta", {}).get("elapsed_sec", 0)
        print(f"\n  ⏱️  Total time: {elapsed:.1f}s")
        print(f"{'='*55}\n")

    def _save(self, results: dict, tag: str):
        path = self.output_dir / f"eval_{tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved: {path}")