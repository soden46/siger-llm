# evaluation/report.py
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional


class EvalReport:
    """
    Generate laporan evaluasi dalam format Markdown dan JSON.
    Bisa dipakai untuk compare beberapa model/checkpoint.
    """

    def __init__(self, output_dir: str = "./evaluation/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_markdown(self, results: dict, tag: str = "eval") -> str:
        """
        Buat laporan Markdown dari hasil evaluasi.
        Return path file yang dibuat.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        lines.append(f"# 📊 Evaluation Report — `{tag}`")
        lines.append(f"\n**Generated:** {timestamp}\n")
        lines.append("---\n")

        # ── Perplexity ────────────────────────────────────
        if "perplexity" in results:
            r = results["perplexity"]
            lines.append("## Perplexity")
            lines.append(f"| Metric | Value |")
            lines.append(f"|--------|-------|")
            lines.append(f"| PPL    | **{r.get('ppl', 'N/A')}** |")
            lines.append(f"| NLL    | {r.get('nll', 'N/A')} |")
            lines.append(f"| Tokens | {r.get('tokens', 'N/A'):,} |")
            lines.append(f"| Grade  | {r.get('grade', 'N/A')} |")
            lines.append("")

        # ── MMLU ──────────────────────────────────────────
        if "mmlu" in results:
            r = results["mmlu"]
            lines.append("## MMLU Benchmark")
            lines.append(f"| Metric   | Value |")
            lines.append(f"|----------|-------|")
            lines.append(f"| Accuracy | **{r.get('accuracy', 'N/A')}%** |")
            lines.append(f"| Correct  | {r.get('correct', 'N/A')}/{r.get('total', 'N/A')} |")
            lines.append(f"| Grade    | {r.get('grade', 'N/A')} |")

            # Top subjects
            if "by_subject" in r and r["by_subject"]:
                lines.append("\n**Top 10 Subjects:**\n")
                lines.append("| Subject | Accuracy |")
                lines.append("|---------|----------|")
                for subj, acc in list(r["by_subject"].items())[:10]:
                    lines.append(f"| {subj} | {acc*100:.1f}% |")
            lines.append("")

        # ── ARC ───────────────────────────────────────────
        if "arc" in results:
            r = results["arc"]
            lines.append("## ARC-Challenge")
            lines.append(f"| Metric   | Value |")
            lines.append(f"|----------|-------|")
            lines.append(f"| Accuracy | **{r.get('accuracy', 'N/A')}%** |")
            lines.append(f"| Correct  | {r.get('correct', 'N/A')}/{r.get('total', 'N/A')} |")
            lines.append(f"| Grade    | {r.get('grade', 'N/A')} |")
            lines.append("")

        # ── Indo Benchmarks ───────────────────────────────
        if "indo" in results:
            r = results["indo"]
            lines.append("## 🇮🇩 Indo Benchmarks")
            lines.append(f"| Benchmark     | Score |")
            lines.append(f"|---------------|-------|")
            if r.get("sentiment"):
                lines.append(f"| Sentiment     | {r['sentiment'].get('accuracy', 'N/A')}% |")
            if r.get("nli"):
                lines.append(f"| NLI           | {r['nli'].get('accuracy', 'N/A')}% |")
            if r.get("qa"):
                lines.append(f"| QA (F1)       | {r['qa'].get('f1', 'N/A')}% |")
            lines.append("")

        # ── Generation ────────────────────────────────────
        if "generation" in results:
            r = results["generation"]
            lines.append("## Generation Quality")
            lines.append(f"| Metric     | Value |")
            lines.append(f"|------------|-------|")
            if "bleu" in r:
                lines.append(f"| BLEU       | {r['bleu'].get('bleu', 'N/A')} |")
                prec = r['bleu'].get('precisions', [])
                if prec:
                    lines.append(f"| BLEU-1     | {prec[0] if len(prec)>0 else 'N/A'} |")
                    lines.append(f"| BLEU-2     | {prec[1] if len(prec)>1 else 'N/A'} |")
                    lines.append(f"| BLEU-4     | {prec[3] if len(prec)>3 else 'N/A'} |")
            if "rouge" in r:
                lines.append(f"| ROUGE-1    | {r['rouge'].get('rouge1', 'N/A')} |")
                lines.append(f"| ROUGE-2    | {r['rouge'].get('rouge2', 'N/A')} |")
                lines.append(f"| ROUGE-L    | {r['rouge'].get('rougeL', 'N/A')} |")
            if "diversity" in r:
                lines.append(f"| Distinct-1 | {r['diversity'].get('distinct1', 'N/A')}% |")
                lines.append(f"| Distinct-2 | {r['diversity'].get('distinct2', 'N/A')}% |")
                lines.append(f"| Grade      | {r['diversity'].get('grade', 'N/A')} |")
            lines.append("")

        # ── Meta ──────────────────────────────────────────
        if "meta" in results:
            m = results["meta"]
            lines.append("---")
            lines.append(f"*Eval time: {m.get('elapsed_sec', 'N/A')}s | Tag: `{m.get('tag', tag)}`*")

        content = "\n".join(lines)

        # Save markdown
        md_path = self.output_dir / f"eval_{tag}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 Report saved: {md_path}")
        return str(md_path)

    def compare(self, tags: list[str]) -> str:
        """
        Buat tabel perbandingan dari beberapa hasil eval.
        tags: list nama tag yang JSON-nya sudah ada di output_dir.
        """
        all_results = {}
        for tag in tags:
            path = self.output_dir / f"eval_{tag}.json"
            if not path.exists():
                print(f"⚠️  Not found: {path}")
                continue
            with open(path) as f:
                all_results[tag] = json.load(f)

        if not all_results:
            print("❌ No results to compare.")
            return ""

        lines = ["# 📊 Model Comparison Report\n"]
        lines.append(f"Models compared: {', '.join(f'`{t}`' for t in all_results.keys())}\n")
        lines.append("---\n")

        # Build comparison table
        metrics = [
            ("Perplexity", lambda r: r.get("perplexity", {}).get("ppl", "N/A")),
            ("MMLU (%)",   lambda r: r.get("mmlu", {}).get("accuracy", "N/A")),
            ("ARC (%)",    lambda r: r.get("arc", {}).get("accuracy", "N/A")),
            ("Indo Sent (%)", lambda r: r.get("indo", {}).get("sentiment", {}).get("accuracy", "N/A")),
            ("IndoNLI (%)", lambda r: r.get("indo", {}).get("nli", {}).get("accuracy", "N/A")),
            ("BLEU",       lambda r: r.get("generation", {}).get("bleu", {}).get("bleu", "N/A")),
            ("ROUGE-L",    lambda r: r.get("generation", {}).get("rouge", {}).get("rougeL", "N/A")),
            ("Distinct-2 (%)", lambda r: r.get("generation", {}).get("diversity", {}).get("distinct2", "N/A")),
        ]

        header = "| Metric | " + " | ".join(all_results.keys()) + " |"
        sep    = "|--------|" + "--------|" * len(all_results)
        lines.append(header)
        lines.append(sep)

        for metric_name, getter in metrics:
            row = f"| {metric_name} |"
            for tag, result in all_results.items():
                val = getter(result)
                row += f" {val} |"
            lines.append(row)

        content = "\n".join(lines)

        # Save
        compare_path = self.output_dir / f"comparison_{'_vs_'.join(tags)}.md"
        with open(compare_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"📄 Comparison saved: {compare_path}")
        print(content)
        return str(compare_path)

    def load(self, tag: str) -> dict:
        """Load hasil eval dari JSON."""
        path = self.output_dir / f"eval_{tag}.json"
        if not path.exists():
            raise FileNotFoundError(f"Eval result not found: {path}")
        with open(path) as f:
            return json.load(f)

    def list_results(self) -> list[str]:
        """List semua eval results yang tersedia."""
        jsons = sorted(self.output_dir.glob("eval_*.json"))
        tags  = [p.stem.replace("eval_", "") for p in jsons]
        print(f"📂 Available results ({len(tags)}):")
        for tag in tags:
            print(f"  - {tag}")
        return tags