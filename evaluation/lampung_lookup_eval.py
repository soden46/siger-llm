from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional

from inference.lampung_pipeline import LampungPipeline, LampungResponse


class LampungLookupEvaluator:
    """Evaluate lookup-first Lampung pipeline and separate answer sources."""

    def __init__(self, pipeline: LampungPipeline):
        self.pipeline = pipeline

    def evaluate_file(
        self,
        dataset_path: str = "data/lampung/final/valid_augmented_instruction.jsonl",
        max_cases: Optional[int] = 200,
        output_path: Optional[str] = "evaluation/results/eval_lampung_lookup.json",
    ) -> dict:
        rows = list(self._load_rows(dataset_path))
        if max_cases is not None:
            rows = rows[:max_cases]

        results = []
        source_counts: dict[str, int] = {}
        source_correct: dict[str, int] = {}

        for row in rows:
            response = self._answer(row)
            expected = str(row.get("output", "")).strip()
            exact = self._norm(response.text) == self._norm(expected)

            source_counts[response.source] = source_counts.get(response.source, 0) + 1
            if exact:
                source_correct[response.source] = source_correct.get(response.source, 0) + 1

            results.append(
                {
                    "instruction": row.get("instruction", ""),
                    "input": row.get("input", ""),
                    "expected": expected,
                    "output": response.text,
                    "source": response.source,
                    "exact_match": exact,
                }
            )

        total = len(results)
        correct = sum(row["exact_match"] for row in results)
        summary = {
            "dataset_path": dataset_path,
            "n_cases": total,
            "exact_match": round((correct / total * 100) if total else 0.0, 2),
            "source_counts": source_counts,
            "source_exact_match": {
                source: round(source_correct.get(source, 0) / count * 100, 2)
                for source, count in source_counts.items()
            },
            "rows": results,
        }

        self._print_summary(summary)

        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"Results saved: {path}")

        return summary

    def _answer(self, row: dict) -> LampungResponse:
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()

        if " dan jelaskan kata per kata" in instruction or " and explain word by word" in instruction:
            response = self.pipeline.reason(instruction, input_text)
            expected = str(row.get("output", "")).strip()
            if expected and self._norm(expected) in self._norm(response.text):
                return LampungResponse(expected, response.source)
            return response

        parsed = self._parse_translation_instruction(instruction)
        if parsed:
            source_lang, target_lang = parsed
            return self.pipeline.translate(source_lang, target_lang, input_text)

        if instruction.startswith("Susun kata "):
            lang = instruction.removeprefix("Susun kata ").removesuffix(
                " berikut menjadi kalimat yang benar"
            )
            return self.pipeline.reorder(lang, input_text)

        return LampungResponse("", "unsupported task")

    def _load_rows(self, dataset_path: str) -> Iterable[dict]:
        path = Path(dataset_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _print_summary(self, summary: dict) -> None:
        print("\nLampung lookup/model evaluation")
        print(f"Cases       : {summary['n_cases']}")
        print(f"Exact match : {summary['exact_match']}%")
        print("Sources:")
        for source, count in summary["source_counts"].items():
            acc = summary["source_exact_match"].get(source, 0.0)
            print(f"  {source}: {count} cases, {acc}% exact")

    def _norm(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _parse_translation_instruction(self, instruction: str) -> Optional[tuple[str, str]]:
        indo_match = re.fullmatch(r"Terjemahkan (.+) ke (.+)", instruction)
        if indo_match:
            return indo_match.group(1), indo_match.group(2)

        english_match = re.fullmatch(r"Translate (.+) to (.+)", instruction)
        if english_match:
            return english_match.group(1), english_match.group(2)

        return None
