import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LookupResult:
    output: str
    source: str


class InstructionLookup:
    """Exact JSONL lookup for small Lampung instruction datasets."""

    SPACE_RE = re.compile(r"\s+")

    def __init__(self, paths: list[str]):
        self.outputs: dict[tuple[str, str], str] = {}
        self.reasoning: dict[tuple[str, str], tuple[str, str]] = {}
        self.bag_of_words: dict[tuple[str, str], str] = {}

        for path in paths:
            self._load(path)

    @classmethod
    def from_final_dir(
        cls,
        final_dir: str = "data/lampung/final",
        splits: tuple[str, ...] = ("train",),
    ) -> "InstructionLookup":
        base = Path(final_dir)
        paths = [
            str(base / f"{split}_augmented_instruction.jsonl")
            for split in splits
        ]
        return cls(paths)

    def _load(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists():
            return

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                instruction = str(row.get("instruction", "")).strip()
                input_text = str(row.get("input", "")).strip()
                output = str(row.get("output", "")).strip()
                reasoning = str(row.get("reasoning", "")).strip()

                if not instruction or not input_text or not output:
                    continue

                key = (self._norm(instruction), self._norm(input_text))
                self.outputs.setdefault(key, output)
                self.bag_of_words.setdefault(
                    (self._norm(instruction), self._word_key(input_text)),
                    output,
                )
                if reasoning:
                    self.reasoning.setdefault(key, (reasoning, output))

    def get(self, instruction: str, input_text: str) -> Optional[str]:
        return self.outputs.get((self._norm(instruction), self._norm(input_text)))

    def get_reasoning(self, instruction: str, input_text: str) -> Optional[str]:
        item = self.reasoning.get((self._norm(instruction), self._norm(input_text)))
        if not item:
            return None

        reasoning, output = item
        return f"Penjelasan:\n{reasoning}\n\nJawaban:\n{output}"

    def reorder(self, lang: str, words: str) -> Optional[str]:
        result = self.reorder_with_source(lang, words)
        return result.output if result else None

    def reorder_with_source(self, lang: str, words: str) -> Optional[LookupResult]:
        instruction = f"Susun kata {lang} berikut menjadi kalimat yang benar"
        exact = self.get(instruction, words)
        if exact:
            return LookupResult(exact, "exact instruction lookup")

        near = self.bag_of_words.get((self._norm(instruction), self._word_key(words)))
        if near:
            return LookupResult(near, "bag-of-words instruction lookup")

        return None

    def translate(self, source_lang: str, target_lang: str, text: str) -> Optional[str]:
        result = self.translate_with_source(source_lang, target_lang, text)
        return result.output if result else None

    def translate_with_source(
        self,
        source_lang: str,
        target_lang: str,
        text: str,
    ) -> Optional[LookupResult]:
        instructions = [
            f"Terjemahkan {source_lang} ke {target_lang}",
            f"Translate {source_lang} to {target_lang}",
        ]

        for instruction in instructions:
            output = self.get(instruction, text)
            if output:
                return LookupResult(output, "exact instruction lookup")

        return None

    def translate_with_reasoning(
        self,
        source_lang: str,
        target_lang: str,
        text: str,
    ) -> Optional[str]:
        instruction = f"Terjemahkan {source_lang} ke {target_lang} dan jelaskan kata per kata"
        return self.get_reasoning(instruction, text)

    def translate_reasoning_with_source(
        self,
        source_lang: str,
        target_lang: str,
        text: str,
    ) -> Optional[LookupResult]:
        output = self.translate_with_reasoning(source_lang, target_lang, text)
        if output:
            return LookupResult(output, "exact instruction lookup")

        return None

    def _norm(self, text: str) -> str:
        return self.SPACE_RE.sub(" ", text.strip()).lower()

    def _word_key(self, text: str) -> str:
        return " ".join(sorted(self._norm(text).split()))
