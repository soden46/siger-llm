from __future__ import annotations

from dataclasses import dataclass

from inference.prompt_builder import build_reasoning_prompt, build_translation_prompt


@dataclass
class ReasoningCase:
    instruction: str
    input_text: str
    expected_answer: str
    expected_terms: tuple[str, ...] = ()


DEFAULT_REASONING_CASES = [
    ReasoningCase(
        instruction="Terjemahkan Lampung O ke Bahasa Indonesia dan jelaskan kata per kata",
        input_text="nyak haga mengan.",
        expected_answer="saya mau makan",
        expected_terms=("nyak", "haga", "mengan"),
    ),
]


class LampungReasoningEvaluator:
    def __init__(self, generator, lexicon=None):
        self.generator = generator
        self.lexicon = lexicon

    def evaluate_cases(self, cases=None) -> dict:
        cases = cases or DEFAULT_REASONING_CASES
        rows = []

        for case in cases:
            context = self.lexicon.build_context(case.input_text) if self.lexicon else ""
            prompt = build_reasoning_prompt(case.instruction, case.input_text, context)
            output = self.generator.generate(
                prompt,
                max_new_tokens=80,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
            ).strip()

            lower = output.lower()
            answer_match = case.expected_answer.lower() in lower
            term_hits = sum(term.lower() in lower for term in case.expected_terms)

            rows.append(
                {
                    "input": case.input_text,
                    "expected": case.expected_answer,
                    "output": output,
                    "answer_match": answer_match,
                    "term_hits": term_hits,
                    "term_total": len(case.expected_terms),
                }
            )

        exact = sum(row["answer_match"] for row in rows) / len(rows)
        return {"exact_like": exact, "rows": rows}


def build_direct_translation_prompt(text: str) -> str:
    return build_translation_prompt("Lampung O", "Bahasa Indonesia", text)

