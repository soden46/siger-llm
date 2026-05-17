from dataclasses import dataclass
from typing import Optional

from inference.generator import Generator
from inference.prompt_builder import (
    build_reasoning_prompt,
    build_translation_prompt,
    build_word_order_prompt,
)
from retrieval.compositional_translator import LampungCompositionalTranslator
from retrieval.instruction_lookup import InstructionLookup
from retrieval.lampung_lexicon import LampungLexicon


@dataclass(frozen=True)
class LampungResponse:
    text: str
    source: str


class LampungPipeline:
    """Lookup-first Lampung inference with model fallback."""

    def __init__(
        self,
        generator: Generator,
        tokenizer,
        lookup: Optional[InstructionLookup] = None,
        lexicon: Optional[LampungLexicon] = None,
        composer: Optional[LampungCompositionalTranslator] = None,
    ):
        self.generator = generator
        self.tokenizer = tokenizer
        self.lookup = lookup or InstructionLookup.from_final_dir()
        self.lexicon = lexicon or LampungLexicon("data/lampung/final/train.jsonl")
        self.composer = composer or LampungCompositionalTranslator()

    def translate(
        self,
        source_lang: str,
        target_lang: str,
        text: str,
        max_new_tokens: int = 60,
    ) -> LampungResponse:
        result = self.lookup.translate_with_source(source_lang, target_lang, text)
        if result:
            return LampungResponse(result.output, result.source)

        if source_lang == "Bahasa Indonesia" and target_lang == "Lampung O":
            composed = self.composer.translate_id_to_lo(text)
            if composed:
                return LampungResponse(composed.output, composed.source)

        if source_lang == "Lampung O" and target_lang == "English":
            composed = self.composer.translate_lo_to_en(text)
            if composed:
                return LampungResponse(composed.output, composed.source)

        prompt = build_translation_prompt(source_lang, target_lang, text)
        return self._generate_response(prompt, max_new_tokens, "model generation")

    def reason_lo_to_id(self, text: str, max_new_tokens: int = 80) -> LampungResponse:
        ordered = self.lookup.reorder_with_source("Lampung O", text)
        lookup_text = ordered.output if ordered else text

        result = self.lookup.translate_reasoning_with_source(
            "Lampung O",
            "Bahasa Indonesia",
            lookup_text,
        )
        if result:
            source = result.source
            if ordered:
                source = f"{source} + {ordered.source}"
            return LampungResponse(result.output, source)

        context = self.lexicon.build_context(text)
        prompt = build_reasoning_prompt(
            "Terjemahkan Lampung O ke Bahasa Indonesia dan jelaskan kata per kata",
            text,
            context=context,
        )
        return self._generate_response(prompt, max_new_tokens, "model generation")

    def reason(
        self,
        instruction: str,
        text: str,
        max_new_tokens: int = 80,
    ) -> LampungResponse:
        reasoning = self.lookup.get_reasoning(instruction, text)
        if reasoning:
            return LampungResponse(reasoning, "exact instruction lookup")

        context = self.lexicon.build_context(text)
        prompt = build_reasoning_prompt(instruction, text, context=context)
        return self._generate_response(prompt, max_new_tokens, "model generation")

    def reorder(self, lang: str, words: str, max_new_tokens: int = 60) -> LampungResponse:
        result = self.lookup.reorder_with_source(lang, words)
        if result:
            return LampungResponse(result.output, result.source)

        prompt = build_word_order_prompt(lang, words)
        return self._generate_response(prompt, max_new_tokens, "model generation")

    def _generate_response(
        self,
        prompt: str,
        max_new_tokens: int,
        source: str,
    ) -> LampungResponse:
        output = self.generator.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1.2,
            stop_tokens=[
                self.tokenizer.special_tokens["<|end_turn|>"],
                self.tokenizer.special_tokens["<|user|>"],
                self.tokenizer.special_tokens["<|assistant|>"],
                self.tokenizer.eos_id,
            ],
        )
        return LampungResponse(output.strip(), source)
