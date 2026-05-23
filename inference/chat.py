from typing import Optional

from memory import ContextManager, SessionMemory

from .generator import Generator
from .tool_result_compressor import CompressionResult, ToolResultCompressor


STOP_MARKERS = ("<|end_turn|>", "<|user|>", "<|assistant|>", "<|system|>", "<|eos|>", "<|bos|>")


def clean_assistant_response(text: str) -> str:
    """Trim generated turn markers before storing or printing chat output."""
    cleaned = text
    for marker in STOP_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker, 1)[0]
    return cleaned.strip()


class ChatSession:
    """
    Stateful chat session backed by long-session memory.

    The session keeps all turns in SessionMemory, retrieves relevant chunks,
    keeps recent turns, and builds a bounded prompt through ContextManager.
    """

    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang cerdas dan helpful. "
        "Jawab dalam bahasa yang sama dengan pertanyaan user. "
        "Jawab dengan jelas, ringkas, dan akurat."
    )

    def __init__(
        self,
        generator: Generator,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
        max_context_tokens: int = 1024,
        retrieval_top_k: int = 5,
        retrieval_token_budget: int | None = None,
        recent_turn_token_budget: int | None = None,
        long_input_threshold_chars: int = 1200,
        document_chunk_size_words: int = 180,
        document_overlap_words: int = 40,
        tool_result_compressor: ToolResultCompressor | None = None,
    ):
        self.generator = generator
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.long_input_threshold_chars = long_input_threshold_chars
        self.document_chunk_size_words = document_chunk_size_words
        self.document_overlap_words = document_overlap_words
        self.history: list[dict] = []
        self.tool_result_compressor = tool_result_compressor or ToolResultCompressor()
        self.tool_compression_stats = {
            "tool_results": 0,
            "compressed_tool_results": 0,
            "original_chars": 0,
            "stored_chars": 0,
        }

        self.memory = SessionMemory(max_recent_turns=max_history)
        self.context_manager = ContextManager(
            tokenizer=self.generator.tokenizer,
            memory=self.memory,
            max_context_tokens=max_context_tokens,
            retrieval_top_k=retrieval_top_k,
            retrieval_token_budget=retrieval_token_budget,
            recent_turn_token_budget=recent_turn_token_budget,
        )

    def _build_prompt(self, user_input: str) -> str:
        return self.context_manager.build_prompt(
            user_message=user_input,
            system_prompt=self.system_prompt,
        )

    def chat(
        self,
        user_input: str,
        stream: bool = False,
        **gen_kwargs,
    ) -> str:
        effective_user_input = self.memory.ingest_long_user_message(
            user_input,
            max_inline_chars=self.long_input_threshold_chars,
            chunk_size_words=self.document_chunk_size_words,
            overlap_words=self.document_overlap_words,
        )
        prompt = self._build_prompt(effective_user_input)

        if stream:
            response = self._stream_response(prompt, **gen_kwargs)
        else:
            response = self.generator.generate(
                prompt,
                stop_tokens=[
                    self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
                    self.generator.tokenizer.special_tokens.get("<|user|>"),
                    self.generator.tokenizer.special_tokens.get("<|assistant|>"),
                    self.generator.tokenizer.eos_id,
                ],
                **gen_kwargs,
            )
            response = clean_assistant_response(response)

        self.history.append({"role": "user", "content": effective_user_input})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        self.memory.add_turn("user", effective_user_input)
        self.memory.add_turn("assistant", response)
        return response

    def _stream_response(self, prompt: str, **kwargs) -> str:
        full_response = ""
        print("Assistant: ", end="", flush=True)

        for token_str in self.generator.stream(prompt, **kwargs):
            if any(s in token_str for s in ["<|end_turn|>", "<|user|>", "<|assistant|>"]):
                break
            print(token_str, end="", flush=True)
            full_response += token_str

        print()
        return clean_assistant_response(full_response)

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        self.memory.add_document(
            text,
            metadata=metadata,
            chunk_size_words=self.document_chunk_size_words,
            overlap_words=self.document_overlap_words,
        )

    def add_tool_result(
        self,
        output: str,
        command: str = "",
        metadata: Optional[dict] = None,
    ) -> CompressionResult:
        result = self.tool_result_compressor.compress(
            output,
            command=command,
            source=(metadata or {}).get("source", "tool"),
        )
        tool_metadata = {
            **result.metadata(),
            **(metadata or {}),
            "command": command,
        }
        self.add_document(result.text, metadata=tool_metadata)
        self.tool_compression_stats["tool_results"] += 1
        self.tool_compression_stats["compressed_tool_results"] += int(result.compressed)
        self.tool_compression_stats["original_chars"] += result.original_chars
        self.tool_compression_stats["stored_chars"] += result.compressed_chars
        return result

    def add_pinned_fact(self, fact: str) -> None:
        self.memory.add_pinned_fact(fact)

    def memory_stats(self) -> dict:
        return {
            "turns": len(self.memory.turns),
            "chunks": len(self.memory.chunk_store.chunks),
            "documents": self.memory.document_count,
            "long_context_chars": self.memory.long_context_chars,
            "pinned_facts": len(self.memory.pinned_facts),
            "summary_chars": len(self.memory.summary),
            "long_summary_chars": len(self.memory.long_context_summary),
            "max_context_tokens": self.max_context_tokens,
            "tool_compression": dict(self.tool_compression_stats),
        }

    def reset(self) -> None:
        self.history.clear()
        self.memory.clear()
        self.tool_compression_stats = {
            "tool_results": 0,
            "compressed_tool_results": 0,
            "original_chars": 0,
            "stored_chars": 0,
        }
        print("Chat session reset.")

    def show_history(self) -> None:
        print("\n" + "-" * 50)
        for turn in self.history:
            role = turn["role"].upper()
            print(f"[{role}]: {turn['content'][:100]}...")
        print("-" * 50 + "\n")
