from typing import Optional

from memory import ContextManager, SessionMemory

from .generator import Generator


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
    ):
        self.generator = generator
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.max_history = max_history
        self.max_context_tokens = max_context_tokens
        self.history: list[dict] = []

        self.memory = SessionMemory(max_recent_turns=max_history)
        self.context_manager = ContextManager(
            tokenizer=self.generator.tokenizer,
            memory=self.memory,
            max_context_tokens=max_context_tokens,
            retrieval_top_k=retrieval_top_k,
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
        prompt = self._build_prompt(user_input)

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

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2 :]

        self.memory.add_turn("user", user_input)
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
        return full_response.strip()

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        self.memory.add_document(text, metadata=metadata)

    def add_pinned_fact(self, fact: str) -> None:
        self.memory.add_pinned_fact(fact)

    def memory_stats(self) -> dict:
        return {
            "turns": len(self.memory.turns),
            "chunks": len(self.memory.chunk_store.chunks),
            "pinned_facts": len(self.memory.pinned_facts),
            "summary_chars": len(self.memory.summary),
            "max_context_tokens": self.max_context_tokens,
        }

    def reset(self) -> None:
        self.history.clear()
        self.memory.clear()
        print("Chat session reset.")

    def show_history(self) -> None:
        print("\n" + "-" * 50)
        for turn in self.history:
            role = turn["role"].upper()
            print(f"[{role}]: {turn['content'][:100]}...")
        print("-" * 50 + "\n")
