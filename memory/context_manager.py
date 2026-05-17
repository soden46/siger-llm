from typing import Optional

from memory.session_memory import SessionMemory


class ContextManager:
    def __init__(
        self,
        tokenizer,
        memory: Optional[SessionMemory] = None,
        max_context_tokens: int = 1024,
        retrieval_top_k: int = 5,
    ):
        self.tokenizer = tokenizer
        self.memory = memory or SessionMemory()
        self.max_context_tokens = max_context_tokens
        self.retrieval_top_k = retrieval_top_k

    def build_prompt(
        self,
        user_message: str,
        system_prompt: str,
    ) -> str:
        parts = []

        parts.append(f"<|system|>{system_prompt}<|end_turn|>")

        if self.memory.pinned_facts:
            facts = "\n".join(f"- {f}" for f in self.memory.pinned_facts)
            parts.append(f"<|system|>Fakta penting:\n{facts}<|end_turn|>")

        if self.memory.summary:
            parts.append(f"<|system|>Ringkasan percakapan sebelumnya:\n{self.memory.summary}<|end_turn|>")

        retrieved = self.memory.retrieve(user_message, top_k=self.retrieval_top_k)
        if retrieved:
            context = "\n\n".join(chunk.text for chunk in retrieved)
            parts.append(f"<|system|>Konteks relevan:\n{context}<|end_turn|>")

        for turn in self.memory.recent_turns():
            parts.append(f"<|{turn.role}|>{turn.content}<|end_turn|>")

        parts.append(f"<|user|>{user_message}<|end_turn|>")
        parts.append("<|assistant|>")

        prompt = "\n".join(parts)
        return self._fit_budget(prompt, system_prompt, user_message)

    def _fit_budget(
        self,
        prompt: str,
        system_prompt: str,
        user_message: str,
    ) -> str:
        if self.tokenizer.count_tokens(prompt) <= self.max_context_tokens:
            return prompt

        # Drop retrieval first
        parts = [
            f"<|system|>{system_prompt}<|end_turn|>",
        ]

        if self.memory.pinned_facts:
            facts = "\n".join(f"- {f}" for f in self.memory.pinned_facts)
            parts.append(f"<|system|>Fakta penting:\n{facts}<|end_turn|>")

        if self.memory.summary:
            parts.append(f"<|system|>Ringkasan percakapan sebelumnya:\n{self.memory.summary}<|end_turn|>")

        for turn in self.memory.recent_turns():
            parts.append(f"<|{turn.role}|>{turn.content}<|end_turn|>")

        parts.append(f"<|user|>{user_message}<|end_turn|>")
        parts.append("<|assistant|>")

        prompt = "\n".join(parts)

        # If still too long, trim recent turns from the oldest side
        recent = self.memory.recent_turns()
        while self.tokenizer.count_tokens(prompt) > self.max_context_tokens and recent:
            recent = recent[1:]
            parts = [f"<|system|>{system_prompt}<|end_turn|>"]

            if self.memory.summary:
                parts.append(f"<|system|>Ringkasan percakapan sebelumnya:\n{self.memory.summary}<|end_turn|>")

            for turn in recent:
                parts.append(f"<|{turn.role}|>{turn.content}<|end_turn|>")

            parts.append(f"<|user|>{user_message}<|end_turn|>")
            parts.append("<|assistant|>")
            prompt = "\n".join(parts)

        return prompt