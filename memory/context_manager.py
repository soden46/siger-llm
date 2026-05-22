from typing import Optional

from memory.session_memory import SessionMemory


class ContextManager:
    def __init__(
        self,
        tokenizer,
        memory: Optional[SessionMemory] = None,
        max_context_tokens: int = 1024,
        retrieval_top_k: int = 5,
        retrieval_token_budget: int | None = None,
        recent_turn_token_budget: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.memory = memory or SessionMemory()
        self.max_context_tokens = max_context_tokens
        self.retrieval_top_k = retrieval_top_k
        self.retrieval_token_budget = retrieval_token_budget
        self.recent_turn_token_budget = recent_turn_token_budget

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

        if self.memory.long_context_summary:
            parts.append(f"<|system|>Ringkasan konteks panjang:\n{self.memory.long_context_summary}<|end_turn|>")

        context = self._build_retrieval_context(user_message)
        if context:
            parts.append(f"<|system|>Konteks relevan:\n{context}<|end_turn|>")

        for turn in self._budget_recent_turns():
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

        if self.memory.long_context_summary:
            parts.append(f"<|system|>Ringkasan konteks panjang:\n{self.memory.long_context_summary}<|end_turn|>")

        for turn in self._budget_recent_turns():
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

            if self.memory.long_context_summary:
                parts.append(f"<|system|>Ringkasan konteks panjang:\n{self.memory.long_context_summary}<|end_turn|>")

            for turn in recent:
                parts.append(f"<|{turn.role}|>{turn.content}<|end_turn|>")

            parts.append(f"<|user|>{user_message}<|end_turn|>")
            parts.append("<|assistant|>")
            prompt = "\n".join(parts)

        if self._count_tokens(prompt) > self.max_context_tokens:
            parts = [f"<|system|>{system_prompt}<|end_turn|>"]

            long_summary_budget = max(0, int(self.max_context_tokens * 0.20))
            if self.memory.long_context_summary and long_summary_budget > 32:
                compact_summary = self._truncate_to_tokens(
                    self.memory.long_context_summary,
                    long_summary_budget,
                )
                parts.append(f"<|system|>Ringkasan konteks panjang:\n{compact_summary}<|end_turn|>")

            reserved = self._count_tokens("\n".join(parts + ["<|assistant|>"]))
            user_budget = max(64, self.max_context_tokens - reserved - 32)
            compact_user = self._truncate_to_tokens(user_message, user_budget)
            parts.append(f"<|user|>{compact_user}<|end_turn|>")
            parts.append("<|assistant|>")
            prompt = "\n".join(parts)

        return prompt

    def _build_retrieval_context(self, user_message: str) -> str:
        retrieved = self.memory.retrieve(user_message, top_k=self.retrieval_top_k)
        if not retrieved:
            return ""

        token_budget = self.retrieval_token_budget
        if token_budget is None:
            token_budget = max(96, int(self.max_context_tokens * 0.35))

        parts: list[str] = []
        used = 0
        for i, chunk in enumerate(retrieved, start=1):
            label = chunk.metadata.get("source") or chunk.metadata.get("type") or f"chunk_{i}"
            candidate = f"[{i}: {label}]\n{chunk.text}"
            n_tokens = self._count_tokens(candidate)
            if used + n_tokens > token_budget:
                remaining = token_budget - used
                if remaining > 40:
                    candidate = self._truncate_to_tokens(candidate, remaining)
                    parts.append(candidate)
                break
            parts.append(candidate)
            used += n_tokens

        return "\n\n".join(parts)

    def _budget_recent_turns(self):
        recent = self.memory.recent_turns()
        token_budget = self.recent_turn_token_budget
        if token_budget is None:
            token_budget = max(96, int(self.max_context_tokens * 0.30))

        selected = []
        used = 0
        for turn in reversed(recent):
            text = f"<|{turn.role}|>{turn.content}<|end_turn|>"
            n_tokens = self._count_tokens(text)
            if used + n_tokens > token_budget:
                continue
            selected.append(turn)
            used += n_tokens
        return list(reversed(selected))

    def _count_tokens(self, text: str) -> int:
        try:
            return int(self.tokenizer.count_tokens(text))
        except Exception:
            return max(1, len(text) // 4)

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        try:
            ids = self.tokenizer.encode(text)
            if len(ids) <= max_tokens:
                return text
            return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=False)
        except Exception:
            return text[: max(1, max_tokens * 4)].rstrip()
