# inference/chat.py
import os
from typing import Optional
from .generator import Generator


class ChatSession:
    """
    Stateful chat session dengan history.
    Analoginya: kayak Session di Laravel — state disimpan per user.

    Format conversation:
    <|system|> ... <|end_turn|>
    <|user|>   ... <|end_turn|>
    <|assistant|> ... <|end_turn|>
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
        max_history: int = 10,          # max turn disimpan
        max_context_tokens: int = 1024, # max total token dikirim ke model
    ):
        self.generator         = generator
        self.system_prompt     = system_prompt or self.SYSTEM_PROMPT
        self.max_history       = max_history
        self.max_context_tokens = max_context_tokens
        self.history: list[dict] = []  # [{"role": ..., "content": ...}]

    def _build_prompt(self) -> str:
        """
        Bangun prompt string dari history.
        Format: system → [user/assistant turns] → assistant prefix
        """
        tok = self.generator.tokenizer
        parts = []

        # System
        parts.append(
            f"<|system|>{self.system_prompt}<|end_turn|>"
        )

        # History turns
        for turn in self.history:
            role    = turn["role"]
            content = turn["content"]
            parts.append(f"<|{role}|>{content}<|end_turn|>")

        # Prefix untuk response berikutnya
        parts.append("<|assistant|>")

        prompt = "\n".join(parts)

        # Truncate kalau terlalu panjang
        token_count = tok.count_tokens(prompt)
        if token_count > self.max_context_tokens:
            # Hapus history paling lama (kecuali system)
            while len(self.history) > 2 and token_count > self.max_context_tokens:
                self.history.pop(0)
                prompt = self._build_prompt()
                token_count = tok.count_tokens(prompt)

        return prompt

    def chat(
        self,
        user_input: str,
        stream: bool = False,
        **gen_kwargs,
    ) -> str:
        """
        Kirim pesan, dapat respons.
        stream=True → print karakter per karakter (terminal effect).
        """
        # Tambah user turn ke history
        self.history.append({"role": "user", "content": user_input})

        # Build full prompt
        prompt = self._build_prompt()

        # Generate
        if stream:
            response = self._stream_response(prompt, **gen_kwargs)
        else:
            response = self.generator.generate(
                prompt,
                stop_tokens=[
                    self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
                    self.generator.tokenizer.special_tokens.get("<|user|>"),
                    self.generator.tokenizer.eos_id,
                ],
                **gen_kwargs
            )

        # Simpan response ke history
        self.history.append({"role": "assistant", "content": response})

        # Trim history kalau udah terlalu panjang
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return response

    def _stream_response(self, prompt: str, **kwargs) -> str:
        """Stream response ke terminal, return full string."""
        full_response = ""
        print("Assistant: ", end="", flush=True)

        stop_ids = [
            self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
            self.generator.tokenizer.eos_id,
        ]

        for token_str in self.generator.stream(prompt, **kwargs):
            # Stop kalau ketemu end_turn dalam token string
            if any(s in token_str for s in ["<|end_turn|>", "<|user|>"]):
                break
            print(token_str, end="", flush=True)
            full_response += token_str

        print()  # newline setelah selesai
        return full_response.strip()

    def reset(self):
        """Reset history — mulai percakapan baru."""
        self.history.clear()
        print("🔄 Chat session reset.")

    def show_history(self):
        """Print seluruh history."""
        print(f"\n{'─'*50}")
        for turn in self.history:
            role = turn["role"].upper()
            print(f"[{role}]: {turn['content'][:100]}...")
        print(f"{'─'*50}\n")