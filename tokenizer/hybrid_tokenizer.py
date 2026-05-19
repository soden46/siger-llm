from __future__ import annotations

from pathlib import Path
from typing import Literal

from tokenizer.hf_tokenizer import HFMultilingualTokenizer
from tokenizer.tokenizer import MultilingualTokenizer


TokenizerBackend = Literal["auto", "tiktoken", "hf_bpe"]

REQUIRED_SPECIAL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
)


class HybridTokenizer:
    """
    Runtime tokenizer selector.

    This does not mix token id spaces inside one model. It selects one backend:
    - hf_bpe: custom GPT-style ByteLevel BPE, if trained files exist
    - tiktoken: cl100k_base wrapper
    - auto: hf_bpe when available, otherwise tiktoken
    """

    def __init__(
        self,
        backend: TokenizerBackend = "auto",
        hf_tokenizer_dir: str = "checkpoints/tokenizer_hf_bpe",
    ):
        self.backend = self._resolve_backend(backend, hf_tokenizer_dir)

        if self.backend == "hf_bpe":
            self.impl = HFMultilingualTokenizer(hf_tokenizer_dir)
        elif self.backend == "tiktoken":
            self.impl = MultilingualTokenizer()
        else:
            raise ValueError(f"Unknown tokenizer backend: {backend}")

        self.vocab_size = self.impl.vocab_size
        self.special_tokens = self.impl.special_tokens
        self.id_to_special = getattr(self.impl, "id_to_special", {})
        self.pad_id = self.impl.pad_id
        self.eos_id = self.impl.eos_id
        self.bos_id = self.impl.bos_id
        self.unk_id = self.impl.unk_id
        self._validate_special_tokens()

    @staticmethod
    def _resolve_backend(backend: TokenizerBackend, hf_tokenizer_dir: str) -> str:
        if backend != "auto":
            return backend

        path = Path(hf_tokenizer_dir)
        if (path / "vocab.json").exists() and (path / "merges.txt").exists():
            return "hf_bpe"
        return "tiktoken"

    def encode(self, *args, **kwargs):
        return self.impl.encode(*args, **kwargs)

    def encode_batch(self, *args, **kwargs):
        return self.impl.encode_batch(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.impl.decode(*args, **kwargs)

    def decode_batch(self, *args, **kwargs):
        return self.impl.decode_batch(*args, **kwargs)

    def pad_sequence(self, *args, **kwargs):
        return self.impl.pad_sequence(*args, **kwargs)

    def pad_batch(self, *args, **kwargs):
        return self.impl.pad_batch(*args, **kwargs)

    def token_to_id(self, *args, **kwargs):
        return self.impl.token_to_id(*args, **kwargs)

    def id_to_token(self, *args, **kwargs):
        return self.impl.id_to_token(*args, **kwargs)

    def count_tokens(self, *args, **kwargs):
        return self.impl.count_tokens(*args, **kwargs)

    def save_config(self, *args, **kwargs):
        return self.impl.save_config(*args, **kwargs)

    def compare_fragmentation(self, words: list[str]) -> list[dict]:
        tiktoken = MultilingualTokenizer()
        rows = []

        hf = None
        try:
            hf = HFMultilingualTokenizer()
        except FileNotFoundError:
            pass

        for word in words:
            row = {
                "word": word,
                "tiktoken_ids": tiktoken.encode(word),
                "tiktoken_n": tiktoken.count_tokens(word),
            }
            if hf is not None:
                row["hf_bpe_ids"] = hf.encode(word)
                row["hf_bpe_n"] = hf.count_tokens(word)
            rows.append(row)

        return rows

    def __repr__(self):
        return f"HybridTokenizer(backend={self.backend}, vocab_size={self.vocab_size})"

    def _validate_special_tokens(self) -> None:
        missing = [
            token
            for token in REQUIRED_SPECIAL_TOKENS
            if self.special_tokens.get(token) is None
        ]
        if missing:
            raise RuntimeError(
                f"Tokenizer backend={self.backend} missing special tokens: {missing}"
            )

        split_tokens: dict[str, list[int]] = {}
        for token in REQUIRED_SPECIAL_TOKENS:
            encoded = self.impl.encode(token)
            expected_id = self.special_tokens[token]
            if encoded != [expected_id]:
                split_tokens[token] = encoded

        if split_tokens:
            details = ", ".join(
                f"{token}->{ids}" for token, ids in split_tokens.items()
            )
            raise RuntimeError(
                "Tokenizer special tokens are not encoded as single registered IDs "
                f"for backend={self.backend}: {details}"
            )


def build_tokenizer(
    backend: TokenizerBackend = "auto",
    hf_tokenizer_dir: str = "checkpoints/tokenizer_hf_bpe",
):
    return HybridTokenizer(backend=backend, hf_tokenizer_dir=hf_tokenizer_dir)
