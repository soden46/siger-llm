# tokenizer/tokenizer.py
import tiktoken
import json
import os
from pathlib import Path
from typing import List, Union, Optional
from .special_tokens import SPECIAL_TOKENS, ID_TO_SPECIAL, ALLOWED_SPECIAL_IN_TEXT


class MultilingualTokenizer:
    """
    Tiktoken-based multilingual tokenizer.
    Base: cl100k_base (GPT-4 encoding) — 100k vocab, handles UTF-8 multilingual
    Extended: tambah special tokens untuk chat, lang-id, dsb.
    """

    BASE_ENCODING = "cl100k_base"  # atau "o200k_base" (GPT-4o, lebih gede)

    def __init__(self, custom_vocab_path: Optional[str] = None):
        self.special_tokens = SPECIAL_TOKENS.copy()
        self.id_to_special = ID_TO_SPECIAL.copy()

        # Load custom vocab tambahan kalau ada
        if custom_vocab_path and os.path.exists(custom_vocab_path):
            self._load_custom_vocab(custom_vocab_path)

        # Build tiktoken encoder dengan special tokens
        self._build_encoder()

        # Shortcuts buat training
        self.pad_id  = self.special_tokens["<|pad|>"]
        self.eos_id  = self.special_tokens["<|eos|>"]
        self.bos_id  = self.special_tokens["<|bos|>"]
        self.unk_id  = self.special_tokens["<|unk|>"]

    def _build_encoder(self):
        base = tiktoken.get_encoding(self.BASE_ENCODING)

        # Extend encoder base dengan special tokens lo
        self.encoder = tiktoken.Encoding(
            name="multilingual_llm",
            pat_str=base._pat_str,           # regex split pattern aslinya
            mergeable_ranks=base._mergeable_ranks,  # BPE merge rules
            special_tokens=self.special_tokens
        )

        self.vocab_size = self.encoder.n_vocab
        print(f"✅ Tokenizer ready | vocab_size={self.vocab_size}")

    # ─────────────────────────────────────────
    # ENCODE
    # ─────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,  # "id", "en", "code"
    ) -> List[int]:
        """
        Encode teks → list of token IDs.

        Args:
            text     : Input string
            add_bos  : Prepend <|bos|>
            add_eos  : Append <|eos|>
            lang     : Prepend language tag token

        Returns:
            List[int] token IDs
        """
        tokens = []

        if add_bos:
            tokens.append(self.bos_id)

        # Prepend language tag
        if lang:
            lang_token = f"<|{lang}|>"
            if lang_token in self.special_tokens:
                tokens.append(self.special_tokens[lang_token])

        # Encode teks utama
        # allowed_special: token spesial yang boleh muncul di teks input
        encoded = self.encoder.encode(
            text,
            allowed_special=ALLOWED_SPECIAL_IN_TEXT,
            disallowed_special=()  # jangan raise error kalau ketemu token aneh
        )
        tokens.extend(encoded)

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,
    ) -> List[List[int]]:
        """Encode multiple texts sekaligus."""
        return [self.encode(t, add_bos, add_eos, lang) for t in texts]

    # ─────────────────────────────────────────
    # DECODE
    # ─────────────────────────────────────────

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode list of token IDs → string.

        Args:
            token_ids           : List of IDs
            skip_special_tokens : Kalau True, buang special tokens dari output
        """
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids
                if t not in self.id_to_special
            ]

        return self.encoder.decode(token_ids)

    def decode_batch(
        self,
        batch: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    # ─────────────────────────────────────────
    # PADDING & TRUNCATION (buat DataLoader)
    # ─────────────────────────────────────────

    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        pad_left: bool = False,     # False = pad kanan (standard)
        truncate: bool = True,
    ) -> List[int]:
        """Pad atau truncate sequence ke max_length."""
        if truncate and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        pad_len = max_length - len(token_ids)
        padding = [self.pad_id] * pad_len

        return padding + token_ids if pad_left else token_ids + padding

    def pad_batch(
        self,
        batch: List[List[int]],
        max_length: Optional[int] = None,
        pad_left: bool = False,
    ) -> tuple:
        """
        Pad seluruh batch ke length yang sama.
        Return: (padded_batch, attention_mask)
        """
        if max_length is None:
            max_length = max(len(ids) for ids in batch)

        padded, masks = [], []
        for ids in batch:
            original_len = min(len(ids), max_length)
            padded_ids = self.pad_sequence(ids, max_length, pad_left)
            padded.append(padded_ids)

            # Attention mask: 1 = real token, 0 = padding
            if pad_left:
                mask = [0] * (max_length - original_len) + [1] * original_len
            else:
                mask = [1] * original_len + [0] * (max_length - original_len)
            masks.append(mask)

        return padded, masks

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def token_to_id(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        ids = self.encoder.encode(token, allowed_special="all")
        return ids[0] if ids else self.unk_id

    def id_to_token(self, token_id: int) -> str:
        if token_id in self.id_to_special:
            return self.id_to_special[token_id]
        return self.encoder.decode([token_id])

    def count_tokens(self, text: str) -> int:
        """Hitung jumlah token tanpa buat list penuh — efisien."""
        return len(self.encoder.encode(text, allowed_special="all"))

    def _load_custom_vocab(self, path: str):
        """Load custom tokens tambahan dari JSON."""
        with open(path) as f:
            custom = json.load(f)
        self.special_tokens.update(custom)
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        print(f"📦 Loaded {len(custom)} custom tokens from {path}")

    def save_config(self, save_dir: str):
        """Simpan konfigurasi tokenizer."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config = {
            "base_encoding": self.BASE_ENCODING,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(f"{save_dir}/tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"💾 Tokenizer config saved to {save_dir}/")

    def __repr__(self):
        return (
            f"MultilingualTokenizer("
            f"base={self.BASE_ENCODING}, "
            f"vocab_size={self.vocab_size}, "
            f"special_tokens={len(self.special_tokens)})"
        )