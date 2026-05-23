import json
from pathlib import Path
from typing import List, Optional

from tokenizers import ByteLevelBPETokenizer


BASE_SPECIAL_TOKEN_LIST = [
    "<|endoftext|>",
    "<|pad|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|lang_id|>",
    "<|id|>",
    "<|en|>",
    "<|code|>",
    "<|bos|>",
    "<|eos|>",
    "<|sep|>",
]

DOMAIN_SPECIAL_TOKEN_LIST = [
    "<|lang:id|>",
    "<|lang:en|>",
    "<|lang:lampung_o|>",
    "<|lang:mixed|>",
    "<|domain:general|>",
    "<|domain:code|>",
    "<|domain:math|>",
    "<|domain:laravel|>",
    "<|domain:debug|>",
    "<|domain:safety|>",
    "<|domain:translation|>",
]

SPECIAL_TOKEN_LIST = BASE_SPECIAL_TOKEN_LIST + DOMAIN_SPECIAL_TOKEN_LIST


class HFMultilingualTokenizer:
    def __init__(self, tokenizer_dir="checkpoints/tokenizer_hf_bpe"):
        tokenizer_path = Path(tokenizer_dir)
        vocab_path = tokenizer_path / "vocab.json"
        merges_path = tokenizer_path / "merges.txt"

        if not vocab_path.exists() or not merges_path.exists():
            raise FileNotFoundError(
                "HF BPE tokenizer files not found. Run "
                "`python tokenizer/train_hf_bpe.py` first."
            )

        self.encoder = ByteLevelBPETokenizer(
            str(vocab_path),
            str(merges_path),
        )
        # Loading vocab/merges alone does not always restore the "special"
        # behavior. Register tokens already present in the tokenizer vocab so
        # old checkpoints do not silently get a larger runtime vocab.
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        runtime_special_tokens = [
            token
            for token in SPECIAL_TOKEN_LIST
            if token in vocab or token in BASE_SPECIAL_TOKEN_LIST
        ]
        self.encoder.add_special_tokens(runtime_special_tokens)

        self.special_tokens = {
            token: self.encoder.token_to_id(token)
            for token in runtime_special_tokens
        }
        missing = [
            token
            for token in BASE_SPECIAL_TOKEN_LIST
            if self.special_tokens.get(token) is None
        ]
        if missing:
            raise RuntimeError(f"HF tokenizer missing special tokens: {missing}")

        self.id_to_special = {v: k for k, v in self.special_tokens.items()}

        self.pad_id = self.special_tokens["<|pad|>"]
        self.eos_id = self.special_tokens["<|eos|>"]
        self.bos_id = self.special_tokens["<|bos|>"]
        self.unk_id = self.special_tokens["<|unk|>"]
        self.vocab_size = self.encoder.get_vocab_size()
        print(f"HF BPE tokenizer ready | vocab_size={self.vocab_size}")

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,
    ) -> List[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)

        if lang:
            token = f"<|{lang}|>"
            if token in self.special_tokens:
                ids.append(self.special_tokens[token])

        ids.extend(self.encoder.encode(text).ids)

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,
    ) -> List[List[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos, lang=lang) for t in texts]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.encoder.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self,
        batch: List[List[int]],
        skip_special_tokens: bool = True,
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens=skip_special_tokens) for ids in batch]

    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        pad_left: bool = False,
        truncate: bool = True,
    ) -> List[int]:
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
        if max_length is None:
            max_length = max(len(ids) for ids in batch)

        padded, masks = [], []
        for ids in batch:
            original_len = min(len(ids), max_length)
            padded_ids = self.pad_sequence(ids, max_length, pad_left)
            padded.append(padded_ids)

            if pad_left:
                mask = [0] * (max_length - original_len) + [1] * original_len
            else:
                mask = [1] * original_len + [0] * (max_length - original_len)
            masks.append(mask)

        return padded, masks

    def token_to_id(self, token: str) -> int:
        token_id = self.special_tokens.get(token)
        if token_id is not None:
            return token_id
        ids = self.encoder.encode(token).ids
        return ids[0] if ids else self.unk_id

    def id_to_token(self, token_id: int) -> str:
        if token_id in self.id_to_special:
            return self.id_to_special[token_id]
        return self.encoder.decode([token_id], skip_special_tokens=False)

    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text).ids)

    def save_config(self, save_dir: str):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config = {
            "backend": "hf_bpe",
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
        }
        with open(Path(save_dir) / "tokenizer_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def __repr__(self):
        return f"HFMultilingualTokenizer(vocab_size={self.vocab_size})"
