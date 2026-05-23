from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from tokenizers import ByteLevelBPETokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizer.hf_tokenizer import SPECIAL_TOKEN_LIST


DEFAULT_CORPUS_PATHS = [
    "data/indonesian.txt",
    "data/english.txt",
    "data/corpus.txt",
    "data/raw/indonesian.txt",
    "data/raw/english.txt",
    "data/raw/code.txt",
    "data/raw/corpus.txt",
    "data/corpus/general_assistant_mining_train.jsonl",
    "data/corpus/siger_base_trilingual_mix_train.jsonl",
    "data/corpus/siger_continue_lampung_identity_train.jsonl",
    "data/corpus/software_engineering_instruction_train.jsonl",
    "data/corpus/reasoning_instruction_train.jsonl",
    "data/capabilities/uncertainty_seed.jsonl",
    "data/lampung/final/train.jsonl",
    "data/lampung/final/train_instruction.jsonl",
    "data/lampung/final/train_augmented_instruction.jsonl",
    "data/lampung/processed/percakapan_1000_pairs.jsonl",
    "data/lampung/processed/compositional_pairs.jsonl",
]


@dataclass
class TokenizerTrainingConfig:
    files: list[str] = field(default_factory=lambda: list(DEFAULT_CORPUS_PATHS))
    save_dir: str = "checkpoints/tokenizer_hf_bpe"
    vocab_size: int = 32000
    min_frequency: int = 2
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKEN_LIST))


def existing_files(paths: Iterable[str | Path]) -> list[str]:
    files = []
    for path in paths:
        candidate = Path(path)
        if candidate.exists() and candidate.is_file():
            files.append(str(candidate))
    return files


def train_bytelevel_bpe(config: TokenizerTrainingConfig) -> Path:
    files = existing_files(config.files)
    if not files:
        raise FileNotFoundError(
            "Tidak ada corpus tokenizer yang ditemukan. "
            "Isi data/*.txt, build corpus JSONL, atau kirim --files path corpus."
        )

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=files,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=config.special_tokens,
    )

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_model(str(save_dir))

    metadata = asdict(config)
    metadata["files"] = files
    metadata["actual_vocab_size"] = tokenizer.get_vocab_size()
    metadata["backend"] = "hf_bpe"
    (save_dir / "tokenizer_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Tokenizer trained | files={len(files)} | vocab_size={tokenizer.get_vocab_size()}")
    print(f"Saved to: {save_dir}")
    return save_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SigerLM HF ByteLevel BPE tokenizer.")
    parser.add_argument("--files", nargs="*", default=None, help="Corpus files. Defaults to known project corpora.")
    parser.add_argument("--save-dir", default="checkpoints/tokenizer_hf_bpe")
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TokenizerTrainingConfig(
        files=args.files if args.files else list(DEFAULT_CORPUS_PATHS),
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    train_bytelevel_bpe(config)


if __name__ == "__main__":
    main()
