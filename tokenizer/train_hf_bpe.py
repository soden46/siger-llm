from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

SPECIAL_TOKENS = [
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

files = [
    "data/indonesian.txt",
    "data/english.txt",
    "data/corpus.txt",
    "data/lampung/final/train.jsonl",
    "data/lampung/final/train_instruction.jsonl",
    "data/lampung/final/train_augmented_instruction.jsonl",
    "data/lampung/processed/percakapan_1000_pairs.jsonl",
    "data/lampung/processed/compositional_pairs.jsonl",
]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=[str(f) for f in files if Path(f).exists()],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=SPECIAL_TOKENS,
)

save_dir = Path("checkpoints/tokenizer_hf_bpe")
save_dir.mkdir(parents=True, exist_ok=True)

tokenizer.save_model(str(save_dir))
