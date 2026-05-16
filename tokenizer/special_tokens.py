# tokenizer/special_tokens.py

SPECIAL_TOKENS = {
    # Core control tokens
    "<|endoftext|>":     100257,   # End of document (udah ada di cl100k)
    "<|pad|>":           100258,   # Padding
    "<|unk|>":           100259,   # Unknown

    # Conversation / instruction tokens (buat chat model nanti)
    "<|system|>":        100260,
    "<|user|>":          100261,
    "<|assistant|>":     100262,
    "<|end_turn|>":      100263,

    # Language tags (multilingual)
    "<|lang_id|>":       100264,   # generic lang marker
    "<|id|>":            100265,   # Bahasa Indonesia
    "<|en|>":            100266,   # English
    "<|code|>":          100267,   # Code block

    # Structural
    "<|bos|>":           100268,   # Beginning of sequence
    "<|eos|>":           100269,   # End of sequence
    "<|sep|>":           100270,   # Separator antar segment
}

# Reverse mapping: id → token string
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

# Token yang TIDAK boleh di-split saat encoding
ALLOWED_SPECIAL_IN_TEXT = {"<|endoftext|>"}