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

    # Explicit corpus tags for trilingual/domain-aware SigerLM.
    # Keep these stable once a tokenizer/checkpoint is trained with them.
    "<|lang:id|>":        100271,
    "<|lang:en|>":        100272,
    "<|lang:lampung_o|>": 100273,
    "<|lang:mixed|>":     100274,
    "<|domain:general|>": 100275,
    "<|domain:code|>":    100276,
    "<|domain:math|>":    100277,
    "<|domain:laravel|>": 100278,
    "<|domain:debug|>":   100279,
    "<|domain:safety|>":  100280,
    "<|domain:translation|>": 100281,

    # Structural
    "<|bos|>":           100268,   # Beginning of sequence
    "<|eos|>":           100269,   # End of sequence
    "<|sep|>":           100270,   # Separator antar segment
}

# Reverse mapping: id → token string
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

# Token spesial yang boleh muncul di teks input dan HARUS dipertahankan
# sebagai satu token utuh oleh tiktoken.
#
# Ini penting untuk:
# - instruction tuning
# - chat format
# - LoRA loss masking
# - language tag tokens
ALLOWED_SPECIAL_IN_TEXT = set(SPECIAL_TOKENS.keys())
