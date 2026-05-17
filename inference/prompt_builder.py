def build_translation_prompt(source_lang: str, target_lang: str, text: str) -> str:
    return (
        "<|system|>Kamu adalah penerjemah Bahasa Lampung, Bahasa Indonesia, "
        "dan English. Jawab hanya hasil terjemahan, tanpa penjelasan.<|end_turn|>\n"
        f"<|user|>Terjemahkan {source_lang} ke {target_lang}:\n\n{text}<|end_turn|>\n"
        "<|assistant|>"
    )


def build_reasoning_prompt(instruction: str, text: str, context: str = "") -> str:
    context_block = f"Konteks kamus:\n{context}\n\n" if context else ""
    return (
        "<|system|>Kamu adalah asisten bahasa Lampung. Jawab dengan format "
        "Penjelasan lalu Jawaban.<|end_turn|>\n"
        f"<|user|>{context_block}{instruction}\n\n{text}<|end_turn|>\n"
        "<|assistant|>Penjelasan:\n"
    )


def build_word_order_prompt(lang: str, words: str) -> str:
    return (
        "<|system|>Kamu adalah asisten bahasa Lampung. Susun kata menjadi "
        "kalimat yang benar. Jawab hanya kalimat hasil susunan.<|end_turn|>\n"
        f"<|user|>Susun kata {lang} berikut menjadi kalimat yang benar:\n\n"
        f"{words}<|end_turn|>\n"
        "<|assistant|>"
    )


def build_chat_prompt(message: str) -> str:
    return (
        "<|system|>Kamu adalah asisten AI yang ringkas, ramah, dan akurat. "
        "Jawab dalam bahasa yang sama dengan user.<|end_turn|>\n"
        f"<|user|>{message}<|end_turn|>\n"
        "<|assistant|>"
    )
