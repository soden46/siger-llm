# tokenizer/vocab_extender.py
"""
Extend vocab tiktoken dengan token-token baru.

Gunakan ini kalau lo mau tambah:
- Token Bahasa Lampung yang sering muncul tapi di-split aneh
- Token khusus domain (nama daerah, istilah adat, dsb)
- Token Bahasa Indonesia yang kurang optimal di cl100k

Catatan penting:
  Tiktoken tidak bisa benar-benar menambah merge rules BPE baru
  (itu butuh re-training tokenizer dari scratch).
  Yang bisa dilakukan: tambah SPECIAL TOKENS yang diperlakukan
  sebagai single token utuh, tidak di-split.
"""
import json
import tiktoken
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

from .special_tokens import SPECIAL_TOKENS


# ── Token Lampung yang direkomendasikan sebagai special tokens ──
# Kata-kata yang sering salah di-split oleh cl100k karena
# karakter/pola yang tidak ada di corpus training tiktoken
LAMPUNG_VOCAB_ADDITIONS = {
    # Sapaan & pronoun Lampung Dialek O
    "<|nyak|>":    100280,   # saya
    "<|niku|>":    100281,   # kamu
    "<|jak|>":     100282,   # kita
    "<|api|>":     100283,   # apa
    "<|haga|>":    100284,   # mau/ingin
    "<|lapah|>":   100285,   # pergi
    "<|mengan|>":  100286,   # makan
    "<|minom|>":   100287,   # minum
    "<|temon|>":   100288,   # betul/benar
    "<|munih|>":   100289,   # juga

    # Language tag Lampung
    "<|lampung_o|>":   100290,   # Bahasa Lampung Dialek O
    "<|lampung_nyo|>": 100291,   # Bahasa Lampung Dialek Nyo
}


class VocabExtender:
    """
    Extend MultilingualTokenizer dengan token tambahan.

    Cara kerja:
    1. Token baru didaftarkan sebagai special tokens
    2. Tiktoken encoder di-rebuild dengan special tokens yang diperluas
    3. Token baru diperlakukan sebagai SINGLE TOKEN (tidak di-split)

    Ini berguna untuk:
    - Kata Lampung yang sering di-split jadi 3-4 token aneh
    - Istilah domain yang panjang tapi harus jadi 1 token
    - Efisiensi tokenisasi untuk bahasa daerah
    """

    def __init__(self, tokenizer):
        """
        Args:
            tokenizer: instance MultilingualTokenizer yang sudah dibuat
        """
        self.tokenizer = tokenizer
        self.added_tokens: Dict[str, int] = {}

    def add_tokens(self, new_tokens: Dict[str, int]) -> int:
        """
        Tambah token baru ke vocabulary.

        Args:
            new_tokens: dict {token_string: token_id}
                        token_id harus lebih besar dari vocab_size saat ini

        Returns:
            Jumlah token yang berhasil ditambahkan
        """
        # Validasi ID tidak bentrok
        existing_ids = set(self.tokenizer.special_tokens.values())
        conflicts    = []

        for token, token_id in new_tokens.items():
            if token in self.tokenizer.special_tokens:
                print(f"⚠️  Token sudah ada, skip: {token}")
                continue
            if token_id in existing_ids:
                conflicts.append((token, token_id))
                continue
            self.added_tokens[token] = token_id
            existing_ids.add(token_id)

        if conflicts:
            print(f"❌ ID conflict untuk: {conflicts}")
            print("   Gunakan ID yang lebih besar dari vocab_size saat ini.")
            return 0

        if not self.added_tokens:
            return 0

        # Update special tokens tokenizer
        self.tokenizer.special_tokens.update(self.added_tokens)
        self.tokenizer.id_to_special = {
            v: k for k, v in self.tokenizer.special_tokens.items()
        }

        # Rebuild encoder
        self.tokenizer._build_encoder()

        print(f"✅ Ditambahkan {len(self.added_tokens)} token baru:")
        for token, token_id in self.added_tokens.items():
            print(f"   {token} → ID {token_id}")

        return len(self.added_tokens)

    def add_lampung_tokens(self) -> int:
        """
        Shortcut: tambah semua token Lampung yang direkomendasikan.
        """
        print("🌴 Menambahkan token Bahasa Lampung...")
        return self.add_tokens(LAMPUNG_VOCAB_ADDITIONS)

    def analyze_tokenization(
        self,
        texts: List[str],
        top_n: int = 20,
    ) -> Dict:
        """
        Analisis kata-kata yang paling banyak di-split jadi banyak token.
        Berguna untuk menentukan kandidat token baru yang perlu ditambah.

        Args:
            texts: list teks dalam bahasa target
            top_n: tampilkan N kata yang paling "boros" token

        Returns:
            dict berisi statistik tokenisasi
        """
        word_token_counts: Counter = Counter()
        total_words  = 0
        total_tokens = 0

        for text in texts:
            words = text.split()
            for word in words:
                word_lower = word.lower().strip(".,!?;:()")
                if not word_lower:
                    continue

                token_ids  = self.tokenizer.encode(word_lower)
                n_tokens   = len(token_ids)

                word_token_counts[word_lower] += n_tokens
                total_words  += 1
                total_tokens += n_tokens

        # Kata yang butuh paling banyak token per karakter
        inefficient = []
        for word, total_tok in word_token_counts.most_common(200):
            freq     = word_token_counts[word]
            avg_tok  = total_tok / max(freq, 1)
            tok_per_char = avg_tok / max(len(word), 1)

            if avg_tok > 1.5:   # kata yang butuh > 1.5 token rata-rata
                inefficient.append({
                    "word":         word,
                    "avg_tokens":   round(avg_tok, 2),
                    "tok_per_char": round(tok_per_char, 3),
                })

        inefficient.sort(key=lambda x: x["avg_tokens"], reverse=True)

        result = {
            "total_words":        total_words,
            "total_tokens":       total_tokens,
            "compression_ratio":  round(total_tokens / max(total_words, 1), 3),
            "inefficient_words":  inefficient[:top_n],
        }

        print(f"\n📊 Tokenization Analysis:")
        print(f"   Total words  : {total_words:,}")
        print(f"   Total tokens : {total_tokens:,}")
        print(f"   Ratio        : {result['compression_ratio']:.2f} tok/word")
        print(f"\n   Top {top_n} kata paling boros token:")
        for item in inefficient[:top_n]:
            print(f"   '{item['word']}' → {item['avg_tokens']} token avg")

        return result

    def suggest_additions(
        self,
        texts: List[str],
        min_avg_tokens: float = 2.0,
        top_n: int = 50,
    ) -> List[str]:
        """
        Sarankan kata-kata yang layak dijadikan token baru.

        Args:
            texts          : corpus teks dalam bahasa target
            min_avg_tokens : minimum rata-rata token supaya disarankan
            top_n          : jumlah kandidat yang dikembalikan

        Returns:
            List kata yang disarankan sebagai token baru
        """
        analysis   = self.analyze_tokenization(texts)
        candidates = [
            item["word"]
            for item in analysis["inefficient_words"]
            if item["avg_tokens"] >= min_avg_tokens
        ]

        print(f"\n💡 Kandidat token baru ({len(candidates)}):")
        for word in candidates[:top_n]:
            print(f"   '{word}'")

        return candidates[:top_n]

    def save_extended_vocab(self, save_path: str):
        """
        Simpan extended vocabulary ke JSON.
        Bisa di-load ulang dengan load_extended_vocab().
        """
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.added_tokens, f, indent=2, ensure_ascii=False)
        print(f"💾 Extended vocab saved: {save_path}")

    @classmethod
    def load_extended_vocab(cls, tokenizer, load_path: str) -> "VocabExtender":
        """
        Load dan apply extended vocabulary dari file JSON.

        Args:
            tokenizer : MultilingualTokenizer
            load_path : path ke JSON file extended vocab

        Returns:
            VocabExtender instance dengan tokens sudah di-apply
        """
        extender = cls(tokenizer)
        with open(load_path, "r", encoding="utf-8") as f:
            tokens = json.load(f)
        extender.add_tokens(tokens)
        print(f"📦 Extended vocab loaded: {load_path}")
        return extender


def demo_lampung_tokenization():
    """
    Demo: bandingkan tokenisasi sebelum dan sesudah extend vocab Lampung.
    """
    from tokenizer.tokenizer import MultilingualTokenizer

    tok      = MultilingualTokenizer()
    extender = VocabExtender(tok)

    test_sentences = [
        "nyak haga mengan di pasar",
        "api kabar niku sekeluarga",
        "jak lapah temon ke pekon",
        "niku munih haga minom kopi",
    ]

    print("=" * 50)
    print("SEBELUM extend vocab Lampung:")
    print("=" * 50)
    for sent in test_sentences:
        ids = tok.encode(sent)
        print(f"'{sent}' → {len(ids)} tokens")

    extender.add_lampung_tokens()

    print("\n" + "=" * 50)
    print("SESUDAH extend vocab Lampung:")
    print("=" * 50)
    for sent in test_sentences:
        ids = tok.encode(sent)
        print(f"'{sent}' → {len(ids)} tokens")


if __name__ == "__main__":
    demo_lampung_tokenization()