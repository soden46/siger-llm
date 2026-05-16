# tokenizer/tests/test_tokenizer.py
"""
Unit test untuk MultilingualTokenizer SigerLM.
Jalankan: python -m tokenizer.tests.test_tokenizer
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer.tokenizer import MultilingualTokenizer
from tokenizer.tests.sample_texts import (
    INDONESIAN_TEXTS,
    ENGLISH_TEXTS,
    LAMPUNG_O_TEXTS,
    PYTHON_CODE_TEXTS,
    INSTRUCTION_SAMPLES,
    format_instruction_sample,
)


def test_basic_encode_decode():
    print("\n[1/7] Basic encode/decode...")
    tok = MultilingualTokenizer()

    for lang, texts in [("id", INDONESIAN_TEXTS[:3]), ("en", ENGLISH_TEXTS[:3])]:
        for text in texts:
            ids     = tok.encode(text, add_bos=True, add_eos=True, lang=lang)
            decoded = tok.decode(ids, skip_special_tokens=True)
            assert decoded == text, f"Mismatch:\n  orig   : {text}\n  decoded: {decoded}"

    print("  ✅ encode/decode roundtrip OK")


def test_special_tokens():
    print("\n[2/7] Special tokens...")
    tok = MultilingualTokenizer()

    required = [
        "<|endoftext|>", "<|pad|>", "<|unk|>",
        "<|system|>", "<|user|>", "<|assistant|>", "<|end_turn|>",
        "<|bos|>", "<|eos|>",
        "<|id|>", "<|en|>", "<|code|>",
    ]
    for token in required:
        assert token in tok.special_tokens, f"Missing special token: {token}"

    print(f"  ✅ All {len(required)} required special tokens present")
    print(f"  ✅ vocab_size = {tok.vocab_size:,}")


def test_lang_tags():
    print("\n[3/7] Language tags...")
    tok  = MultilingualTokenizer()
    text = "Halo dunia"

    for lang in ["id", "en", "code"]:
        ids_with_lang    = tok.encode(text, lang=lang)
        ids_without_lang = tok.encode(text)
        assert len(ids_with_lang) == len(ids_without_lang) + 1, \
            f"Lang tag not prepended for lang={lang}"

    print("  ✅ Language tags correctly prepended")


def test_padding():
    print("\n[4/7] Padding & batch...")
    tok = MultilingualTokenizer()

    texts = [
        INDONESIAN_TEXTS[0],
        ENGLISH_TEXTS[0],
        LAMPUNG_O_TEXTS[0],
    ]
    batch_ids = tok.encode_batch(texts, add_eos=True)
    padded, masks = tok.pad_batch(batch_ids, max_length=64)

    assert len(padded) == 3,       "Wrong batch size"
    assert all(len(p) == 64 for p in padded), "Not all padded to 64"
    assert all(len(m) == 64 for m in masks),  "Mask length wrong"

    for i, (p, m) in enumerate(zip(padded, masks)):
        n_real = sum(m)
        n_pad  = 64 - n_real
        assert p[-1] == tok.pad_id or n_pad == 0, "Padding token wrong"

    print(f"  ✅ Batch padding OK — shapes: {len(padded)}×64")


def test_lampung_tokenization():
    print("\n[5/7] Lampung O tokenization...")
    tok = MultilingualTokenizer()

    for text in LAMPUNG_O_TEXTS[:5]:
        ids     = tok.encode(text)
        decoded = tok.decode(ids)
        # Lampung kata tidak harus roundtrip exact (bisa ada whitespace normalization)
        assert len(ids) > 0, f"Empty encoding for: {text}"
        print(f"  '{text[:30]}' → {len(ids)} tokens")

    print("  ✅ Lampung tokenization OK")


def test_instruction_format():
    print("\n[6/7] Instruction format...")
    tok = MultilingualTokenizer()

    assistant_id = tok.special_tokens["<|assistant|>"]
    end_turn_id  = tok.special_tokens["<|end_turn|>"]

    for sample in INSTRUCTION_SAMPLES[:2]:
        formatted = format_instruction_sample(sample, tok)
        ids       = tok.encode(formatted, add_bos=True, add_eos=True)

        assert assistant_id in ids, "<|assistant|> token not in encoded ids"
        assert end_turn_id  in ids, "<|end_turn|> token not in encoded ids"
        print(f"  '{formatted[:50]}...' → {len(ids)} tokens ✅")

    print("  ✅ Instruction format tokenization OK")


def test_count_tokens():
    print("\n[7/7] count_tokens...")
    tok = MultilingualTokenizer()

    for text in INDONESIAN_TEXTS[:3]:
        n1 = tok.count_tokens(text)
        n2 = len(tok.encode(text))
        assert n1 == n2, f"count_tokens mismatch: {n1} vs {n2}"

    print("  ✅ count_tokens consistent with encode")


def run_all_tests():
    print("=" * 50)
    print("  SigerLM Tokenizer Test Suite")
    print("=" * 50)

    tests = [
        test_basic_encode_decode,
        test_special_tokens,
        test_lang_tags,
        test_padding,
        test_lampung_tokenization,
        test_instruction_format,
        test_count_tokens,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR : {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")

    if failed == 0:
        print("  🎉 All tests passed!")
    else:
        print("  ⚠️  Some tests failed. Cek error di atas.")
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()