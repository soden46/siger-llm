# tokenizer/tests/test_tokenizer.py
from tokenizer.tokenizer import MultilingualTokenizer

tok = MultilingualTokenizer()

# ── Test 1: Basic encode/decode ──────────────────────────
text_id = "Halo, apa kabar? Saya sedang belajar membuat LLM."
text_en = "Hello! I'm building a language model from scratch."
text_code = "def hitung(a, b):\n    return a + b"

ids_id = tok.encode(text_id, add_bos=True, add_eos=True, lang="id")
ids_en = tok.encode(text_en, add_bos=True, add_eos=True, lang="en")
ids_code = tok.encode(text_code, lang="code")

print(f"[ID]   tokens={len(ids_id)} | {ids_id[:8]}...")
print(f"[EN]   tokens={len(ids_en)} | {ids_en[:8]}...")
print(f"[CODE] tokens={len(ids_code)} | {ids_code[:8]}...")

# ── Test 2: Decode ───────────────────────────────────────
decoded = tok.decode(ids_id, skip_special_tokens=True)
assert decoded == text_id, f"Mismatch: {decoded}"
print(f"✅ Decode OK: '{decoded[:40]}...'")

# ── Test 3: Padding ──────────────────────────────────────
batch_texts = [text_id, text_en, text_code]
batch_ids = tok.encode_batch(batch_texts, add_eos=True)
padded, masks = tok.pad_batch(batch_ids, max_length=64)

print(f"\n📐 Padded batch shape: {len(padded)} x {len(padded[0])}")
for i, (p, m) in enumerate(zip(padded, masks)):
    real = sum(m)
    print(f"  Seq {i}: {real} real tokens, {64 - real} padding")

# ── Test 4: Token count ──────────────────────────────────
long_text = "Ini teks panjang untuk ngetes. " * 100
count = tok.count_tokens(long_text)
print(f"\n🔢 '{long_text[:30]}...' → {count} tokens")