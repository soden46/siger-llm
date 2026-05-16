# 🤖 AGENTS.md — Panduan untuk AI Agents & LLM Tools

Dokumen ini ditujukan untuk **AI agents, coding assistants, dan LLM tools** (seperti Claude, Cursor, Copilot, dsb) yang bekerja dengan codebase MambaLM.

---

## Project Summary

MambaLM adalah implementasi Large Language Model berbasis **State Space Model (Mamba)** yang dioptimasi untuk CPU-only deployment. Dibangun dengan PyTorch, menggunakan tiktoken untuk tokenisasi, ONNX Runtime untuk inference, dan FastAPI untuk serving.

**Stack utama:**
- Language   : Python 3.11
- Framework  : PyTorch 2.2+
- Tokenizer  : tiktoken (cl100k_base)
- Inference  : ONNX Runtime
- API        : FastAPI + uvicorn
- Fine-tuning: LoRA (custom implementation)

---

## Struktur Codebase

```
mamba-llm/
├── config/model_config.py          # MambaConfig dataclass — BACA INI DULU
├── model/
│   ├── ssm_core.py                 # Core SSM math (A, B, C, D matrices)
│   ├── ssm_block.py                # Satu SSM block lengkap
│   └── mamba_model.py              # Full model: embed → blocks → LM head
├── tokenizer/tokenizer.py          # MultilingualTokenizer (tiktoken wrapper)
├── tokenizer/special_tokens.py     # Semua special token dan ID-nya
├── training/trainer.py             # Main training loop
├── training/optimizer.py           # AdamW + CosineScheduler
├── lora/
│   ├── layer.py                    # LoRALinear — core LoRA math
│   └── model.py                    # LoRAModel — inject/save/load/merge
├── optimization/
│   ├── quantization/quantize.py    # INT8/INT4 quantization
│   └── onnx/export.py              # ONNX export + ONNXGenerator
├── inference/
│   ├── generator.py                # Generator + stream()
│   ├── sampler.py                  # Sampler.sample() — temperature/top-k/top-p
│   └── chat.py                     # ChatSession
├── evaluation/runner.py            # EvaluationRunner — jalanin semua eval
└── deploy.py                       # Entry point deployment
```

---

## Konvensi Kode

### Naming

```python
# Kelas     : PascalCase
class MambaLM, LoRAModel, ChatSession

# Fungsi    : snake_case
def build_optimizer(), def encode_batch()

# Konstanta : UPPER_SNAKE
SPECIAL_TOKENS = {...}
IGNORE_INDEX   = -100

# Config    : dataclass dengan field bernama jelas
@dataclass
class MambaConfig:
    vocab_size: int = 100277
    d_model:    int = 512
```

### Type Hints

Selalu gunakan type hints untuk fungsi publik:

```python
def encode(
    self,
    text: str,
    add_bos: bool = False,
    lang: Optional[str] = None,
) -> List[int]:
```

### Tensor Shape Conventions

```python
# Selalu dokumentasikan shape tensor sebagai komentar:
x: torch.Tensor  # (B, L, d_model)  — B=batch, L=seq_len
logits: torch.Tensor  # (B, L, vocab_size)
h: torch.Tensor  # (B, d_inner, d_state)  — SSM hidden state
```

---

## Pola Umum yang Sering Dipakai

### 1. Forward Pass

```python
# Selalu: model(input_ids) → (logits, loss)
logits, loss = model(input_ids, targets=labels)

# Inference only (tanpa targets):
logits, _ = model(input_ids)
next_logits = logits[0, -1, :]  # ambil token terakhir
```

### 2. Encode/Decode

```python
# Encode selalu return List[int]
ids = tokenizer.encode(text, add_bos=True, add_eos=True, lang="id")

# Decode selalu return str
text = tokenizer.decode(ids, skip_special_tokens=True)
```

### 3. Checkpoint

```python
# Save
torch.save(model.state_dict(), path)

# Load (efficient)
model.load_state_dict(
    torch.load(path, map_location="cpu", weights_only=True),
    assign=True
)
```

### 4. LoRA Pattern

```python
# Inject → Train → Save → Load → Merge
lora_model = LoRAModel(base_model, config)   # inject
trainer.train(dataset)                         # train
lora_model.save_lora("lora.pt")              # save (kecil!)
lora_model.load_lora("lora.pt")              # load
merged = lora_model.merge_and_export("merged.pt")  # merge
```

---

## Constraints & Gotchas

### Memory

- VPS target: 2 core, 4GB RAM. **Selalu** pertimbangkan RAM usage.
- Batch size default (8) mungkin perlu diturunkan ke 2-4 di VPS
- Gunakan `load_model_efficient()` bukan `torch.load()` langsung
- `max_seq_len` di inference sebaiknya tidak lebih dari 1024

### Tensor Device

```python
# Model dan input harus di device yang sama
x = x.to(self.device)  # selalu pindahkan input ke device model

# Jangan lupa .cpu() sebelum numpy atau json serialization
value = tensor.cpu().numpy()
```

### ONNX

- ONNX session tidak support `torch.Tensor` input — gunakan `np.ndarray`
- Dynamic axes (`batch` dan `seq_len`) sudah di-set saat export
- Jangan gunakan ONNX session bersamaan dari multiple thread tanpa lock

### LoRA

- `lora_A` di-init dengan kaiming uniform, `lora_B` dengan zeros
- Jangan freeze LoRA params saat training LoRA
- `model.parameters()` setelah inject LoRA akan include frozen base params
- Untuk optimizer, filter: `[p for p in model.parameters() if p.requires_grad]`

### Tokenizer

- Token ID `100257-100270` adalah special tokens (bukan vocabulary biasa)
- Selalu gunakan `skip_special_tokens=True` saat decode untuk output bersih
- `count_tokens()` lebih efisien dari `len(encode())` untuk hitung panjang

---

## Common Tasks untuk Agent

### Task: Tambah Layer Baru ke Model

```python
# 1. Edit config/model_config.py — tambah field baru
# 2. Edit model/ssm_block.py — implementasi layer
# 3. Edit model/mamba_model.py — integrasikan
# 4. Pastikan forward() signature tetap: (input_ids, targets=None) → (logits, loss)
# 5. Test: python -c "from model.mamba_model import MambaLM; ..."
```

### Task: Tambah Special Token Baru

```python
# 1. Edit tokenizer/special_tokens.py
SPECIAL_TOKENS["<|new_token|>"] = 100271  # increment dari yang terakhir
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}  # rebuild reverse map

# 2. Rebuild tokenizer encoder di tokenizer/tokenizer.py (_build_encoder)
# 3. Update vocab_size di config jika perlu
```

### Task: Tambah Dataset Format Baru

```python
# Di lora/dataset.py, tambah case baru di format_instruction():
elif "new_dataset" in dataset_name:
    instruction = example.get("instruction", "")
    response    = example.get("response", "")
    return (
        f"<|system|>...<|end_turn|>\n"
        f"<|user|>{instruction}<|end_turn|>\n"
        f"<|assistant|>{response}<|end_turn|>"
    )
```

### Task: Tambah Benchmark Baru

```python
# Di evaluation/benchmarks.py atau evaluation/indo_eval.py:
# 1. Tambah method baru
def evaluate_new_benchmark(self, n_samples=200) -> Dict:
    dataset = load_dataset("dataset/name", split="test")
    # ... scoring logic ...
    return {"accuracy": ..., "total": ...}

# 2. Register di evaluation/runner.py
results["new_bench"] = mc_eval.evaluate_new_benchmark(n_samples)
```

---

## Testing

```bash
# Unit test tokenizer
python -m tokenizer.tests.test_tokenizer

# Smoke test model
python -c "
import torch
from config.model_config import MambaConfig
from model.mamba_model   import MambaLM

config = MambaConfig(vocab_size=1000, d_model=64, n_layers=2)
model  = MambaLM(config)
x      = torch.randint(0, 1000, (2, 32))
logits, _ = model(x)
assert logits.shape == (2, 32, 1000), f'Wrong shape: {logits.shape}'
print('✅ All tests passed')
"

# Full test suite
pytest tests/ -v
```

---

## Urutan Modifikasi yang Aman

Ketika membuat perubahan besar, ikuti urutan ini agar tidak ada yang break:

```
1. config/model_config.py     ← tambah config baru
2. model/ssm_core.py          ← perubahan math SSM
3. model/ssm_block.py         ← perubahan block structure
4. model/mamba_model.py       ← integrasi di level model
5. tokenizer/special_tokens.py ← kalau ada token baru
6. tokenizer/tokenizer.py     ← kalau ada logic tokenizer baru
7. training/dataset.py        ← kalau ada format data baru
8. lora/layer.py + model.py   ← kalau ada perubahan LoRA
9. inference/generator.py     ← kalau ada perubahan generation
10. evaluation/               ← tambah metric baru
```

---

## File yang TIDAK Boleh Dimodifikasi Tanpa Review

```
tokenizer/special_tokens.py   ← perubahan ID token merusak checkpoint
config/model_config.py         ← perubahan default merusak backward compat
model/ssm_core.py              ← math SSM harus tetap konsisten
lora/layer.py                  ← init strategy (kaiming A, zero B) krusial
```