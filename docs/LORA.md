# 🔧 LORA.md — Fine-tuning dengan LoRA

## Apa itu LoRA?

**LoRA (Low-Rank Adaptation)** adalah teknik fine-tuning yang efisien secara parameter. Alih-alih mengubah semua weight model (mahal), LoRA menambahkan **matriks kecil tambahan** yang ditraining sementara weight asli di-freeze.

### Konsep Matematika

```
Normal fine-tuning:
  W_new = W_original + ΔW
  ΔW punya dimensi sama dengan W → 262,144 params (untuk 512×512)

LoRA:
  W_new = W_original + (B × A) × (α/r)
  A: (r × d_in)   → 8 × 512   = 4,096 params
  B: (d_out × r)  → 512 × 8   = 4,096 params
  Total LoRA      → 8,192 params  (32× lebih sedikit!)

Inisialisasi:
  A → kaiming uniform (non-zero dari awal, ada gradient)
  B → zeros (output LoRA = 0 di awal, sama kayak pretrained)
```

### Kenapa B di-init ke Zeros?

Supaya di awal training, `W_new = W_original + 0 = W_original`. Model mulai dari pretrained checkpoint tanpa gangguan, lalu LoRA perlahan-lahan belajar adjustment yang dibutuhkan.

---

## Perbandingan: Full Fine-tune vs LoRA

```
                    Full Fine-tune    LoRA (r=8)      LoRA (r=16)
────────────────────────────────────────────────────────────────
Trainable params  : 66,000,000       508,000          1,016,000
Params percentage : 100%             0.77%            1.54%
RAM untuk training: ~3.8 GB ⚠️       ~1.2 GB ✅        ~1.4 GB ✅
Training speed    : ~0.5 tok/s       ~3.0 tok/s       ~2.5 tok/s
LoRA file size    : N/A              ~4 MB             ~8 MB
Quality vs full   : 100%            ~95-98%           ~97-99%
Waktu 5k steps    : ~8 jam           ~2.5 jam         ~3 jam
```

---

## Konfigurasi LoRA

```python
# lora/config.py

LoRAConfig(
    # ── Core LoRA ────────────────────────────────────────
    rank    = 8,       # r: dimensi low-rank
                       # Nilai umum: 4, 8, 16, 32, 64
                       # Makin besar = makin expressive = makin berat
    alpha   = 16.0,    # scaling factor
                       # scaling = alpha/rank = 2.0
                       # Biasanya alpha = 2×rank
    dropout = 0.05,    # regularization

    # ── Target Layers ────────────────────────────────────
    target_modules = [
        "in_proj",    # input projection (SSM block)
        "out_proj",   # output projection
        "x_proj",     # SSM x projection
        "dt_proj",    # delta time projection
    ],
    # Catatan: makin banyak layer = makin banyak LoRA params

    # ── Training ─────────────────────────────────────────
    learning_rate  = 2e-4,   # bisa lebih tinggi dari full finetune
    max_steps      = 5_000,
    batch_size     = 4,
    grad_accum     = 8,      # effective batch = 32
    warmup_steps   = 100,
    max_seq_len    = 512,
    weight_decay   = 0.01,

    # ── Dataset ──────────────────────────────────────────
    dataset_name   = "HuggingFaceH4/ultrachat_200k",
    dataset_split  = "train_sft",
    max_samples    = 20_000,

    # ── Save ─────────────────────────────────────────────
    save_dir       = "./checkpoints/lora",
    save_every     = 500,
)
```

### Panduan Memilih Rank

| Rank | Params | Use Case |
|---|---|---|
| 4 | ~0.4% | Fine-tune ringan, domain kecil |
| **8** | **~0.8%** | **Sweet spot, recommended** |
| 16 | ~1.5% | Domain kompleks, dataset besar |
| 32 | ~3.0% | Butuh perubahan besar |
| 64 | ~6.0% | Hampir seperti full fine-tune |

---

## Dataset Format

LoRA menggunakan **instruction fine-tuning format**. Dataset harus diformat sebagai pasangan instruksi-respons.

### Format Chat (Recommended)

```
<|system|>Kamu adalah asisten yang helpful.<|end_turn|>
<|user|>Apa itu machine learning?<|end_turn|>
<|assistant|>Machine learning adalah cabang AI yang...<|end_turn|>
```

### Dataset Publik yang Direkomendasikan

| Dataset | Bahasa | Size | Kualitas | Link |
|---|---|---|---|---|
| **UltraChat 200k** | EN | 200k | ⭐⭐⭐⭐⭐ | HuggingFaceH4/ultrachat_200k |
| Indonesian Alpaca | ID | ~52k | ⭐⭐⭐⭐ | indonlp/indonesian-alpaca |
| OpenAssistant v2 | Multi | ~161k | ⭐⭐⭐⭐⭐ | OpenAssistant/oasst2 |
| Dolly 15k | EN | 15k | ⭐⭐⭐⭐ | databricks/databricks-dolly-15k |
| FLAN | Multi | 1.8M | ⭐⭐⭐⭐ | Muennighoff/flan |

### Loss Masking — Kunci Instruction Fine-tuning

```python
# Loss HANYA dihitung di bagian <|assistant|>
# Bagian <|system|> dan <|user|> di-mask dengan -100

Input:  [sys_tok, ...sys..., end_tok, user_tok, ...question..., end_tok, asst_tok, ...answer..., end_tok]
Labels: [-100,   ...-100.., -100,    -100,     ...-100......., -100,    real_id,  ...real_id.., real_id]
                                                                          ↑ hanya ini yang dihitung loss
```

Tanpa loss masking, model belajar mengulang pertanyaan — bukan menjawab. Loss masking memastikan model fokus belajar **cara merespons**.

---

## Cara Kerja Inject LoRA

```python
# Sebelum inject:
# model.layers[0].ssm_block.in_proj = nn.Linear(512, 1024)

# Setelah inject:
# model.layers[0].ssm_block.in_proj = LoRALinear(
#     original = nn.Linear(512, 1024),  ← frozen
#     lora_A   = Parameter(8×512),      ← trainable
#     lora_B   = Parameter(1024×8),     ← trainable
# )

lora_model = LoRAModel(base_model, lora_config)

# Output summary:
# ══════════════════════════════════════════════════
# LoRA Model Summary
# ══════════════════════════════════════════════════
# Base model params :   66,000,000
# LoRA params       :      508,000  (0.77%)
# Total params      :   66,508,000
# LoRA layers       :   48
# Rank / Alpha      :   8 / 16.0
# ══════════════════════════════════════════════════
```

---

## Jalanin LoRA Fine-tuning

```bash
# Basic
python run_lora.py

# Monitor training
tail -f training_lora.log

# Background di VPS
nohup python run_lora.py > training_lora.log 2>&1 &
```

### Output yang Diharapkan

```
✅ Tokenizer ready | vocab_size=100277
📦 Loading base model...
📥 Loading dataset: HuggingFaceH4/ultrachat_200k
📊 Processing 20,000 examples...
✅ Dataset ready: 18,432 valid examples

══════════════════════════════════════════════════
LoRA Model Summary
══════════════════════════════════════════════════
Base model params :   66,000,000
LoRA params       :      508,000  (0.77%)
══════════════════════════════════════════════════

🚀 LoRA Training
   Dataset   : 18,432 examples
   Max steps : 5,000
   Eff. batch: 32

step=     10 | loss=2.3421 | avg_loss=2.45 | lr=2.0e-05
step=     20 | loss=2.1823 | avg_loss=2.28 | lr=4.0e-05
...
step=  5,000 | loss=1.2341 | avg_loss=1.31 | lr=2.0e-05
✅ LoRA training complete!
🔀 Merging LoRA into base model...
✅ Merged model saved: ./checkpoints/lora/model_merged.pt
🎉 Done!
```

---

## Save, Load & Merge

### Save LoRA (hanya adapter — kecil!)

```python
lora_model.save_lora("./checkpoints/lora/lora_v1.pt")
# → File ~4MB (bukan ~260MB full model)
```

### Load LoRA ke Base Model Baru

```python
# Berguna untuk: switch antar LoRA adapters, deploy multi-task
base_model  = load_model(...)
lora_model  = LoRAModel(base_model, lora_config)
lora_model.load_lora("./checkpoints/lora/lora_v1.pt")
```

### Merge LoRA → Full Model (untuk deployment)

```python
# Setelah merge, tidak ada overhead LoRA di inference
merged = lora_model.merge_and_export("./checkpoints/model_v1_merged.pt")
# Hasilnya: full model dengan LoRA sudah ter-baked ke weights
```

---

## Multi-LoRA: Beberapa Adapter untuk Tugas Berbeda

```python
# Buat beberapa LoRA adapter untuk task berbeda:
# lora_chat.pt    → general conversation
# lora_code.pt    → coding assistant
# lora_indo.pt    → bahasa indonesia khusus

# Di inference, swap adapter sesuai kebutuhan:
def load_adapter(lora_model, adapter_path):
    lora_model.load_lora(adapter_path)
    return lora_model

# Chat request → load lora_chat.pt
# Code request → load lora_code.pt
```

---

## Tips Fine-tuning

```python
# Dataset quality > quantity
# 1,000 contoh berkualitas tinggi > 100,000 contoh noise

# Learning rate LoRA bisa lebih tinggi dari pretraining
# Pretraining: 3e-4
# LoRA:        2e-4 (sedikit lebih rendah, model udah pretrained)

# Kalau loss turun terlalu cepat → overfitting
# → Tambah dropout (0.05 → 0.1)
# → Kurangi max_steps
# → Tambah data augmentation

# Kalau loss tidak turun → underfitting
# → Naikkan rank (8 → 16)
# → Tambah target_modules
# → Naikkan learning rate
```