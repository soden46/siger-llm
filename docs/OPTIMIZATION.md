# ⚡ OPTIMIZATION.md — Optimasi untuk CPU/VPS

## Overview

Optimasi SIGERLM difokuskan untuk deployment di **CPU-only VPS** (2 core, 4GB RAM). Ada tiga teknik utama yang dikombinasikan:

```
Raw PyTorch FP32
      │
      ▼  INT8 Quantization (4× lebih kecil)
INT8 Model
      │
      ▼  ONNX Export (2-3× lebih cepat)
ONNX Runtime
      │
      ▼  CPU Thread Tuning
Optimized Inference ✅
```

---

## 1. Quantization

Quantization mengubah presisi numerik parameter model:

```
FP32 (float32) : 4 byte per value  ← default training
FP16 (float16) : 2 byte per value  ← GPU inference
INT8           : 1 byte per value  ← CPU inference optimal
INT4           : 0.5 byte per value ← paling kecil
```

### Perbandingan

| Mode | Size | RAM | Speed | Quality |
|---|---|---|---|---|
| FP32 | 260 MB | ~3.8 GB ⚠️ | 1× | 100% |
| INT8 Dynamic | 70 MB | ~1.0 GB ✅ | ~2× | ~97% |
| INT8 Static | 65 MB | ~0.9 GB ✅ | ~2.5× | ~96% |
| INT4 (NF4) | 35 MB | ~0.5 GB ✅ | ~3× | ~92% |

### INT8 Dynamic (Paling Mudah, Recommended)

```python
from optimization.quantization.quantize import ModelQuantizer

quantizer  = ModelQuantizer(model)
model_int8 = quantizer.quantize_int8_dynamic()

# Output:
# ✅ INT8 Dynamic done!
#    Original : 260.3 MB
#    Quantized: 68.2 MB
#    Reduction: 3.8x smaller
```

Cara kerja:
- Weight di-quantize ke INT8 saat load
- Activation di-quantize saat runtime (dynamic)
- Tidak perlu calibration data
- Langsung bisa digunakan

### INT8 Static (Lebih Cepat, Perlu Calibration)

```python
# Perlu calibration data (beberapa batch dari training set)
from torch.utils.data import DataLoader

calib_loader = DataLoader(dataset, batch_size=4, shuffle=True)
model_static = quantizer.quantize_int8_static(calib_loader)
```

Cara kerja:
- Observer dipasang di semua layer
- Jalankan calibration data → observer catat range activation
- Scale factor di-freeze berdasarkan range observed
- Inference lebih cepat karena tidak ada dynamic quantization overhead

### INT4 NF4 (Terkecil, via bitsandbytes)

```python
# Install: pip install bitsandbytes
model_int4 = quantizer.quantize_int4_bnb()

# NF4 = NormalFloat4, distribusi optimal untuk weight LLM
# compress_statistics=True → kompresi ekstra ~10%
```

---

## 2. ONNX Export & Runtime

ONNX Runtime adalah inference engine yang lebih optimal dari PyTorch untuk CPU:

```
PyTorch (CPU)   : general purpose → tidak optimal
ONNX Runtime    : dioptimasi khusus untuk inference
Speedup         : 2-3× lebih cepat
```

### Export ke ONNX

```python
from optimization.onnx.export import ONNXExporter

exporter  = ONNXExporter(model_int8)

# Step 1: Export
onnx_path = exporter.export(seq_len=512, batch_size=1)

# Step 2: Optimize graph
opt_path  = exporter.optimize_onnx(onnx_path)
# → Fold constants, fuse operators, eliminate redundancies

# Step 3: Build session
session   = exporter.build_session(opt_path)
```

### ONNX Session Config untuk 2-Core VPS

```python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2   # thread per operator (= jumlah core)
opts.inter_op_num_threads = 1   # thread antar operator (sequential)
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# Max optimization
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Memory reuse
opts.enable_mem_pattern   = True
opts.enable_mem_reuse     = True
opts.enable_cpu_mem_arena = True
```

### Gunakan ONNXGenerator sebagai Drop-in

```python
from optimization.onnx.export import ONNXGenerator

# Gantikan Generator biasa dengan ONNXGenerator
gen = ONNXGenerator(session, tokenizer)

# Interface sama persis
output = gen.generate("Halo!", max_new_tokens=100)
```

---

## 3. CPU Thread Configuration

### Setup Optimal untuk 2-Core VPS

```python
from optimization.cpu.threading import configure_cpu

# Panggil SEBELUM import model apapun
configure_cpu(n_cores=2)

# Yang diset:
# torch.set_num_threads(2)
# OMP_NUM_THREADS=2
# MKL_NUM_THREADS=2
# OPENBLAS_NUM_THREADS=2
```

### Kenapa Harus Diset Manual?

PyTorch secara default menggunakan semua core yang tersedia. Di VPS shared, ini bisa menyebabkan throttling dari provider. Setting manual ke jumlah core yang dialokasikan = performa lebih stabil.

---

## 4. Memory Management

### Load Model Efisien (Hemat RAM)

```python
from optimization.cpu.memory import load_model_efficient

# Cara biasa (double RAM saat loading):
model = SigerLM(config)
model.load_state_dict(torch.load("model.pt"))  # ← RAM = 2× size model

# Cara efisien (single RAM):
model = load_model_efficient(SigerLM, config, "model.pt")
# Teknik:
# 1. Init model dengan meta device (no memory allocation)
# 2. Load state dict langsung ke CPU
# 3. assign=True untuk skip copy
```

### Monitor RAM

```python
from optimization.cpu.memory import MemoryManager

mem = MemoryManager()

# Check current usage
mem.check("before inference")
# → 🧠 RAM [before inference]: used=1.23GB | avail=2.45GB

# Track per operasi
with mem.track("onnx_export"):
    session = exporter.build_session(opt_path)
# → 📊 [onnx_export] RAM delta: +0.12GB (total: 1.35GB)

# Clear cache
mem.clear()  # gc.collect() + empty CUDA cache
```

### Swap File sebagai Safety Net

```bash
# Buat swap 2GB di VPS (sekali setup)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Swap = slowdown (disk I/O), tapi lebih baik dari OOM crash
# Dengan swap: model masih jalan walau RAM penuh
```

---

## 5. SSM Cache untuk Fast Generation

Tanpa cache, setiap token yang di-generate membutuhkan forward pass seluruh sequence:

```
Token 1: forward([t1])           → 1 step
Token 2: forward([t1, t2])       → 2 steps
Token 3: forward([t1, t2, t3])   → 3 steps
Total: 1+2+3+...+N = O(N²)
```

Dengan SSM cache:

```
Prefill: forward([t1, t2, ..., prompt_len])  → 1 pass, save h_cache
Token 1: forward([t_new], cache=h_cache)     → 1 step
Token 2: forward([t_new], cache=h_cache)     → 1 step
Total: prompt_len + N = O(N) ✅
```

```python
from optimization.kvcache import SSMCache, CachedSSMBlock

# Cache otomatis digunakan di CachedSSMBlock
cache = SSMCache()

# Prefill phase
output, cache = model(prompt_tokens, cache=cache, use_cache=True)

# Decode phase (tiap token hanya 1 step)
for _ in range(max_new_tokens):
    output, cache = model(next_token, cache=cache, use_cache=True)
```

---

## 6. Benchmark

```bash
# Jalanin benchmark lengkap
python -c "
from optimization.benchmark import benchmark
from optimization.onnx.export import ONNXGenerator
# ... setup generator ...
benchmark(gen, prompt='Halo, apa kabar?', n_tokens=50, n_runs=3)
"

# Output:
# Run 1: 12.3 tok/s | 'Halo! Kabar saya baik, terima kasih...'
# Run 2: 11.8 tok/s | 'Halo! Saya baik-baik saja...'
# Run 3: 12.1 tok/s | 'Halo! Apa kabar juga...'
#
# ════════════════════════════════════════
# ⚡ 12.1 tokens/sec
# ⏱️  4132ms per 50 tokens
# 🧠 RAM: 1243MB
# ════════════════════════════════════════
```

---

## 7. Deploy Pipeline Lengkap

```python
# deploy.py — urutan yang benar

# Step 1: Configure CPU (HARUS PERTAMA)
from optimization.cpu.threading import configure_cpu
configure_cpu(n_cores=2)

# Step 2: Load model efisien
from optimization.cpu.memory import load_model_efficient
model = load_model_efficient(SigerLM, config, "./checkpoints/best_model.pt")

# Step 3: Quantize
from optimization.quantization.quantize import ModelQuantizer
model_q = ModelQuantizer(model).quantize_int8_dynamic()

# Step 4: Export ONNX
from optimization.onnx.export import ONNXExporter, ONNXGenerator
exporter  = ONNXExporter(model_q)
onnx_path = exporter.export()
opt_path  = exporter.optimize_onnx(onnx_path)
session   = exporter.build_session(opt_path)

# Step 5: Build generator
tok = MultilingualTokenizer()
gen = ONNXGenerator(session, tok)

# Step 6: Benchmark
from optimization.benchmark import benchmark
benchmark(gen)

# Step 7: Serve API
from inference.api import app, init_api
init_api(gen)
uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
```

---

## Expected Performance di DewaCloud VPS

```
Spec: 2 vCPU shared, 4GB RAM, Ubuntu 22.04

                    Size      RAM Used    Speed
──────────────────────────────────────────────
Raw FP32            260 MB    ~3.8 GB ⚠️  ~1-2 tok/s
After INT8          70 MB     ~1.0 GB ✅  ~4-6 tok/s
After ONNX          70 MB     ~1.2 GB ✅  ~8-15 tok/s
After Cache         70 MB     ~1.3 GB ✅  ~12-20 tok/s
With Swap (safety)  70 MB     ~1.3 GB ✅  sama (no swap hit)

Concurrent requests:
  1 user   : ~12 tok/s (smooth)
  2 users  : ~6 tok/s each
  3+ users : queue recommended
```