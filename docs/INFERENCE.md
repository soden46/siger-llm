# 🚀 INFERENCE.md — Panduan Inference & Generation

## Overview

Inference adalah proses menggunakan model yang sudah ditraining untuk menghasilkan teks baru. SIGERLM mendukung tiga mode:

1. **Single generation** — satu prompt, satu output
2. **Streaming** — output muncul token per token (seperti ChatGPT)
3. **Chat session** — multi-turn conversation dengan memory

---

## Cara Kerja Autoregressive Generation

```
Prompt: "Halo, siapa"

Step 1: encode prompt → [101, 202, 303]
Step 2: forward pass  → logits untuk semua token
Step 3: ambil logits token terakhir → sample → "kamu" (token 404)
Step 4: append 404 → [101, 202, 303, 404]
Step 5: forward pass lagi → sample → "?" (token 505)
Step 6: [101, 202, 303, 404, 505] → decode → "Halo, siapa kamu?"
... ulangi sampai max_new_tokens atau EOS
```

Tanpa SSM cache: tiap step recompute seluruh sequence → O(n²) total
Dengan SSM cache: tiap step hanya compute 1 token → O(n) total

---

## Sampling Strategies

### 1. Greedy (Temperature = 0)

Selalu pilih token dengan probabilitas tertinggi. Deterministic, cenderung repetitif.

```python
gen.generate(prompt, temperature=0.0)  # do_sample=False
```

### 2. Temperature Sampling

```python
# Temperature < 1.0 → lebih fokus, kurang kreatif
gen.generate(prompt, temperature=0.3)

# Temperature = 1.0 → distribusi natural
gen.generate(prompt, temperature=1.0)

# Temperature > 1.0 → lebih random, lebih kreatif
gen.generate(prompt, temperature=1.3)
```

```
Distribusi logits sebelum softmax:
Token A: 5.0
Token B: 3.0
Token C: 1.0

Temp=0.5 → A:99%, B:1%, C:~0%   (fokus)
Temp=1.0 → A:87%, B:11%, C:2%   (normal)
Temp=2.0 → A:62%, B:29%, C:9%   (random)
```

### 3. Top-K Sampling

Filter hanya K token terbaik, sample dari sana.

```python
gen.generate(prompt, top_k=50)  # hanya 50 token terbaik
```

### 4. Top-P (Nucleus) Sampling

Filter token sampai cumulative probability >= p. Lebih adaptif dari Top-K.

```python
gen.generate(prompt, top_p=0.9)
# → ambil token sampai cumulative prob >= 90%
# Misal: A(50%) + B(25%) + C(15%) = 90% → hanya A,B,C yang dipertimbangkan
```

### 5. Repetition Penalty

Kurangi probabilitas token yang sudah muncul sebelumnya.

```python
gen.generate(prompt, repetition_penalty=1.3)
# penalty > 1.0 → makin jarang ngulang kata yang sama
```

### Kombinasi Recommended

```python
# General chat (balanced)
gen.generate(prompt,
    temperature=0.8,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.15,
)

# Factual / Q&A (focused)
gen.generate(prompt,
    temperature=0.3,
    top_k=20,
    top_p=0.85,
    repetition_penalty=1.1,
)

# Creative writing (random)
gen.generate(prompt,
    temperature=1.1,
    top_k=100,
    top_p=0.95,
    repetition_penalty=1.05,
)

# Coding (deterministic tapi tidak terlalu rigid)
gen.generate(prompt,
    temperature=0.4,
    top_k=30,
    top_p=0.9,
    repetition_penalty=1.2,
)
```

---

## Usage — Generator

```python
from model.siger_model    import SIGERLM
from tokenizer.tokenizer  import MultilingualTokenizer
from inference.generator  import Generator
from config.model_config  import SigerConfig
import torch

# Load model
config = SigerConfig(vocab_size=100277, d_model=512, n_layers=12)
model  = SIGERLM(config)
model.load_state_dict(torch.load("./checkpoints/best_model.pt"))

tok = MultilingualTokenizer()
gen = Generator(model, tok)

# Single generation
output = gen.generate(
    "Jelaskan apa itu machine learning:",
    max_new_tokens     = 200,
    temperature        = 0.8,
    top_k              = 50,
    top_p              = 0.9,
    repetition_penalty = 1.15,
    lang               = "id",
)
print(output)

# Streaming (print token by token)
for token in gen.stream("Ceritakan tentang Jakarta:", max_new_tokens=100):
    print(token, end="", flush=True)
print()

# Batch generation
outputs = gen.generate_batch(
    ["Apa itu AI?", "What is Python?", "def hello():"],
    max_new_tokens=50,
)
```

---

## Usage — Chat Session

```python
from inference.chat import ChatSession

chat = ChatSession(
    generator         = gen,
    system_prompt     = "Kamu adalah asisten coding yang ahli Laravel dan Python.",
    max_history       = 10,
    max_context_tokens = 1024,
)

# Multi-turn conversation
response1 = chat.chat("Apa itu LoRA?")
print("Bot:", response1)

response2 = chat.chat("Bagaimana cara implementasinya?")  # ingat context sebelumnya
print("Bot:", response2)

# Streaming mode
response3 = chat.chat("Berikan contoh kodenya", stream=True)

# Reset session
chat.reset()
```

---

## REST API

### Jalanin Server

```bash
python deploy.py
# → Server berjalan di http://0.0.0.0:8000
# → Docs di http://localhost:8000/docs
```

### Endpoints

#### `GET /health`
```bash
curl http://localhost:8000/health
# → {"status": "ok", "model_loaded": true}
```

#### `POST /generate`
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Apa ibu kota Indonesia?",
    "max_new_tokens": 100,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "lang": "id"
  }'

# Response:
# {
#   "text": "Ibu kota Indonesia adalah Jakarta...",
#   "token_count": 24
# }
```

#### `POST /generate` (Streaming)
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Halo!", "stream": true}'

# Response (SSE):
# data: {"token": "Halo"}
# data: {"token": "!"}
# data: {"token": " Apa"}
# ...
# data: [DONE]
```

#### `POST /chat`
```bash
# Buat session baru
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "Siapa kamu?",
    "temperature": 0.8
  }'

# Lanjutkan conversation (session_id sama)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123",
    "message": "Apa yang bisa kamu bantu?"
  }'
```

#### `DELETE /chat/{session_id}`
```bash
curl -X DELETE http://localhost:8000/chat/user123
# → {"status": "reset"}
```

### Contoh Client Python

```python
import requests
import json

BASE = "http://localhost:8000"

# Generate
resp = requests.post(f"{BASE}/generate", json={
    "prompt": "Jelaskan Laravel dalam 3 kalimat:",
    "max_new_tokens": 150,
    "temperature": 0.7,
})
print(resp.json()["text"])

# Streaming dengan SSE
import sseclient

resp = requests.post(f"{BASE}/generate",
    json={"prompt": "Hello!", "stream": True},
    stream=True,
    headers={"Accept": "text/event-stream"},
)
client = sseclient.SSEClient(resp)
for event in client.events():
    if event.data == "[DONE]":
        break
    token = json.loads(event.data)["token"]
    print(token, end="", flush=True)
```

---

## Performance Tips

### Kurangi Latency First Token

```python
# Pre-warm model dengan dummy forward pass saat startup
dummy = torch.zeros(1, 1, dtype=torch.long)
with torch.no_grad():
    model(dummy)
print("Model warmed up ✅")
```

### Batching untuk Throughput

```python
# Kalau ada banyak request masuk, batch mereka:
outputs = gen.generate_batch(
    prompts       = ["prompt1", "prompt2", "prompt3"],
    max_new_tokens = 100,
)
# Lebih efisien daripada generate satu per satu
```

### Context Length Trade-off

```python
# Makin panjang context = makin lambat (linear dengan SSM)
# Rekomendasi: batasi max_context_tokens di ChatSession

chat = ChatSession(
    generator          = gen,
    max_context_tokens = 512,  # untuk VPS, jangan lebih dari 1024
)
```