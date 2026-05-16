# 📊 EVALUATION.md — Evaluasi Model SIGERLM

## Overview

Evaluasi LLM terdiri dari tiga dimensi:

```
1. Intrinsic  → Seberapa baik model prediksi next token
   Metric: Perplexity (PPL)

2. Extrinsic  → Seberapa baik model di task nyata
   Metric: MMLU, ARC-Challenge, IndoNLI, IndoQA

3. Human-like → Seberapa mirip output model dengan referensi
   Metric: BLEU, ROUGE, Diversity
```

---

## 1. Perplexity (PPL)

### Apa itu Perplexity?

```
PPL = exp(average negative log likelihood)
    = exp(- (1/N) × Σ log P(token_i | context))
```

Intuisi: PPL = berapa banyak token yang "equally likely" menurut model.

```
PPL = 1     → model 100% yakin (tidak mungkin)
PPL = 10    → model punya ~10 pilihan yang roughly equally likely → bagus
PPL = 100   → model punya ~100 pilihan → bingung
PPL = 50000 → model hampir random
```

### Interpretasi PPL

```
PPL Range     Grade          Interpretasi
────────────────────────────────────────────────────
< 15          🟢 Excellent   Model sangat percaya diri
15 - 30       🟢 Good        Kompresi bahasa yang baik
30 - 50       🟡 Acceptable  Masih OK untuk model kecil
50 - 100      🟠 Poor        Perlu lebih banyak training
> 100         🔴 Very Poor   Model belum belajar banyak
```

### Referensi PPL Model Terkenal

```
GPT-2 Small  (117M, WikiText-103): PPL ~29
GPT-2 Medium (345M)              : PPL ~26
GPT-2 Large  (774M)              : PPL ~22
GPT-3 (175B)                     : PPL ~20
LLaMA-7B                         : PPL ~12
SIGERLM (66M, target)         : PPL < 50 ✅
```

### Cara Hitung

```python
from evaluation.perplexity import PerplexityEvaluator

eval_texts = [
    "Pemerintah Indonesia mengumumkan kebijakan baru.",
    "The quick brown fox jumps over the lazy dog.",
    "def calculate(a, b): return a + b",
]

evaluator = PerplexityEvaluator(model, tokenizer)
result    = evaluator.compute(eval_texts, max_seq_len=512)

# Output:
# 📊 Perplexity: 42.31 | NLL: 3.7443 | Tokens: 1,234 | Grade: 🟡 Acceptable
```

### Sliding Window PPL

Untuk sequence panjang, digunakan sliding window agar lebih akurat:

```
Tanpa sliding window:
  text[0:512]  → PPL  (tapi token 513+ tidak dievaluasi)

Dengan sliding window (stride=256):
  window 1: text[0:512]      → hitung NLL untuk token 0-511
  window 2: text[256:768]    → hitung NLL untuk token 256-767
                                tapi HANYA hitung token 512-767 (non-overlap)
  window 3: text[512:1024]   → hitung NLL untuk token 768-1023
```

---

## 2. MMLU Benchmark

### Apa itu MMLU?

**Massive Multitask Language Understanding** — 57 subjek akademik dari matematika, sains, hukum, medis, sosial, dll. Setiap soal adalah pilihan ganda (A/B/C/D).

### Cara Evaluasi (Log-likelihood Scoring)

```
Model tidak diminta untuk generate teks.
Sebaliknya: hitung log prob model untuk tiap pilihan,
pilih yang probabilitasnya tertinggi.

Question: "What is the capital of Indonesia?"
A) Surabaya
B) Bandung
C) Jakarta
D) Medan

Score A: log P("A" | question + choices) = -2.3
Score B: log P("B" | question + choices) = -3.1
Score C: log P("C" | question + choices) = -0.8  ← tertinggi → predicted
Score D: log P("D" | question + choices) = -2.9
Answer : C ✅
```

### Target Score MMLU

```
Score         Grade          Interpretasi
──────────────────────────────────────────────────────
> 70%         🟢 Strong      Setara model besar
55% - 70%     🟡 Moderate    Lebih baik dari random guess
40% - 55%     🟠 Weak        Sedikit di atas random
~25%          🔴 Random      Sama seperti asal pilih
```

Target SIGERLM (66M): > 35-40%

```python
from evaluation.benchmarks import MultiplChoiceBenchmark

bench  = MultiplChoiceBenchmark(model, tokenizer)
result = bench.evaluate_mmlu(n_samples=200)

# Output:
# 📊 MMLU Results:
#    Accuracy : 38.5%  🟠 Weak
#    Correct  : 77/200
#
#    Top 5 subjects:
#    world_religions                  55.0%
#    high_school_geography            52.0%
#    international_law                50.0%
#    ...
```

---

## 3. ARC-Challenge

**AI2 Reasoning Challenge** — soal sains grade school yang butuh reasoning, bukan hanya hafalan. Lebih mudah dari MMLU, cocok untuk model kecil.

```python
result = bench.evaluate_arc(n_samples=200)

# 📊 ARC-Challenge: 41.5%  🟡 Moderate
```

Target SIGERLM: > 40%

---

## 4. Evaluasi Bahasa Indonesia

### IndoSentiment (SmSA)

Sentiment analysis teks Indonesia: positif / negatif / netral.

```python
from evaluation.indo_eval import IndoEvaluator

indo  = IndoEvaluator(model, tokenizer)
result = indo.evaluate_sentiment(n_samples=200)

# 📊 IndoSentiment: 58.5%
```

### IndoNLI

Natural Language Inference: apakah hypothesis benar / salah / tidak tentu berdasarkan premise?

```
Premis: "Jakarta adalah ibu kota Indonesia"
Hipotesis: "Indonesia memiliki ibu kota"
Label: ya (entailment) ✅
```

```python
result = indo.evaluate_nli(n_samples=200)
# 📊 IndoNLI: 52.0%
```

### IndoQA

Extractive question answering dari konteks Bahasa Indonesia. Metric: token-level F1.

```python
result = indo.evaluate_qa(n_samples=100)
# 📊 IndoQA F1: 35.2%
```

---

## 5. Generation Quality

### BLEU Score

Mengukur n-gram overlap antara output model dengan referensi.

```
BLEU-1: overlap unigram (kata per kata)
BLEU-2: overlap bigram (pasangan kata)
BLEU-4: overlap 4-gram (standar umum)
```

```python
from evaluation.generation import GenerationEvaluator

gen_eval = GenerationEvaluator()
hypotheses = ["Jakarta adalah ibu kota Indonesia"]
references = ["Ibu kota Indonesia adalah Jakarta"]

result = gen_eval.bleu(hypotheses, references)
# {"bleu": 28.5, "precisions": [75.0, 50.0, 33.3, 0.0], "bp": 1.0}
```

### ROUGE Score

Fokus pada recall (seberapa banyak referensi tercakup).

```
ROUGE-1: unigram recall
ROUGE-2: bigram recall
ROUGE-L: Longest Common Subsequence
```

```python
result = gen_eval.rouge(hypotheses, references)
# {"rouge1": 75.0, "rouge2": 40.0, "rougeL": 75.0}
```

### Diversity (Distinct-N)

Mengukur keberagaman output — model yang bagus tidak repetitif.

```python
outputs = [gen.generate(p) for p in prompts]
result  = gen_eval.diversity(outputs)

# {"distinct1": 72.3, "distinct2": 85.1, "grade": "🟢 Diverse"}
```

```
Distinct-2 > 70%  → 🟢 Diverse
Distinct-2 40-70% → 🟡 Moderate
Distinct-2 < 40%  → 🔴 Repetitive (model sering ngulang kata)
```

---

## 6. Jalankan Semua Evaluasi

```bash
# Eval lengkap
python run_eval.py

# Eval cepat (subset kecil)
python -c "
from evaluation.runner import EvaluationRunner
# ... setup ...
runner.run(n_samples=50, tag='quick_eval')
"
```

### Output Lengkap

```
═══════════════════════════════════════════════════════
  🔬 LLM Evaluation Suite
═══════════════════════════════════════════════════════

📐 [1/5] Perplexity...
📊 Perplexity: 42.31 | NLL: 3.74 | Tokens: 1,234 | Grade: 🟡 Acceptable

📚 [2/5] MMLU Benchmark...
📊 MMLU Results:
   Accuracy : 38.5%  🟠 Weak
   Correct  : 77/200

🔭 [3/5] ARC-Challenge...
📊 ARC-Challenge: 41.5%  🟡 Moderate

🇮🇩 [4/5] Indo Benchmarks...
📊 IndoSentiment: 58.5%
📊 IndoNLI: 52.0%
📊 IndoQA F1: 35.2%

✍️  [5/5] Generation Quality...
  Q: ...Apa ibu kota Indonesia?...
  A: Jakarta adalah ibu kota...

═══════════════════════════════════════════════════════
  📊 EVALUATION SUMMARY
═══════════════════════════════════════════════════════
  Perplexity   : 42.31    🟡 Acceptable
  MMLU         : 38.5%    🟠 Weak
  ARC-Challenge: 41.5%    🟡 Moderate
  Indo Sentiment: 58.5%
  Indo NLI      : 52.0%
  Indo QA F1    : 35.2%
  BLEU         : 22.3
  ROUGE-L      : 48.7
  Diversity-2  : 78.4%    🟢 Diverse

  ⏱️  Total time: 842.3s
═══════════════════════════════════════════════════════
```

---

## 7. Target Score Realistis

```
Metric            Random    SIGERLM Target    SOTA (besar)
──────────────────────────────────────────────────────────
Perplexity        ∞         < 50              ~8
MMLU              25%       > 35%             89%
ARC-Challenge     25%       > 40%             85%
Indo Sentiment    33%       > 55%             82%
IndoNLI           33%       > 50%             79%
IndoQA F1         0%        > 30%             75%
BLEU              0         > 15              30+
Distinct-2        var       > 60%             80%+
```

---

## 8. Eval Before vs After Fine-tuning

```python
# Eval base model
runner.run(tag="base_model")

# Load LoRA, eval lagi
lora_model.load_lora("./checkpoints/lora/lora_step_005000.pt")
runner.model = lora_model
runner.run(tag="lora_finetuned")

# Compare hasil dari dua JSON
import json

base  = json.load(open("evaluation/results/eval_base_model.json"))
lora  = json.load(open("evaluation/results/eval_lora_finetuned.json"))

print(f"PPL   : {base['perplexity']['ppl']} → {lora['perplexity']['ppl']}")
print(f"MMLU  : {base['mmlu']['accuracy']}% → {lora['mmlu']['accuracy']}%")
```