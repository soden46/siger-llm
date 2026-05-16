This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.gitignore
AGENTS.md
checkpoints/tokenizer/tokenizer_config.json
config/model_config.py
evaluation/__init__.py
evaluation/benchmarks.py
evaluation/generation.py
evaluation/indo_eval.py
evaluation/perplexity.py
evaluation/report.py
evaluation/run_eval.py
evaluation/runner.py
inference/__init__.py
inference/api.py
inference/chat.py
inference/generator.py
inference/sampler.py
lora/__init__.py
lora/config.py
lora/dataset.py
lora/layer.py
lora/merge.py
lora/model.py
lora/run_lora.py
lora/trainer.py
main.py
model/mamba_model.py
model/ssm_block.py
model/ssm_core.py
optimization/benchmark.py
optimization/cpu/threading.py
optimization/kvcache.py
optimization/onnx/export.py
optimization/quantization/calibrate.py
optimization/quantization/quantize.py
PROJECT_CONTEXT.md
README.md
tokenizer/__init__.py
tokenizer/special_tokens.py
tokenizer/tests/sample_texts.py
tokenizer/tests/test_tokenizer.py
tokenizer/tokenizer.py
tokenizer/trainer.py
tokenizer/vocab_extender.py
training/__init__.py
training/checkpoint.py
training/dataset.py
training/logger.py
training/optimizer.py
training/trainer.py
```

# Files

## File: .gitignore
````
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environment
.venv/
venv/
env/

# Environment
.env
.env.local

# Build
build/
dist/
*.egg-info/

# Logs
*.log
logs/

# Data
data/raw/
data/processed/
data/cache/

# Checkpoints and models
checkpoints/base/
checkpoints/lora/
checkpoints/quantized/
models/
*.pt
*.pth
*.bin
*.safetensors
*.onnx

# Experiment tracking
runs/
wandb/
mlruns/

# OS / Editor
.DS_Store
Thumbs.db
.vscode/
.idea/
````

## File: AGENTS.md
````markdown
# AGENTS.md

## Project Name

SIGER_LLM

## Project Type

Custom Python LLM framework.

This project is an experimental custom language model framework focused on building, training, evaluating, optimizing, and serving a lightweight LLM-like model.

## Main Goals

The main goals of this project are:

1. Build a custom tokenizer.
2. Build a custom model architecture.
3. Support training and checkpointing.
4. Support LoRA fine-tuning.
5. Support inference through chat, generator, sampler, and API modules.
6. Support evaluation through benchmarks, perplexity, and generation tests.
7. Support optimization such as KV cache, CPU threading, ONNX export, quantization, and benchmarking.
8. Provide clean documentation for installation, usage, architecture, and development.

## Tech Stack

Expected stack:

- Python 3.10+
- PyTorch
- NumPy
- Tokenizer utilities
- Optional FastAPI / Uvicorn for API inference
- Optional ONNX / ONNX Runtime for export and optimized inference
- Optional tqdm for progress bars
- Optional safetensors or torch checkpoints for model saving

Do not assume unavailable dependencies unless they are found in the codebase.

## Repository Structure

```txt
SIGER_LLM/
├── checkpoints/
│   └── tokenizer/
│       └── tokenizer_config.json
├── config/
│   └── model_config.py
├── evaluation/
│   ├── benchmarks.py
│   ├── generation.py
│   ├── indo_eval.py
│   ├── perplexity.py
│   ├── report.py
│   ├── run_eval.py
│   └── runner.py
├── inference/
│   ├── api.py
│   ├── chat.py
│   ├── generator.py
│   └── sampler.py
├── lora/
│   ├── config.py
│   ├── dataset.py
│   ├── layer.py
│   ├── merge.py
│   ├── model.py
│   ├── run_lora.py
│   └── trainer.py
├── model/
│   ├── mamba_model.py
│   ├── ssm_block.py
│   └── ssm_core.py
├── optimization/
│   ├── benchmark.py
│   ├── kvcache.py
│   ├── cpu/
│   │   └── threading.py
│   ├── onnx/
│   │   └── export.py
│   └── quantization/
│       ├── calibrate.py
│       └── quantize.py
├── tokenizer/
│   ├── sample_texts.py
│   ├── special_tokens.py
│   ├── tokenizer.py
│   ├── trainer.py
│   ├── vocab_extender.py
│   └── tests/
│       └── test_tokenizer.py
├── training/
│   ├── checkpoint.py
│   ├── dataset.py
│   ├── logger.py
│   ├── optimizer.py
│   └── trainer.py
├── main.py
├── README.md
└── AGENTS.md
````

## File: checkpoints/tokenizer/tokenizer_config.json
````json
{
  "base_encoding": "cl100k_base",
  "special_tokens": {
    "<|endoftext|>": 100257,
    "<|pad|>": 100258,
    "<|unk|>": 100259,
    "<|system|>": 100260,
    "<|user|>": 100261,
    "<|assistant|>": 100262,
    "<|end_turn|>": 100263,
    "<|lang_id|>": 100264,
    "<|id|>": 100265,
    "<|en|>": 100266,
    "<|code|>": 100267,
    "<|bos|>": 100268,
    "<|eos|>": 100269,
    "<|sep|>": 100270
  },
  "vocab_size": 100271
}
````

## File: config/model_config.py
````python
# config/model_config.py
from dataclasses import dataclass

@dataclass
class MambaConfig:
    vocab_size: int = 32000
    d_model: int = 512        # dimension utama
    n_layers: int = 12        # jumlah SSM block
    d_state: int = 16         # ukuran state SSM (N)
    d_conv: int = 4           # kernel conv lokal
    expand: int = 2           # expansion factor
    dt_rank: str = "auto"     # rank untuk delta
    dropout: float = 0.1
    max_seq_len: int = 2048
````

## File: evaluation/__init__.py
````python

````

## File: evaluation/benchmarks.py
````python
# evaluation/benchmarks.py
import torch
import torch.nn.functional as F
from datasets import load_dataset
from typing import List, Dict, Optional
import json


class MultiplChoiceBenchmark:
    """
    Evaluasi model di task multiple choice.
    Cara: hitung log prob tiap pilihan jawaban,
    pilih yang prob-nya paling tinggi.
    
    Ga perlu model bisa generate — cukup score!
    Cocok banget buat model yang masih training.
    """

    SUPPORTED = {
        "mmlu":      ("cais/mmlu", "all"),
        "arc":       ("ai2_arc", "ARC-Challenge"),
        "hellaswag": ("Rowan/hellaswag", None),
        "indo_mmlu": ("indolem/indo-mmlu", None),   # Indo version!
    }

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def score_completion(self, prefix: str, completion: str) -> float:
        """
        Hitung log prob model buat 'completion' dikasih 'prefix'.
        Makin tinggi = model lebih yakin completion ini benar.
        """
        prefix_ids     = self.tokenizer.encode(prefix, add_bos=True)
        completion_ids = self.tokenizer.encode(completion)
        full_ids       = prefix_ids + completion_ids

        x      = torch.tensor([full_ids], dtype=torch.long).to(self.device)
        logits, _ = self.model(x)

        # Ambil logits di posisi prefix saja (untuk prediksi completion)
        prefix_len = len(prefix_ids)
        log_probs  = F.log_softmax(logits[0], dim=-1)

        total_logp = 0.0
        for i, token_id in enumerate(completion_ids):
            pos        = prefix_len + i - 1   # posisi yang predict token ini
            total_logp += log_probs[pos, token_id].item()

        # Normalize by length biar fair antar pilihan yang panjangnya beda
        return total_logp / len(completion_ids)

    @torch.no_grad()
    def evaluate_mmlu(
        self,
        n_samples: int    = 200,
        subjects: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluasi di MMLU — 57 subject akademik.
        Format: question + 4 pilihan (A/B/C/D)
        """
        print(f"📚 Loading MMLU dataset...")
        dataset = load_dataset("cais/mmlu", "all", split="test")

        if subjects:
            dataset = dataset.filter(lambda x: x["subject"] in subjects)
        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        correct      = 0
        total        = 0
        by_subject   = {}
        choices_map  = ["A", "B", "C", "D"]

        for ex in dataset:
            question = ex["question"]
            choices  = ex["choices"]
            answer   = ex["answer"]   # index 0-3
            subject  = ex["subject"]

            # Build prefix
            prefix = (
                f"Question: {question}\n"
                f"A) {choices[0]}\n"
                f"B) {choices[1]}\n"
                f"C) {choices[2]}\n"
                f"D) {choices[3]}\n"
                f"Answer:"
            )

            # Score tiap pilihan
            scores = []
            for i, choice_text in enumerate(choices):
                label  = f" {choices_map[i]}"
                score  = self.score_completion(prefix, label)
                scores.append(score)

            predicted = scores.index(max(scores))
            is_correct = (predicted == answer)

            correct += int(is_correct)
            total   += 1

            if subject not in by_subject:
                by_subject[subject] = {"correct": 0, "total": 0}
            by_subject[subject]["correct"] += int(is_correct)
            by_subject[subject]["total"]   += 1

        accuracy = correct / total if total > 0 else 0

        # Sort subject by accuracy
        subject_acc = {
            s: v["correct"] / v["total"]
            for s, v in by_subject.items()
        }
        subject_acc = dict(sorted(subject_acc.items(),
                                  key=lambda x: x[1], reverse=True))

        result = {
            "accuracy":    round(accuracy * 100, 2),
            "correct":     correct,
            "total":       total,
            "by_subject":  subject_acc,
            "grade":       self._grade_mmlu(accuracy),
        }

        print(f"\n📊 MMLU Results:")
        print(f"   Accuracy : {accuracy*100:.1f}%  {result['grade']}")
        print(f"   Correct  : {correct}/{total}")
        print(f"\n   Top 5 subjects:")
        for subj, acc in list(subject_acc.items())[:5]:
            print(f"   {subj:<30} {acc*100:.1f}%")

        return result

    @torch.no_grad()
    def evaluate_arc(self, n_samples: int = 200) -> Dict:
        """
        ARC (AI2 Reasoning Challenge) — soal science grade school.
        Lebih gampang dari MMLU, cocok buat model kecil.
        """
        print("📚 Loading ARC-Challenge...")
        dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        correct = 0
        total   = 0

        for ex in dataset:
            question = ex["question"]
            choices  = ex["choices"]
            answer   = ex["answerKey"]   # "A", "B", "C", "D", atau "1","2","3","4"

            labels  = choices["label"]
            texts   = choices["text"]

            prefix = f"Question: {question}\nAnswer:"

            scores = []
            for label, text in zip(labels, texts):
                score = self.score_completion(prefix, f" {text}")
                scores.append(score)

            predicted_idx  = scores.index(max(scores))
            predicted_label = labels[predicted_idx]

            # Normalize answer key (1→A, 2→B, dsb)
            answer = answer.strip()
            if answer.isdigit():
                answer = "ABCD"[int(answer) - 1]

            correct += int(predicted_label == answer)
            total   += 1

        accuracy = correct / total if total > 0 else 0
        result   = {
            "accuracy": round(accuracy * 100, 2),
            "correct":  correct,
            "total":    total,
            "grade":    self._grade_arc(accuracy),
        }
        print(f"📊 ARC-Challenge: {accuracy*100:.1f}%  {result['grade']}")
        return result

    @staticmethod
    def _grade_mmlu(acc: float) -> str:
        if acc >= 0.70: return "🟢 Strong"
        if acc >= 0.55: return "🟡 Moderate"
        if acc >= 0.40: return "🟠 Weak"
        return                 "🔴 Random (≈25%)"

    @staticmethod
    def _grade_arc(acc: float) -> str:
        if acc >= 0.65: return "🟢 Strong"
        if acc >= 0.50: return "🟡 Moderate"
        if acc >= 0.35: return "🟠 Weak"
        return                 "🔴 Poor"
````

## File: evaluation/generation.py
````python
# evaluation/generation.py
import torch
from typing import List, Dict
from collections import Counter
import math
import re


class GenerationEvaluator:
    """
    Evaluasi kualitas teks yang di-generate.
    
    BLEU   → overlap n-gram vs referensi (precision-focused)
    ROUGE  → overlap n-gram vs referensi (recall-focused)
    Diversity → seberapa variatif output model
    """

    # ── BLEU ──────────────────────────────────────────────

    @staticmethod
    def _ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))

    @classmethod
    def bleu(
        cls,
        hypotheses: List[str],
        references: List[str],
        max_n: int = 4,
    ) -> Dict:
        """
        Corpus-level BLEU score.
        Makin tinggi makin mirip referensi (max 100).
        """
        clipped_counts = Counter()
        total_counts   = Counter()
        hyp_len = ref_len = 0

        for hyp, ref in zip(hypotheses, references):
            hyp_tok = hyp.lower().split()
            ref_tok = ref.lower().split()
            hyp_len += len(hyp_tok)
            ref_len += len(ref_tok)

            for n in range(1, max_n + 1):
                hyp_ngrams = cls._ngrams(hyp_tok, n)
                ref_ngrams = cls._ngrams(ref_tok, n)
                clipped    = {k: min(v, ref_ngrams[k]) for k, v in hyp_ngrams.items()}
                clipped_counts[n] += sum(clipped.values())
                total_counts[n]   += sum(hyp_ngrams.values())

        # Precision per n-gram
        precisions = []
        for n in range(1, max_n + 1):
            if total_counts[n] == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped_counts[n] / total_counts[n])

        # Brevity penalty
        bp   = 1.0 if hyp_len >= ref_len else math.exp(1 - ref_len / hyp_len)
        logsum = sum(math.log(p) if p > 0 else float("-inf") for p in precisions)
        bleu   = bp * math.exp(logsum / max_n)

        return {
            "bleu":       round(bleu * 100, 2),
            "precisions": [round(p * 100, 2) for p in precisions],
            "bp":         round(bp, 4),
            "grade":      "🟢 Good" if bleu > 0.3 else
                          "🟡 Fair" if bleu > 0.15 else "🔴 Poor",
        }

    # ── ROUGE ─────────────────────────────────────────────

    @classmethod
    def rouge(
        cls,
        hypotheses: List[str],
        references: List[str],
    ) -> Dict:
        """ROUGE-1, ROUGE-2, ROUGE-L scores."""

        def rouge_n(hyp, ref, n):
            hyp_ng = cls._ngrams(hyp.lower().split(), n)
            ref_ng = cls._ngrams(ref.lower().split(), n)
            match  = sum((hyp_ng & ref_ng).values())
            prec   = match / max(sum(hyp_ng.values()), 1)
            rec    = match / max(sum(ref_ng.values()), 1)
            f1     = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0
            return f1

        def rouge_l(hyp, ref):
            hyp_tok = hyp.lower().split()
            ref_tok = ref.lower().split()
            # LCS dynamic programming
            m, n = len(hyp_tok), len(ref_tok)
            dp   = [[0] * (n+1) for _ in range(m+1)]
            for i in range(1, m+1):
                for j in range(1, n+1):
                    if hyp_tok[i-1] == ref_tok[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            lcs  = dp[m][n]
            prec = lcs / max(m, 1)
            rec  = lcs / max(n, 1)
            return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0

        r1 = sum(rouge_n(h, r, 1) for h, r in zip(hypotheses, references))
        r2 = sum(rouge_n(h, r, 2) for h, r in zip(hypotheses, references))
        rl = sum(rouge_l(h, r) for h, r in zip(hypotheses, references))
        n  = len(hypotheses)

        return {
            "rouge1": round(r1 / n * 100, 2),
            "rouge2": round(r2 / n * 100, 2),
            "rougeL": round(rl / n * 100, 2),
        }

    # ── Diversity ─────────────────────────────────────────

    @staticmethod
    def diversity(texts: List[str]) -> Dict:
        """
        Distinct-1 dan Distinct-2:
        Seberapa beragam unigram/bigram di semua output.
        Makin tinggi = makin variatif, ga repetitif.
        """
        all_unigrams, all_bigrams = [], []
        for text in texts:
            tokens = text.lower().split()
            all_unigrams.extend(tokens)
            all_bigrams.extend(zip(tokens, tokens[1:]))

        d1 = len(set(all_unigrams)) / max(len(all_unigrams), 1)
        d2 = len(set(all_bigrams))  / max(len(all_bigrams),  1)

        return {
            "distinct1": round(d1 * 100, 2),
            "distinct2": round(d2 * 100, 2),
            "grade":     "🟢 Diverse"    if d2 > 0.7 else
                         "🟡 Moderate"   if d2 > 0.4 else
                         "🔴 Repetitive",
        }
````

## File: evaluation/indo_eval.py
````python
# evaluation/indo_eval.py
import torch
from typing import List, Dict
from datasets import load_dataset


class IndoEvaluator:
    """
    Benchmark khusus Bahasa Indonesia.
    
    Dataset yang dipake:
    - IndoNLI    → Natural Language Inference
    - IndoQA     → Question Answering
    - SmSA       → Sentiment Analysis
    - IndoMMLU   → Academic knowledge Indo
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def evaluate_sentiment(self, n_samples: int = 200) -> Dict:
        """
        SmSA — Sentiment Analysis Bahasa Indonesia.
        Label: positif / negatif / netral
        """
        print("📚 Loading SmSA (Indo Sentiment)...")
        try:
            dataset = load_dataset("indonlp/SmSA", split="test")
        except Exception:
            print("⚠️  SmSA tidak tersedia, skip.")
            return {}

        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        labels_map = {0: "negatif", 1: "netral", 2: "positif"}
        correct    = 0

        for ex in dataset:
            text  = ex["sentence"]
            label = labels_map[ex["sentiment"]]

            prefix = f'Teks: "{text}"\nSentimen teks ini adalah:'
            scores = {
                sent: self._score(prefix, f" {sent}")
                for sent in ["positif", "negatif", "netral"]
            }
            predicted = max(scores, key=scores.get)
            correct  += int(predicted == label)

        acc = correct / len(dataset)
        print(f"📊 IndoSentiment: {acc*100:.1f}%")
        return {"accuracy": round(acc * 100, 2), "total": len(dataset)}

    @torch.no_grad()
    def evaluate_nli(self, n_samples: int = 200) -> Dict:
        """
        IndoNLI — apakah premise entail/contradict/neutral hypothesis.
        """
        print("📚 Loading IndoNLI...")
        try:
            dataset = load_dataset("indonlp/indonli", split="test_lay")
        except Exception:
            print("⚠️  IndoNLI tidak tersedia, skip.")
            return {}

        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        label_map = {"e": "ya", "c": "tidak", "n": "mungkin"}
        correct   = 0

        for ex in dataset:
            premise    = ex["premise"]
            hypothesis = ex["hypothesis"]
            label      = label_map.get(ex["label"], "mungkin")

            prefix = (
                f"Premis: {premise}\n"
                f"Hipotesis: {hypothesis}\n"
                f"Apakah hipotesis benar berdasarkan premis? (ya/tidak/mungkin):"
            )
            scores = {
                ans: self._score(prefix, f" {ans}")
                for ans in ["ya", "tidak", "mungkin"]
            }
            predicted = max(scores, key=scores.get)
            correct  += int(predicted == label)

        acc = correct / len(dataset)
        print(f"📊 IndoNLI: {acc*100:.1f}%")
        return {"accuracy": round(acc * 100, 2), "total": len(dataset)}

    @torch.no_grad()
    def evaluate_qa(self, n_samples: int = 100) -> Dict:
        """
        IndoQA — Question Answering Bahasa Indonesia.
        Pakai F1 token overlap sebagai metric.
        """
        print("📚 Loading IndoQA...")
        try:
            dataset = load_dataset("jakartaresearch/indoqa", split="validation")
        except Exception:
            print("⚠️  IndoQA tidak tersedia, skip.")
            return {}

        if n_samples:
            dataset = dataset.select(range(min(n_samples, len(dataset))))

        from inference.generator import Generator
        from inference.sampler   import Sampler

        f1_scores = []
        for ex in dataset:
            context   = ex["context"][:500]   # truncate context
            question  = ex["question"]
            answer    = ex["answers"]["text"][0] if ex["answers"]["text"] else ""

            prompt = (
                f"Konteks: {context}\n"
                f"Pertanyaan: {question}\n"
                f"Jawaban:"
            )
            # Simple greedy generation buat QA
            input_ids = self.tokenizer.encode(prompt, add_bos=True)
            x         = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            logits, _ = self.model(x)
            pred_id   = logits[0, -1, :].argmax().item()
            predicted = self.tokenizer.decode([pred_id])

            f1 = self._token_f1(predicted, answer)
            f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores)
        print(f"📊 IndoQA F1: {avg_f1*100:.1f}%")
        return {"f1": round(avg_f1 * 100, 2), "total": len(dataset)}

    def _score(self, prefix: str, completion: str) -> float:
        import torch.nn.functional as F
        p_ids = self.tokenizer.encode(prefix, add_bos=True)
        c_ids = self.tokenizer.encode(completion)
        full  = torch.tensor([p_ids + c_ids], dtype=torch.long).to(self.device)
        logits, _ = self.model(full)
        log_probs  = F.log_softmax(logits[0], dim=-1)
        score = sum(
            log_probs[len(p_ids) + i - 1, tok].item()
            for i, tok in enumerate(c_ids)
        )
        return score / max(len(c_ids), 1)

    @staticmethod
    def _token_f1(pred: str, ref: str) -> float:
        pred_toks = set(pred.lower().split())
        ref_toks  = set(ref.lower().split())
        if not pred_toks or not ref_toks:
            return 0.0
        common = pred_toks & ref_toks
        if not common:
            return 0.0
        prec = len(common) / len(pred_toks)
        rec  = len(common) / len(ref_toks)
        return 2 * prec * rec / (prec + rec)
````

## File: evaluation/perplexity.py
````python
# evaluation/perplexity.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
import math


class PerplexityEvaluator:
    """
    Perplexity = exp(average negative log likelihood)
    
    Intuisi:
    PPL = 1     → model 100% yakin (impossible)
    PPL = 10    → model punya 10 pilihan yang equally likely → bagus
    PPL = 100   → model bingung → jelek
    PPL = 1000+ → model random banget → ga belajar apa-apa

    Referensi:
    GPT-2 small  : PPL ~29 (WikiText-103)
    GPT-3        : PPL ~20
    Target lo    : PPL < 50 (acceptable buat model kecil)
    """

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model     = model.eval().to(device)
        self.tokenizer = tokenizer
        self.device    = device

    @torch.no_grad()
    def compute(
        self,
        texts: List[str],
        max_seq_len: int  = 512,
        stride: int       = 256,   # sliding window — lebih akurat
        batch_size: int   = 4,
    ) -> dict:
        """
        Hitung PPL dengan sliding window.
        Sliding window penting buat sequence panjang:
        daripada truncate, lo overlap tiap window
        dan hanya hitung loss di bagian non-overlap.
        """
        total_nll  = 0.0   # negative log likelihood
        total_toks = 0

        for text in texts:
            input_ids = self.tokenizer.encode(text, add_bos=True)
            seq_len   = len(input_ids)

            if seq_len < 2:
                continue

            ids_tensor = torch.tensor(input_ids, dtype=torch.long)

            # Sliding window
            prev_end = 0
            for begin in range(0, seq_len - 1, stride):
                end        = min(begin + max_seq_len, seq_len)
                window     = ids_tensor[begin:end].unsqueeze(0).to(self.device)
                target_len = end - max(begin, prev_end)   # hanya hitung bagian baru

                logits, _ = self.model(window)

                # Shift: predict token ke-i dari token ke-(i-1)
                shift_logits = logits[0, :-1, :]   # (L-1, vocab)
                shift_labels = window[0, 1:]        # (L-1,)

                # Hanya hitung loss di bagian baru (bukan overlap)
                target_start = max(0, prev_end - begin - 1)
                if target_start >= shift_logits.size(0):
                    continue

                sl = shift_logits[target_start:]
                tl = shift_labels[target_start:]

                nll = F.cross_entropy(sl, tl, reduction="sum")
                total_nll  += nll.item()
                total_toks += tl.size(0)

                prev_end = end
                if end == seq_len:
                    break

        if total_toks == 0:
            return {"ppl": float("inf"), "nll": float("inf"), "tokens": 0}

        avg_nll = total_nll / total_toks
        ppl     = math.exp(avg_nll)

        result = {
            "ppl":    round(ppl, 2),
            "nll":    round(avg_nll, 4),
            "tokens": total_toks,
            "grade":  self._grade(ppl),
        }

        print(f"📊 Perplexity: {ppl:.2f} | NLL: {avg_nll:.4f} "
              f"| Tokens: {total_toks:,} | Grade: {result['grade']}")
        return result

    @staticmethod
    def _grade(ppl: float) -> str:
        if ppl < 15:   return "🟢 Excellent"
        if ppl < 30:   return "🟢 Good"
        if ppl < 50:   return "🟡 Acceptable"
        if ppl < 100:  return "🟠 Poor"
        return             "🔴 Very Poor"

    def compare(
        self,
        texts: List[str],
        model_names: List[str],
        models: list,
    ) -> dict:
        """Bandingkan PPL beberapa model sekaligus."""
        results = {}
        for name, model in zip(model_names, models):
            self.model = model.eval().to(self.device)
            print(f"\nEvaluating: {name}")
            results[name] = self.compute(texts)
        return results
````

## File: evaluation/report.py
````python

````

## File: evaluation/run_eval.py
````python
# run_eval.py
from optimization.cpu.threading import configure_cpu
from optimization.cpu.memory    import load_model_efficient
from config.model_config        import MambaConfig
from model.mamba_model          import MambaLM
from tokenizer.tokenizer        import MultilingualTokenizer
from inference.generator        import Generator
from evaluation.runner          import EvaluationRunner


def main():
    configure_cpu(n_cores=2)

    # Load model
    config = MambaConfig(vocab_size=100277, d_model=512, n_layers=12)
    model  = load_model_efficient(MambaLM, config, "./checkpoints/best_model.pt")
    tok    = MultilingualTokenizer()
    gen    = Generator(model, tok)

    # Run evaluation
    runner = EvaluationRunner(
        model     = model,
        tokenizer = tok,
        generator = gen,
        device    = "cpu",
    )

    # Eval base model
    runner.run(
        n_samples = 100,   # kecil dulu di VPS
        tag       = "base_model",
    )

    # Kalau udah ada LoRA:
    # lora_model = LoRAModel(model, lora_config)
    # lora_model.load_lora("./checkpoints/lora/lora_step_005000.pt")
    # runner.model = lora_model
    # runner.run(tag="lora_finetuned")


if __name__ == "__main__":
    main()
````

## File: evaluation/runner.py
````python
# evaluation/runner.py
import json
import time
from pathlib import Path
from typing import List, Optional

from .perplexity import PerplexityEvaluator
from .benchmarks import MultiplChoiceBenchmark
from .generation import GenerationEvaluator
from .indo_eval  import IndoEvaluator
from .report     import EvalReport


class EvaluationRunner:
    """
    Jalanin semua evaluasi dalam 1 call.
    Kayak `php artisan test` — run semua test suite sekaligus.
    """

    # Teks Indo + EN buat PPL evaluation
    PPL_TEXTS = [
        "Pemerintah Indonesia mengumumkan kebijakan baru terkait pendidikan nasional.",
        "Teknologi kecerdasan buatan berkembang pesat dalam beberapa tahun terakhir.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "Machine learning models require large amounts of training data.",
        "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa.",
    ]

    # Instruction following test cases
    INSTRUCTION_TESTS = [
        {
            "prompt":    "<|user|>Apa ibu kota Indonesia?<|end_turn|>\n<|assistant|>",
            "reference": "Jakarta adalah ibu kota Indonesia.",
        },
        {
            "prompt":    "<|user|>Jelaskan apa itu machine learning dalam 1 kalimat.<|end_turn|>\n<|assistant|>",
            "reference": "Machine learning adalah cabang AI yang memungkinkan komputer belajar dari data.",
        },
        {
            "prompt":    "<|user|>Sebutkan 3 bahasa pemrograman populer.<|end_turn|>\n<|assistant|>",
            "reference": "Python, JavaScript, dan Java adalah bahasa pemrograman populer.",
        },
    ]

    def __init__(
        self,
        model,
        tokenizer,
        generator=None,
        device: str = "cpu",
        output_dir: str = "./evaluation/results",
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.device    = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        run_ppl:   bool = True,
        run_mmlu:  bool = True,
        run_arc:   bool = True,
        run_indo:  bool = True,
        run_gen:   bool = True,
        n_samples: int  = 200,
        tag:       str  = "eval",
    ) -> dict:

        results  = {}
        start    = time.time()

        print(f"\n{'='*55}")
        print(f"  🔬 LLM Evaluation Suite")
        print(f"{'='*55}\n")

        # ── 1. Perplexity ──────────────────────────────────
        if run_ppl:
            print("📐 [1/5] Perplexity...")
            ppl_eval     = PerplexityEvaluator(self.model, self.tokenizer, self.device)
            results["perplexity"] = ppl_eval.compute(self.PPL_TEXTS)

        # ── 2. MMLU ────────────────────────────────────────
        if run_mmlu:
            print("\n📚 [2/5] MMLU Benchmark...")
            mc_eval = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["mmlu"] = mc_eval.evaluate_mmlu(n_samples=n_samples)

        # ── 3. ARC ─────────────────────────────────────────
        if run_arc:
            print("\n🔭 [3/5] ARC-Challenge...")
            mc_eval       = MultiplChoiceBenchmark(self.model, self.tokenizer, self.device)
            results["arc"] = mc_eval.evaluate_arc(n_samples=n_samples)

        # ── 4. Indo Benchmarks ─────────────────────────────
        if run_indo:
            print("\n🇮🇩 [4/5] Indo Benchmarks...")
            indo_eval = IndoEvaluator(self.model, self.tokenizer, self.device)
            results["indo"] = {
                "sentiment": indo_eval.evaluate_sentiment(n_samples),
                "nli":       indo_eval.evaluate_nli(n_samples),
                "qa":        indo_eval.evaluate_qa(n_samples // 2),
            }

        # ── 5. Generation Quality ──────────────────────────
        if run_gen and self.generator:
            print("\n✍️  [5/5] Generation Quality...")
            gen_eval = GenerationEvaluator()
            hypotheses = []
            references = []

            for test in self.INSTRUCTION_TESTS:
                out = self.generator.generate(
                    test["prompt"],
                    max_new_tokens=50,
                    temperature=0.1,   # low temp buat eval
                )
                hypotheses.append(out)
                references.append(test["reference"])
                print(f"  Q: {test['prompt'][-40:]}")
                print(f"  A: {out[:60]}")

            results["generation"] = {
                "bleu":      gen_eval.bleu(hypotheses, references),
                "rouge":     gen_eval.rouge(hypotheses, references),
                "diversity": gen_eval.diversity(hypotheses),
            }

        # ── Summary ────────────────────────────────────────
        elapsed = time.time() - start
        results["meta"] = {"elapsed_sec": round(elapsed, 1), "tag": tag}

        self._print_summary(results)
        self._save(results, tag)
        return results

    def _print_summary(self, results: dict):
        print(f"\n{'='*55}")
        print(f"  📊 EVALUATION SUMMARY")
        print(f"{'='*55}")

        if "perplexity" in results:
            r = results["perplexity"]
            print(f"  Perplexity   : {r['ppl']:<8} {r.get('grade','')}")

        if "mmlu" in results:
            r = results["mmlu"]
            print(f"  MMLU         : {r['accuracy']}%    {r.get('grade','')}")

        if "arc" in results:
            r = results["arc"]
            print(f"  ARC-Challenge: {r['accuracy']}%    {r.get('grade','')}")

        if "indo" in results:
            r = results["indo"]
            if r.get("sentiment"):
                print(f"  Indo Sentiment: {r['sentiment'].get('accuracy','N/A')}%")
            if r.get("nli"):
                print(f"  Indo NLI      : {r['nli'].get('accuracy','N/A')}%")
            if r.get("qa"):
                print(f"  Indo QA F1    : {r['qa'].get('f1','N/A')}%")

        if "generation" in results:
            r = results["generation"]
            print(f"  BLEU         : {r['bleu']['bleu']}")
            print(f"  ROUGE-L      : {r['rouge']['rougeL']}")
            print(f"  Diversity-2  : {r['diversity']['distinct2']}%  "
                  f"{r['diversity']['grade']}")

        elapsed = results.get("meta", {}).get("elapsed_sec", 0)
        print(f"\n  ⏱️  Total time: {elapsed:.1f}s")
        print(f"{'='*55}\n")

    def _save(self, results: dict, tag: str):
        path = self.output_dir / f"eval_{tag}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"💾 Results saved: {path}")
````

## File: inference/__init__.py
````python

````

## File: inference/api.py
````python
# inference/api.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import asyncio
import json

from .generator import Generator
from .chat      import ChatSession


app = FastAPI(title="LLM API", version="1.0.0")

# Generator instance (diinit saat startup)
_generator: Optional[Generator] = None
_sessions: dict[str, ChatSession] = {}


# ── Request/Response Models ────────────────────────────────
class GenerateRequest(BaseModel):
    prompt:             str
    max_new_tokens:     int   = Field(200, ge=1, le=2048)
    temperature:        float = Field(0.8, ge=0.0, le=2.0)
    top_k:              int   = Field(50, ge=0)
    top_p:              float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.15, ge=1.0)
    lang:               Optional[str] = None
    stream:             bool  = False

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    stream:     bool  = False
    temperature: float = 0.8
    max_new_tokens: int = 300

class GenerateResponse(BaseModel):
    text:        str
    token_count: int


# ── Endpoints ──────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _generator is not None}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    if req.stream:
        # Streaming response (SSE-style)
        async def token_stream():
            for token in _generator.stream(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                repetition_penalty=req.repetition_penalty,
                lang=req.lang,
            ):
                data = json.dumps({"token": token})
                yield f"data: {data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(token_stream(), media_type="text/event-stream")

    # Normal response
    text = _generator.generate(
        req.prompt,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        lang=req.lang,
    )
    token_count = _generator.tokenizer.count_tokens(text)
    return GenerateResponse(text=text, token_count=token_count)


@app.post("/chat")
async def chat(req: ChatRequest):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    # Get or create session
    if req.session_id not in _sessions:
        _sessions[req.session_id] = ChatSession(_generator)

    session  = _sessions[req.session_id]
    response = session.chat(
        req.message,
        stream=False,
        temperature=req.temperature,
        max_new_tokens=req.max_new_tokens,
    )
    return {"session_id": req.session_id, "response": response}


@app.delete("/chat/{session_id}")
def reset_chat(session_id: str):
    if session_id in _sessions:
        _sessions[session_id].reset()
    return {"status": "reset"}


# ── Startup ────────────────────────────────────────────────
def init_api(generator: Generator):
    """Panggil ini sebelum jalanin uvicorn."""
    global _generator
    _generator = generator
````

## File: inference/chat.py
````python
# inference/chat.py
import os
from typing import Optional
from .generator import Generator


class ChatSession:
    """
    Stateful chat session dengan history.
    Analoginya: kayak Session di Laravel — state disimpan per user.

    Format conversation:
    <|system|> ... <|end_turn|>
    <|user|>   ... <|end_turn|>
    <|assistant|> ... <|end_turn|>
    """

    SYSTEM_PROMPT = (
        "Kamu adalah asisten AI yang cerdas dan helpful. "
        "Jawab dalam bahasa yang sama dengan pertanyaan user. "
        "Jawab dengan jelas, ringkas, dan akurat."
    )

    def __init__(
        self,
        generator: Generator,
        system_prompt: Optional[str] = None,
        max_history: int = 10,          # max turn disimpan
        max_context_tokens: int = 1024, # max total token dikirim ke model
    ):
        self.generator         = generator
        self.system_prompt     = system_prompt or self.SYSTEM_PROMPT
        self.max_history       = max_history
        self.max_context_tokens = max_context_tokens
        self.history: list[dict] = []  # [{"role": ..., "content": ...}]

    def _build_prompt(self) -> str:
        """
        Bangun prompt string dari history.
        Format: system → [user/assistant turns] → assistant prefix
        """
        tok = self.generator.tokenizer
        parts = []

        # System
        parts.append(
            f"<|system|>{self.system_prompt}<|end_turn|>"
        )

        # History turns
        for turn in self.history:
            role    = turn["role"]
            content = turn["content"]
            parts.append(f"<|{role}|>{content}<|end_turn|>")

        # Prefix untuk response berikutnya
        parts.append("<|assistant|>")

        prompt = "\n".join(parts)

        # Truncate kalau terlalu panjang
        token_count = tok.count_tokens(prompt)
        if token_count > self.max_context_tokens:
            # Hapus history paling lama (kecuali system)
            while len(self.history) > 2 and token_count > self.max_context_tokens:
                self.history.pop(0)
                prompt = self._build_prompt()
                token_count = tok.count_tokens(prompt)

        return prompt

    def chat(
        self,
        user_input: str,
        stream: bool = False,
        **gen_kwargs,
    ) -> str:
        """
        Kirim pesan, dapat respons.
        stream=True → print karakter per karakter (terminal effect).
        """
        # Tambah user turn ke history
        self.history.append({"role": "user", "content": user_input})

        # Build full prompt
        prompt = self._build_prompt()

        # Generate
        if stream:
            response = self._stream_response(prompt, **gen_kwargs)
        else:
            response = self.generator.generate(
                prompt,
                stop_tokens=[
                    self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
                    self.generator.tokenizer.special_tokens.get("<|user|>"),
                    self.generator.tokenizer.eos_id,
                ],
                **gen_kwargs
            )

        # Simpan response ke history
        self.history.append({"role": "assistant", "content": response})

        # Trim history kalau udah terlalu panjang
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-self.max_history * 2:]

        return response

    def _stream_response(self, prompt: str, **kwargs) -> str:
        """Stream response ke terminal, return full string."""
        full_response = ""
        print("Assistant: ", end="", flush=True)

        stop_ids = [
            self.generator.tokenizer.special_tokens.get("<|end_turn|>"),
            self.generator.tokenizer.eos_id,
        ]

        for token_str in self.generator.stream(prompt, **kwargs):
            # Stop kalau ketemu end_turn dalam token string
            if any(s in token_str for s in ["<|end_turn|>", "<|user|>"]):
                break
            print(token_str, end="", flush=True)
            full_response += token_str

        print()  # newline setelah selesai
        return full_response.strip()

    def reset(self):
        """Reset history — mulai percakapan baru."""
        self.history.clear()
        print("🔄 Chat session reset.")

    def show_history(self):
        """Print seluruh history."""
        print(f"\n{'─'*50}")
        for turn in self.history:
            role = turn["role"].upper()
            print(f"[{role}]: {turn['content'][:100]}...")
        print(f"{'─'*50}\n")
````

## File: inference/generator.py
````python
# inference/generator.py
import torch
from typing import Iterator, Optional
from tokenizer.tokenizer import MultilingualTokenizer
from model.mamba_model   import MambaLM


class Generator:
    """
    Autoregressive text generator.
    
    Cara kerja:
    Input tokens → model → logits → sample → append → repeat
    Kayak loop query SQL satu-satu, tiap iterasi dapat 1 token baru.
    """

    def __init__(
        self,
        model: MambaLM,
        tokenizer: MultilingualTokenizer,
        device: str = None,
    ):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
        repetition_penalty: float = 1.15,
        lang: Optional[str]  = None,     # "id", "en", "code"
        stop_tokens: list    = None,
    ) -> str:
        """
        Generate teks dari prompt.
        Return string lengkap (prompt + generated).
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(
            prompt, add_bos=True, lang=lang
        )
        generated_ids = list(input_ids)

        stop_tokens = stop_tokens or [self.tokenizer.eos_id]

        for _ in range(max_new_tokens):
            # Prepare input tensor
            x = torch.tensor([generated_ids], dtype=torch.long, device=self.device)

            # Forward pass
            logits, _ = self.model(x)

            # Ambil logits token terakhir saja
            next_logits = logits[0, -1, :]  # (vocab_size,)

            # Sample next token
            next_token = Sampler.sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty=repetition_penalty,
                generated_ids=generated_ids,
                do_sample=(temperature > 0),
            )

            generated_ids.append(next_token)

            # Stop kalau ketemu EOS
            if next_token in stop_tokens:
                break

        # Decode — skip prompt, return generated only
        output_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def stream(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
        repetition_penalty: float = 1.15,
        lang: Optional[str]  = None,
    ) -> Iterator[str]:
        """
        Streaming generation — yield 1 token per iterasi.
        Kayak Server-Sent Events di Laravel, token ngalir satu-satu.
        """
        input_ids     = self.tokenizer.encode(prompt, add_bos=True, lang=lang)
        generated_ids = list(input_ids)

        for _ in range(max_new_tokens):
            x = torch.tensor([generated_ids], dtype=torch.long, device=self.device)
            logits, _ = self.model(x)
            next_logits = logits[0, -1, :]

            next_token = Sampler.sample(
                next_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                penalty=repetition_penalty,
                generated_ids=generated_ids,
            )

            if next_token == self.tokenizer.eos_id:
                break

            generated_ids.append(next_token)

            # Decode token baru aja (bukan seluruh sequence)
            token_str = self.tokenizer.decode([next_token], skip_special_tokens=True)
            yield token_str

    @torch.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 200,
        **kwargs,
    ) -> list[str]:
        """Generate multiple prompts sekaligus (parallel)."""
        return [self.generate(p, max_new_tokens, **kwargs) for p in prompts]
````

## File: inference/sampler.py
````python
# inference/sampler.py
import torch
import torch.nn.functional as F
from typing import Optional


class Sampler:
    """
    Kumpulan strategi sampling dari logits → next token.
    
    Analoginya ke Laravel:
    - Greedy  = ambil value tertinggi langsung (array_key_first)
    - Top-K   = filter 5 kandidat terbaik, random dari situ
    - Top-P   = filter sampai cumulative prob > threshold
    - Temp    = "seberapa random" outputnya
    """

    @staticmethod
    def greedy(logits: torch.Tensor) -> int:
        """Selalu ambil token dengan prob tertinggi. Deterministic."""
        return logits.argmax(dim=-1).item()

    @staticmethod
    def temperature(logits: torch.Tensor, temp: float) -> torch.Tensor:
        """
        Scale logits sebelum softmax.
        temp < 1.0 → lebih fokus / repetitif
        temp > 1.0 → lebih random / kreatif
        temp = 1.0 → normal
        """
        if temp <= 0:
            raise ValueError("Temperature harus > 0")
        return logits / temp

    @staticmethod
    def top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
        """
        Buang semua token kecuali K terbesar.
        Set token di luar top-K ke -inf (jadi prob=0 setelah softmax).
        """
        if k <= 0:
            return logits
        k = min(k, logits.size(-1))
        values, _ = torch.topk(logits, k)
        threshold = values[..., -1, None]  # nilai terkecil dari top-K
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
        """
        Nucleus sampling: ambil token sampai cumulative prob >= p.
        Lebih adaptif dari top-K karena jumlah token bisa beda tiap step.
        """
        if p >= 1.0:
            return logits

        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)

        # Tandai token yang bikin cumulative prob > p
        remove_mask = cumulative - sorted_probs > p
        sorted_probs[remove_mask] = 0.0

        # Kembalikan ke urutan asli
        filtered = torch.zeros_like(probs)
        filtered.scatter_(-1, sorted_indices, sorted_probs)

        # Convert balik ke logits space
        filtered = filtered.log().clamp(min=-1e9)
        return filtered

    @staticmethod
    def repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list[int],
        penalty: float = 1.3
    ) -> torch.Tensor:
        """
        Kurangi probabilitas token yang udah sering muncul.
        Cegah model ngulang-ngulang kata yang sama.
        penalty > 1.0 = makin dihukum kalau udah pernah muncul.
        """
        if penalty == 1.0 or not generated_ids:
            return logits

        for token_id in set(generated_ids):
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
        return logits

    @classmethod
    def sample(
        cls,
        logits: torch.Tensor,       # (vocab_size,)
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        penalty: float = 1.0,
        generated_ids: list = None,
        do_sample: bool = True,
    ) -> int:
        """
        Pipeline lengkap: logits → 1 token ID.
        Urutan: temp → rep penalty → top-k → top-p → sample
        """
        logits = logits.clone().float()

        if not do_sample:
            return cls.greedy(logits)

        # 1. Repetition penalty
        if penalty != 1.0 and generated_ids:
            logits = cls.repetition_penalty(logits, generated_ids, penalty)

        # 2. Temperature
        logits = cls.temperature(logits, temperature)

        # 3. Top-K filter
        logits = cls.top_k(logits, top_k)

        # 4. Top-P (nucleus) filter
        logits = cls.top_p(logits, top_p)

        # 5. Sample dari distribusi
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
````

## File: lora/__init__.py
````python

````

## File: lora/config.py
````python
# lora/config.py
from dataclasses import dataclass, field
from typing import List


@dataclass
class LoRAConfig:
    # ── LoRA core ─────────────────────────────────────────
    rank: int           = 8       # r: makin gede makin expressive, makin berat
    alpha: float        = 16.0    # scaling: alpha/rank = scaling factor
    dropout: float      = 0.05    # regularization

    # Layer mana yang di-inject LoRA
    # Buat SSM: target projection layers
    target_modules: List[str] = field(default_factory=lambda: [
        "in_proj",     # input projection
        "out_proj",    # output projection
        "x_proj",      # SSM x projection
        "dt_proj",     # delta projection
    ])

    # ── Training ──────────────────────────────────────────
    learning_rate:  float = 2e-4   # LoRA lr bisa lebih gede dari full finetune
    max_steps:      int   = 5_000
    batch_size:     int   = 4
    grad_accum:     int   = 8      # effective batch = 32
    warmup_steps:   int   = 100
    max_seq_len:    int   = 512
    weight_decay:   float = 0.01

    # ── Dataset ───────────────────────────────────────────
    dataset_name:   str   = "HuggingFaceH4/ultrachat_200k"
    dataset_split:  str   = "train_sft"
    max_samples:    int   = 50_000  # ambil sebagian dulu

    # ── Save ──────────────────────────────────────────────
    save_dir:       str   = "./checkpoints/lora"
    save_every:     int   = 500
    log_interval:   int   = 10

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank
````

## File: lora/dataset.py
````python
# lora/dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.tokenizer import MultilingualTokenizer
from typing import List, Dict


# ── Dataset yang recommended buat instruction following ───
RECOMMENDED_DATASETS = {
    "ultrachat":    "HuggingFaceH4/ultrachat_200k",       # 200k multi-turn chat
    "alpaca_id":    "indonlp/indonesian-alpaca",            # Indo instruction
    "oasst":        "OpenAssistant/oasst2",                 # multilingual RLHF
    "dolly":        "databricks/databricks-dolly-15k",      # 15k diverse tasks
    "flan":         "Muennighoff/flan",                     # 1.8M instruction
}


def format_instruction(example: Dict, dataset_name: str) -> str:
    """
    Convert berbagai format dataset → format chat model lo.

    Format output:
    <|system|>...<|end_turn|>
    <|user|>...<|end_turn|>
    <|assistant|>...<|end_turn|>
    """
    if "ultrachat" in dataset_name:
        # Format: {"messages": [{"role": ..., "content": ...}]}
        messages = example.get("messages", [])
        parts    = ["<|system|>Kamu adalah asisten yang helpful.<|end_turn|>"]
        for msg in messages:
            role    = msg["role"]
            content = msg["content"].strip()
            if role in ("user", "assistant"):
                parts.append(f"<|{role}|>{content}<|end_turn|>")
        return "\n".join(parts)

    elif "alpaca" in dataset_name or "dolly" in dataset_name:
        # Format: {"instruction": ..., "input": ..., "output": ...}
        instruction = example.get("instruction", "").strip()
        inp         = example.get("input", "").strip()
        output      = example.get("output", "").strip()
        user_msg    = f"{instruction}\n{inp}".strip() if inp else instruction
        return (
            f"<|system|>Kamu adalah asisten yang helpful.<|end_turn|>\n"
            f"<|user|>{user_msg}<|end_turn|>\n"
            f"<|assistant|>{output}<|end_turn|>"
        )

    elif "oasst" in dataset_name:
        # Format: {"role": ..., "text": ...}
        role = example.get("role", "user")
        text = example.get("text", "").strip()
        return f"<|{role}|>{text}<|end_turn|>"

    else:
        # Fallback: treat sebagai plain text
        return example.get("text", str(example))


class InstructionDataset(Dataset):
    """
    Dataset untuk instruction fine-tuning.
    
    Loss mask: hanya hitung loss di bagian ASSISTANT,
    bukan di system/user prompt — ini kunci supaya model
    belajar nge-generate response, bukan ngulang prompt.
    """

    IGNORE_INDEX = -100   # index ini di-ignore sama CrossEntropyLoss

    def __init__(
        self,
        dataset_name: str,
        tokenizer: MultilingualTokenizer,
        split: str      = "train",
        max_seq_len: int = 512,
        max_samples: int = 50_000,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        print(f"📥 Loading dataset: {dataset_name}")
        raw = load_dataset(dataset_name, split=split, streaming=False)

        if max_samples and len(raw) > max_samples:
            raw = raw.select(range(max_samples))

        print(f"📊 Processing {len(raw):,} examples...")
        self.examples = self._process(raw, dataset_name)
        print(f"✅ Dataset ready: {len(self.examples):,} valid examples")

    def _process(self, raw_dataset, dataset_name: str) -> List[Dict]:
        examples = []

        for example in raw_dataset:
            text = format_instruction(example, dataset_name)
            if not text or len(text) < 20:
                continue

            # Tokenize full text
            input_ids = self.tokenizer.encode(
                text, add_bos=True, add_eos=True
            )

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]

            if len(input_ids) < 10:
                continue

            # Build loss mask
            labels = self._build_labels(input_ids, text)

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        return examples

    def _build_labels(self, input_ids: List[int], text: str) -> List[int]:
        """
        Buat labels dengan masking:
        - system / user tokens → IGNORE_INDEX (-100)
        - assistant tokens     → actual token id (dihitung loss-nya)

        Ini yang bikin model fokus belajar ngejawab,
        bukan belajar nge-repeat pertanyaan.
        """
        labels = list(input_ids)   # copy

        tok = self.tokenizer
        assistant_id = tok.special_tokens.get("<|assistant|>")
        end_turn_id  = tok.special_tokens.get("<|end_turn|>")

        in_assistant = False
        for i, token_id in enumerate(input_ids):
            if token_id == assistant_id:
                in_assistant = True
                labels[i] = self.IGNORE_INDEX   # mask token <|assistant|> itu sendiri
                continue
            if token_id == end_turn_id and in_assistant:
                in_assistant = False

            if not in_assistant:
                labels[i] = self.IGNORE_INDEX   # mask system & user

        return labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(ex["labels"],    dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad batch ke length yang sama."""
    max_len    = max(b["input_ids"].size(0) for b in batch)
    input_ids  = []
    labels     = []
    attn_masks = []

    for b in batch:
        seq_len = b["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([b["input_ids"],
                       torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attn_masks.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids":      torch.stack(input_ids),
        "labels":         torch.stack(labels),
        "attention_mask": torch.stack(attn_masks),
    }
````

## File: lora/layer.py
````python
# lora/layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALinear(nn.Module):
    """
    Drop-in replacement untuk nn.Linear dengan LoRA adapter.

    Forward:
        y = x @ W.T               ← frozen original weight
          + x @ A.T @ B.T * scale ← trainable LoRA path
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int   = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank         = rank
        self.scaling      = alpha / rank

        # ── Frozen original weight ────────────────────────
        self.weight = original_linear.weight  # reference, bukan copy
        self.bias   = original_linear.bias
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # ── Trainable LoRA matrices ───────────────────────
        # A: init dengan kaiming (bukan nol — biar ada gradient dari awal)
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features)
        )
        # B: init dengan nol (output LoRA = 0 di awal, sama kayak pretrained)
        self.lora_B = nn.Parameter(
            torch.zeros(self.out_features, rank)
        )

        self.lora_dropout = nn.Dropout(dropout)

        # Init A dengan kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        # Flag: apakah LoRA aktif
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen path
        base_out = F.linear(x, self.weight, self.bias)

        if not self.enabled or self.rank == 0:
            return base_out

        # LoRA path: x → dropout → A → B → scale
        lora_out = (
            self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        ) * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA ke weight asli → hapus overhead inference.
        Panggil ini setelah training selesai.
        W_merged = W + (B @ A) * scaling
        """
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.bias is not None)
        merged.weight.data = (
            self.weight.data +
            (self.lora_B @ self.lora_A) * self.scaling
        )
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"rank={self.rank}, scaling={self.scaling:.2f}")
````

## File: lora/merge.py
````python

````

## File: lora/model.py
````python
# lora/model.py
import torch
import torch.nn as nn
from typing import Dict, List
from .layer  import LoRALinear
from .config import LoRAConfig


class LoRAModel(nn.Module):
    """
    Wrapper yang inject LoRA ke MambaLM.

    Cara kerja:
    1. Freeze SEMUA weight base model
    2. Replace target Linear layers dengan LoRALinear
    3. Hanya LoRA params yang di-train
    """

    def __init__(self, base_model: nn.Module, config: LoRAConfig):
        super().__init__()
        self.base_model = base_model
        self.config     = config
        self.lora_layers: Dict[str, LoRALinear] = {}

        # Step 1: Freeze semua params
        self._freeze_base()

        # Step 2: Inject LoRA
        self._inject_lora()

        # Step 3: Print summary
        self._print_summary()

    def _freeze_base(self):
        """Freeze semua parameter base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False

    def _inject_lora(self):
        """
        Ganti target Linear layers dengan LoRALinear.
        Traversal rekursif kayak tree traversal.
        """
        def _replace(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name

                # Cek apakah nama layer masuk target
                is_target = any(
                    t in name for t in self.config.target_modules
                )

                if is_target and isinstance(child, nn.Linear):
                    # Replace dengan LoRALinear
                    lora_layer = LoRALinear(
                        child,
                        rank=self.config.rank,
                        alpha=self.config.alpha,
                        dropout=self.config.dropout,
                    )
                    setattr(module, name, lora_layer)
                    self.lora_layers[full_name] = lora_layer
                else:
                    _replace(child, full_name)

        _replace(self.base_model)

    def _print_summary(self):
        total_params    = sum(p.numel() for p in self.parameters())
        trainable       = sum(p.numel() for p in self.parameters()
                              if p.requires_grad)
        pct             = 100 * trainable / total_params

        print(f"\n{'='*50}")
        print(f"LoRA Model Summary")
        print(f"{'='*50}")
        print(f"Base model params : {total_params - trainable:>12,}")
        print(f"LoRA params       : {trainable:>12,}  ({pct:.2f}%)")
        print(f"Total params      : {total_params:>12,}")
        print(f"LoRA layers       : {len(self.lora_layers)}")
        print(f"Rank / Alpha      : {self.config.rank} / {self.config.alpha}")
        print(f"{'='*50}\n")
        for name in self.lora_layers:
            print(f"  ✅ {name}")
        print()

    def forward(self, input_ids, targets=None):
        return self.base_model(input_ids, targets)

    # ── Save & Load ───────────────────────────────────────

    def save_lora(self, path: str):
        """Simpan HANYA LoRA weights — kecil banget (< 50MB)."""
        lora_state = {
            name: {
                "lora_A": layer.lora_A.data,
                "lora_B": layer.lora_B.data,
            }
            for name, layer in self.lora_layers.items()
        }
        torch.save({"lora_state": lora_state, "config": self.config}, path)
        size_mb = sum(
            v["lora_A"].nelement() * v["lora_A"].element_size() +
            v["lora_B"].nelement() * v["lora_B"].element_size()
            for v in lora_state.values()
        ) / 1e6
        print(f"💾 LoRA saved: {path} ({size_mb:.1f}MB)")

    def load_lora(self, path: str):
        """Load LoRA weights ke model yang udah ada."""
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        lora_state = ckpt["lora_state"]

        for name, weights in lora_state.items():
            if name in self.lora_layers:
                self.lora_layers[name].lora_A.data = weights["lora_A"]
                self.lora_layers[name].lora_B.data = weights["lora_B"]
            else:
                print(f"⚠️  Layer not found: {name}")

        print(f"✅ LoRA loaded from {path}")

    def merge_and_export(self, save_path: str):
        """
        Merge LoRA → base model, export sebagai model biasa.
        Hasilnya bisa di-deploy tanpa LoRA overhead.
        """
        import copy
        merged_model = copy.deepcopy(self.base_model)

        def _merge(module, ref_module, prefix=""):
            for name, child in ref_module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if full_name in self.lora_layers:
                    merged_linear = self.lora_layers[full_name].merge_weights()
                    setattr(module, name, merged_linear)
                else:
                    _merge(
                        getattr(module, name),
                        child, full_name
                    )

        _merge(merged_model, self.base_model)
        torch.save(merged_model.state_dict(), save_path)
        print(f"✅ Merged model saved: {save_path}")
        return merged_model
````

## File: lora/run_lora.py
````python
# run_lora.py
from optimization.cpu.threading  import configure_cpu
from optimization.cpu.memory     import load_model_efficient
from config.model_config         import MambaConfig
from model.mamba_model           import MambaLM
from tokenizer.tokenizer         import MultilingualTokenizer
from lora.config                 import LoRAConfig
from lora.model                  import LoRAModel
from lora.dataset                import InstructionDataset
from lora.trainer                import LoRATrainer


def main():
    configure_cpu(n_cores=2)

    # ── 1. Load base model ────────────────────────────────
    print("📦 Loading base model...")
    model_config = MambaConfig(vocab_size=100277, d_model=512, n_layers=12)
    base_model   = load_model_efficient(
        MambaLM, model_config, "./checkpoints/best_model.pt"
    )

    # ── 2. Setup LoRA ─────────────────────────────────────
    lora_config = LoRAConfig(
        rank            = 8,
        alpha           = 16.0,
        target_modules  = ["in_proj", "out_proj", "x_proj", "dt_proj"],
        learning_rate   = 2e-4,
        max_steps       = 5_000,
        batch_size      = 4,
        grad_accum      = 8,
        max_seq_len     = 512,

        # Dataset — pilih salah satu dari RECOMMENDED_DATASETS
        dataset_name    = "HuggingFaceH4/ultrachat_200k",
        dataset_split   = "train_sft",
        max_samples     = 20_000,   # mulai kecil dulu di VPS

        save_dir        = "./checkpoints/lora",
        save_every      = 500,
    )

    lora_model = LoRAModel(base_model, lora_config)

    # ── 3. Load dataset ───────────────────────────────────
    tok     = MultilingualTokenizer()
    dataset = InstructionDataset(
        dataset_name = lora_config.dataset_name,
        tokenizer    = tok,
        split        = lora_config.dataset_split,
        max_seq_len  = lora_config.max_seq_len,
        max_samples  = lora_config.max_samples,
    )

    # ── 4. Train ──────────────────────────────────────────
    trainer = LoRATrainer(lora_model, lora_config, tok)
    trainer.train(dataset)

    # ── 5. Merge & export ─────────────────────────────────
    print("\n🔀 Merging LoRA into base model...")
    lora_model.merge_and_export("./checkpoints/lora/model_merged.pt")
    print("🎉 Done! Model siap di-deploy.")


if __name__ == "__main__":
    main()
````

## File: lora/trainer.py
````python
# lora/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from functools import partial
from pathlib import Path

from .config  import LoRAConfig
from .model   import LoRAModel
from .dataset import InstructionDataset, collate_fn
from training.optimizer import CosineScheduler
from training.logger    import TrainingLogger


class LoRATrainer:
    def __init__(
        self,
        lora_model: LoRAModel,
        config: LoRAConfig,
        tokenizer,
    ):
        self.model     = lora_model
        self.config    = config
        self.tokenizer = tokenizer
        self.device    = "cpu"   # VPS lo CPU only
        self.model.to(self.device)

        # Hanya optimize LoRA params
        lora_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
        )
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            max_lr=config.learning_rate,
            min_lr=config.learning_rate / 10,
        )
        self.logger = TrainingLogger(log_interval=config.log_interval)

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    def train(self, dataset: InstructionDataset):
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, pad_id=self.tokenizer.pad_id),
            num_workers=1,
            pin_memory=False,
        )

        print(f"\n🚀 LoRA Training")
        print(f"   Dataset   : {len(dataset):,} examples")
        print(f"   Max steps : {self.config.max_steps:,}")
        print(f"   Batch size: {self.config.batch_size} × {self.config.grad_accum} accum")
        print(f"   Eff. batch: {self.config.batch_size * self.config.grad_accum}\n")

        self.model.train()
        step = 0
        self.optimizer.zero_grad()

        while step < self.config.max_steps:
            for batch in loader:
                if step >= self.config.max_steps:
                    break

                input_ids = batch["input_ids"].to(self.device)
                labels    = batch["labels"].to(self.device)

                # Forward
                logits, _ = self.model(input_ids)

                # Loss — hanya di posisi yang bukan -100
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,    # skip system/user tokens
                    label_smoothing=0.1,  # regularization ringan
                )
                loss = loss / self.config.grad_accum
                loss.backward()

                if (step + 1) % self.config.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr = self.scheduler.step()

                    self.logger.log(step, loss.item() * self.config.grad_accum, lr)

                    if step > 0 and step % self.config.save_every == 0:
                        self._save(step, loss.item())

                step += 1

        # Final save
        self._save(step, loss.item())
        print("\n✅ LoRA training complete!")

    def _save(self, step: int, loss: float):
        path = f"{self.config.save_dir}/lora_step_{step:06d}.pt"
        self.model.save_lora(path)
        print(f"💾 Saved: {path} | loss={loss:.4f}")
````

## File: main.py
````python
# main.py
import torch
from config.model_config   import MambaConfig
from model.mamba_model     import MambaLM
from tokenizer.tokenizer   import MultilingualTokenizer
from training.dataset      import TextDataset
from training.trainer      import Trainer

# ── Config ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Model
    "vocab_size":    100277,
    "d_model":       512,
    "n_layers":      12,

    # Training
    "max_steps":     100_000,
    "batch_size":    8,
    "max_seq_len":   1024,
    "grad_accum_steps": 4,      # effective batch = 8 * 4 = 32

    # Optimizer
    "max_lr":        3e-4,
    "min_lr":        3e-5,
    "warmup_steps":  2_000,
    "weight_decay":  0.1,
    "grad_clip":     1.0,

    # Logging & saving
    "log_interval":  10,
    "save_every":    500,
    "checkpoint_dir": "./checkpoints",
    "num_workers":   2,
}

def main():
    # 1. Tokenizer
    tok = MultilingualTokenizer()

    # 2. Dataset — ganti dengan corpus lo
    sample_texts = [
        "Ini adalah contoh teks bahasa Indonesia untuk training.",
        "This is a sample English text for training the model.",
        "def hello():\n    print('Hello, World!')",
        # ... load dari file/dataset lo
    ]
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=tok,
        max_seq_len=TRAIN_CONFIG["max_seq_len"]
    )

    # 3. Model
    model_config = MambaConfig(
        vocab_size=TRAIN_CONFIG["vocab_size"],
        d_model=TRAIN_CONFIG["d_model"],
        n_layers=TRAIN_CONFIG["n_layers"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
    )
    model = MambaLM(model_config)

    # 4. Trainer
    trainer = Trainer(model, TRAIN_CONFIG)
    trainer.train(dataset, resume=True)

if __name__ == "__main__":
    main()
````

## File: model/mamba_model.py
````python
# model/mamba_model.py
import torch.nn as nn

class MambaLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Stack N SSM blocks
        self.layers = nn.ModuleList([
            SSMBlock(config) for _ in range(config.n_layers)
        ])

        self.norm_f = nn.LayerNorm(config.d_model)

        # LM Head: project ke vocab (weight tying optional)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying (hemat parameter, trik umum)
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, targets=None):
        x = self.embedding(input_ids)  # (B, L, d_model)

        for layer in self.layers:
            x = layer(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)  # (B, L, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss
````

## File: model/ssm_block.py
````python
# model/ssm_block.py
import torch.nn as nn

class SSMBlock(nn.Module):
    """
    Full Mamba block dengan gating mechanism.
    Analoginya ke Laravel: ini satu 'middleware' dalam pipeline.
    """
    def __init__(self, config):
        super().__init__()
        d_inner = config.d_model * config.expand

        self.norm = nn.LayerNorm(config.d_model)

        # Input projection: split jadi 2 branch
        self.in_proj = nn.Linear(config.d_model, d_inner * 2, bias=False)

        # Conv lokal untuk locality bias
        self.conv1d = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=d_inner  # depthwise conv
        )

        self.ssm = SSMCore(config)

        # Output projection
        self.out_proj = nn.Linear(d_inner, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        # Split 2 branch: SSM branch & gate branch
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)  # (B, L, d_inner) each

        # Conv lokal (B, L, D) → (B, D, L) → conv → (B, L, D)
        x_conv = self.conv1d(x_branch.transpose(1, 2))
        x_conv = x_conv[:, :, :x_branch.size(1)].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM
        y = self.ssm(x_conv)

        # Gating: multiply dengan sigmoid branch
        y = y * F.silu(z_gate)

        # Output projection + residual
        out = self.out_proj(y)
        return self.dropout(out) + residual
````

## File: model/ssm_core.py
````python
# model/ssm_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SSMCore(nn.Module):
    """
    Implementasi discrete SSM:
    h_t = A * h_{t-1} + B * x_t
    y_t = C * h_t
    """
    def __init__(self, config):
        super().__init__()
        self.d_model = config.d_model
        self.d_state = config.d_state  # N
        d_inner = config.d_model * config.expand

        # Learnable matrices
        # A: state transition (d_inner, d_state)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, config.d_state + 1)
            .float().unsqueeze(0).repeat(d_inner, 1))
        )

        # D: skip connection (residual)
        self.D = nn.Parameter(torch.ones(d_inner))

        # Projections untuk B, C, delta (input-dependent = selective!)
        dt_rank = max(1, d_inner // 16)
        self.x_proj = nn.Linear(d_inner, dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

    def forward(self, x):
        # x shape: (batch, seq_len, d_inner)
        B, L, D = x.shape
        d_state = self.d_state

        # Compute A (always negative untuk stability)
        A = -torch.exp(self.A_log.float())  # (D, N)

        # Compute input-dependent B, C, delta (ini yg bikin "selective")
        x_proj = self.x_proj(x)  # (B, L, dt_rank + 2*N)
        dt_rank = self.dt_proj.in_features
        delta, B_mat, C_mat = x_proj.split([dt_rank, d_state, d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta))  # (B, L, D) — step size

        # Discretize: ZOH (Zero-Order Hold)
        # A_bar = exp(delta * A)
        # B_bar = (A_bar - I) * inv(A) * B ≈ delta * B (simplified)
        dA = torch.exp(delta.unsqueeze(-1) * A)          # (B, L, D, N)
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)   # (B, L, D, N)

        # Selective scan — recurrent state update
        h = torch.zeros(B, D, d_state, device=x.device)
        ys = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y = (h * C_mat[:, t].unsqueeze(-2)).sum(-1)  # (B, D)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        y = y + x * self.D          # skip connection

        return y
````

## File: optimization/benchmark.py
````python
# optimization/benchmark.py
import torch
import time
import psutil
from dataclasses import dataclass


@dataclass
class BenchResult:
    tokens_per_sec: float
    latency_ms:     float
    ram_mb:         float
    model_size_mb:  float


def benchmark(generator, prompt: str = "Halo, apa kabar?",
              n_tokens: int = 50, n_runs: int = 3) -> BenchResult:
    """Ukur kecepatan dan RAM usage."""

    ram_before = psutil.Process().memory_info().rss / 1e6

    latencies = []
    for i in range(n_runs):
        start = time.perf_counter()
        out   = generator.generate(prompt, max_new_tokens=n_tokens)
        end   = time.perf_counter()
        latencies.append(end - start)
        print(f"Run {i+1}: {n_tokens/(end-start):.1f} tok/s | '{out[:40]}...'")

    ram_after = psutil.Process().memory_info().rss / 1e6
    avg_lat   = sum(latencies) / n_runs

    result = BenchResult(
        tokens_per_sec = n_tokens / avg_lat,
        latency_ms     = avg_lat * 1000,
        ram_mb         = ram_after,
        model_size_mb  = ram_after - ram_before,
    )

    print(f"\n{'='*40}")
    print(f"⚡ {result.tokens_per_sec:.1f} tokens/sec")
    print(f"⏱️  {result.latency_ms:.0f}ms per {n_tokens} tokens")
    print(f"🧠 RAM: {result.ram_mb:.0f}MB")
    print(f"{'='*40}")
    return result
````

## File: optimization/cpu/threading.py
````python
# optimization/cpu/threading.py
import torch
import os
import psutil


def configure_cpu(n_cores: int = 2):
    """
    Setup optimal CPU config buat VPS 2 core.
    Harus dipanggil SEBELUM import model.
    """

    # PyTorch threads
    torch.set_num_threads(n_cores)
    torch.set_num_interop_threads(1)

    # OpenMP (used by PyTorch internals)
    os.environ["OMP_NUM_THREADS"]        = str(n_cores)
    os.environ["MKL_NUM_THREADS"]        = str(n_cores)
    os.environ["OPENBLAS_NUM_THREADS"]   = str(n_cores)

    # Disable unnecessary torch features di CPU
    torch.backends.cudnn.enabled = False

    # Enable CPU optimizations
    # AVX2/AVX512 auto-detected, tapi bisa di-force:
    os.environ["PYTORCH_JIT"] = "1"

    print(f"✅ CPU configured: {n_cores} threads")
    print(f"   Available cores: {psutil.cpu_count()}")
    print(f"   Available RAM  : {psutil.virtual_memory().available / 1e9:.1f}GB")


# optimization/cpu/memory.py
import torch
import gc
import psutil
from contextlib import contextmanager


class MemoryManager:
    """
    Kelola RAM ketat buat VPS 4GB.
    """

    RAM_LIMIT_GB = 3.0   # batas safety sebelum OOM

    @staticmethod
    def current_usage_gb() -> float:
        return psutil.Process().memory_info().rss / 1e9

    @staticmethod
    def available_gb() -> float:
        return psutil.virtual_memory().available / 1e9

    @classmethod
    def check(cls, label: str = ""):
        used = cls.current_usage_gb()
        avail = cls.available_gb()
        print(f"🧠 RAM [{label}]: used={used:.2f}GB | avail={avail:.2f}GB")
        if used > cls.RAM_LIMIT_GB:
            print("⚠️  WARNING: RAM usage tinggi, clearing cache...")
            cls.clear()

    @staticmethod
    def clear():
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    @contextmanager
    def track(self, label: str):
        """Context manager buat track RAM usage per operasi."""
        before = self.current_usage_gb()
        yield
        after = self.current_usage_gb()
        delta = after - before
        print(f"📊 [{label}] RAM delta: {delta:+.2f}GB (total: {after:.2f}GB)")


def load_model_efficient(model_class, config, checkpoint_path: str):
    """
    Load model dengan RAM footprint minimal.
    Pakai map_location='cpu' + lazy loading.
    """
    mem = MemoryManager()

    with mem.track("model_load"):
        # Init model dengan empty weights dulu (hemat RAM saat load)
        with torch.device("meta"):
            model = model_class(config)

        # Load weights langsung ke CPU tanpa double memory
        state_dict = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,    # security + hemat RAM
        )
        model.load_state_dict(state_dict, assign=True)

    model.eval()
    mem.check("after_load")
    return model
````

## File: optimization/kvcache.py
````python
# optimization/kvcache.py
import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SSMCache:
    """
    Cache SSM hidden state antar token generation.
    
    Tanpa cache: tiap token, lo recompute SELURUH sequence dari awal.
    Dengan cache: lo cuma compute 1 token baru, state-nya dilanjutin.
    
    Ini yang bikin generation 10-50x lebih cepat!
    """
    # Hidden state SSM per layer: (n_layers, batch, d_inner, d_state)
    h: Optional[torch.Tensor] = None
    # Conv state per layer: (n_layers, batch, d_inner, d_conv)
    conv: Optional[torch.Tensor] = None

    def is_empty(self) -> bool:
        return self.h is None


class CachedSSMBlock(nn.Module):
    """
    SSM Block yang support incremental decode dengan cache.
    Saat prefill (prompt): jalanin full sequence sekali.
    Saat decode (generate): jalanin 1 token, update cache.
    """

    def forward(
        self,
        x: torch.Tensor,              # (B, L, d_model)
        cache: Optional[SSMCache] = None,
        use_cache: bool = False,
    ):
        is_decoding = use_cache and cache and not cache.is_empty()

        if is_decoding:
            # Decode mode: x = (B, 1, d_model) — 1 token aja
            return self._decode_step(x, cache)
        else:
            # Prefill mode: proses full prompt
            return self._prefill(x, cache, use_cache)

    def _prefill(self, x, cache, use_cache):
        """Full forward pass untuk prompt."""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)

        # Conv — simpan state akhir untuk decode
        x_conv_t = x_branch.transpose(1, 2)   # (B, D, L)
        x_conv   = self.conv1d(x_conv_t)
        x_conv   = x_conv[:, :, :x_branch.size(1)].transpose(1, 2)
        x_conv   = F.silu(x_conv)

        y = self.ssm(x_conv)
        y = y * F.silu(z_gate)
        out = self.out_proj(y) + residual

        # Simpan cache
        if use_cache and cache is not None:
            # Simpan conv state (D, d_conv terakhir)
            cache.conv = x_conv_t[:, :, -self.conv1d.kernel_size[0]:]
            # Simpan SSM hidden state terakhir
            cache.h = self.ssm._last_h   # perlu diexpose dari SSMCore

        return out

    def _decode_step(self, x, cache):
        """Single token decode — cuma hitung 1 step."""
        residual = x
        x = self.norm(x)
        xz = self.in_proj(x)
        x_branch, z_gate = xz.chunk(2, dim=-1)   # (B, 1, D)

        # Conv dengan cache: geser window, append token baru
        # cache.conv: (B, D, d_conv-1)
        x_t   = x_branch.transpose(1, 2)          # (B, D, 1)
        conv_input = torch.cat([cache.conv, x_t], dim=-1)  # (B, D, d_conv)
        x_conv = (conv_input * self.conv1d.weight.squeeze(1)).sum(-1, keepdim=True)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias.unsqueeze(-1)
        x_conv = F.silu(x_conv.transpose(1, 2))   # (B, 1, D)

        # Update conv cache
        cache.conv = conv_input[:, :, 1:]          # geser 1

        # SSM single step
        y, new_h  = self.ssm.step(x_conv, cache.h)
        cache.h   = new_h

        y   = y * F.silu(z_gate)
        out = self.out_proj(y) + residual
        return out
````

## File: optimization/onnx/export.py
````python
# optimization/onnx/export.py
import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path


class ONNXExporter:
    """
    Export model ke ONNX → jalanin via ONNXRuntime.
    ONNXRuntime di CPU bisa 2-3x lebih cepat dari PyTorch CPU.
    Ini yang dipake production buat CPU deployment.
    """

    def __init__(self, model, save_dir: str = "./checkpoints/onnx"):
        self.model    = model.cpu().eval()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        seq_len: int     = 64,
        batch_size: int  = 1,
        opset: int       = 17,          # ONNX opset terbaru
    ) -> str:
        print("📦 Exporting to ONNX...")

        onnx_path = str(self.save_dir / "model.onnx")
        dummy_input = torch.randint(0, 1000, (batch_size, seq_len))

        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits":    {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
            do_constant_folding=True,   # fold constant ops = lebih cepat
        )

        # Verify
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX export valid: {onnx_path}")

        return onnx_path

    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimasi graph ONNX:
        - Fuse operators
        - Eliminate redundant nodes
        - Constant folding
        """
        from onnxruntime.transformers import optimizer as ort_optimizer

        opt_path = onnx_path.replace(".onnx", "_optimized.onnx")

        opt_model = ort_optimizer.optimize_model(
            onnx_path,
            model_type="bert",        # paling dekat sama LLM
            num_heads=8,
            hidden_size=512,
            optimization_level=99,    # max optimization
        )
        opt_model.save_model_to_file(opt_path)
        print(f"✅ Optimized ONNX: {opt_path}")
        return opt_path

    def build_session(self, onnx_path: str) -> ort.InferenceSession:
        """
        Build ONNXRuntime session dengan CPU optimizations.
        """
        opts = ort.SessionOptions()

        # Threading — sesuai 2 core VPS lo
        opts.intra_op_num_threads = 2   # thread per operator
        opts.inter_op_num_threads = 1   # thread antar operator
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Graph optimization level (max)
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Enable memory pattern reuse
        opts.enable_mem_pattern    = True
        opts.enable_mem_reuse      = True
        opts.enable_cpu_mem_arena  = True

        session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        print("✅ ONNX Runtime session ready")
        return session


class ONNXGenerator:
    """
    Generator yang pakai ONNX session instead of PyTorch.
    Drop-in replacement buat Generator class sebelumnya.
    """

    def __init__(self, session: ort.InferenceSession, tokenizer):
        self.session   = session
        self.tokenizer = tokenizer

    def generate(
        self,
        prompt: str,
        max_new_tokens: int  = 200,
        temperature: float   = 0.8,
        top_k: int           = 50,
        top_p: float         = 0.9,
    ) -> str:
        from inference.sampler import Sampler

        input_ids     = self.tokenizer.encode(prompt, add_bos=True)
        generated_ids = list(input_ids)

        for _ in range(max_new_tokens):
            x = np.array([generated_ids], dtype=np.int64)

            # ONNX inference — jauh lebih cepat dari PyTorch CPU
            outputs  = self.session.run(["logits"], {"input_ids": x})
            logits   = torch.tensor(outputs[0][0, -1, :])

            next_token = Sampler.sample(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                generated_ids=generated_ids,
            )

            if next_token == self.tokenizer.eos_id:
                break
            generated_ids.append(next_token)

        output_ids = generated_ids[len(input_ids):]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
````

## File: optimization/quantization/calibrate.py
````python

````

## File: optimization/quantization/quantize.py
````python
# optimization/quantization/quantize.py
import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic
from pathlib import Path
import os


class ModelQuantizer:
    """
    3 level quantization, pilih sesuai kebutuhan:
    
    INT8 Dynamic  → 4x lebih kecil, ~95% quality, paling gampang
    INT8 Static   → lebih cepat dari dynamic, perlu calibration data
    INT4 GPTQ     → 8x lebih kecil, ~90% quality, paling kecil
    """

    def __init__(self, model, save_dir: str = "./checkpoints/quantized"):
        self.model    = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # ── INT8 Dynamic (recommended buat lo) ────────────────
    def quantize_int8_dynamic(self):
        """
        Dynamic quantization: weight di-quantize, activation runtime.
        Zero calibration needed — langsung jalan.
        Cocok banget buat CPU inference.
        """
        print("⚙️  Applying INT8 Dynamic Quantization...")

        model_fp32 = self.model.cpu()
        model_fp32.eval()

        # Quantize semua layer Linear & Embedding
        quantized = quantize_dynamic(
            model_fp32,
            qconfig_spec={
                nn.Linear:    torch.quantization.default_dynamic_qconfig,
                nn.Embedding: torch.quantization.default_dynamic_qconfig,
            },
            dtype=torch.qint8,
        )

        # Hitung size reduction
        original_size = self._model_size_mb(self.model)
        quantized_size = self._model_size_mb(quantized)

        print(f"✅ INT8 Dynamic done!")
        print(f"   Original : {original_size:.1f} MB")
        print(f"   Quantized: {quantized_size:.1f} MB")
        print(f"   Reduction: {original_size/quantized_size:.1f}x smaller")

        # Save
        path = self.save_dir / "model_int8_dynamic.pt"
        torch.save(quantized.state_dict(), path)
        print(f"💾 Saved: {path}")

        return quantized

    # ── INT8 Static (lebih cepat, perlu calibration) ──────
    def quantize_int8_static(self, calibration_loader):
        """
        Static quantization: observer jalan dulu di calibration data,
        lalu scale factor di-freeze. Lebih cepat saat inference.
        """
        print("⚙️  Applying INT8 Static Quantization...")

        model = self.model.cpu().eval()

        # Setup qconfig
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")  # CPU x86

        # Fuse layers yang bisa digabung (Linear + ReLU, dsb)
        # Ini bikin inference lebih cepat karena kurang kernel call
        torch.quantization.fuse_modules(
            model,
            [["norm", "ssm"]],   # sesuaikan dengan layer lo
            inplace=True
        )

        # Prepare: insert observer
        torch.quantization.prepare(model, inplace=True)

        # Calibration: jalanin beberapa batch tanpa gradient
        print("📊 Running calibration...")
        with torch.no_grad():
            for i, (x, _) in enumerate(calibration_loader):
                model(x)
                if i >= 100:   # 100 batch cukup
                    break

        # Convert ke quantized
        torch.quantization.convert(model, inplace=True)

        original_size  = self._model_size_mb(self.model)
        quantized_size = self._model_size_mb(model)
        print(f"✅ INT8 Static done! {original_size:.1f}MB → {quantized_size:.1f}MB")

        path = self.save_dir / "model_int8_static.pt"
        torch.save(model.state_dict(), path)
        return model

    # ── INT4 via bitsandbytes (paling kecil) ──────────────
    def quantize_int4_bnb(self):
        """
        4-bit quantization via bitsandbytes.
        Butuh: pip install bitsandbytes
        Paling agresif — model bisa 8x lebih kecil.
        """
        try:
            import bitsandbytes as bnb
        except ImportError:
            print("❌ pip install bitsandbytes")
            return None

        print("⚙️  Applying INT4 Quantization (bitsandbytes)...")

        # Replace semua nn.Linear dengan 4-bit version
        def replace_linear_4bit(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    new_layer = bnb.nn.Linear4bit(
                        child.in_features,
                        child.out_features,
                        bias=child.bias is not None,
                        compute_dtype=torch.float32,   # compute dalam fp32
                        compress_statistics=True,
                        quant_type="nf4",              # NormalFloat4 — paling bagus
                    )
                    new_layer.weight = bnb.nn.Params4bit(
                        child.weight.data,
                        requires_grad=False,
                        quant_type="nf4"
                    )
                    setattr(module, name, new_layer)
                else:
                    replace_linear_4bit(child)

        model = self.model.cpu()
        replace_linear_4bit(model)

        size = self._model_size_mb(model)
        print(f"✅ INT4 NF4 done! Model size: {size:.1f}MB")

        path = self.save_dir / "model_int4.pt"
        torch.save(model.state_dict(), path)
        return model

    @staticmethod
    def _model_size_mb(model) -> float:
        total = sum(
            p.nelement() * p.element_size()
            for p in model.parameters()
        )
        return total / (1024 ** 2)
````

## File: PROJECT_CONTEXT.md
````markdown

````

## File: README.md
````markdown
---

## `README.md`

```md
# SIGER_LLM

SIGER_LLM is an experimental custom Python LLM framework for building, training, evaluating, optimizing, and serving a lightweight language model.

This project is designed as a learning and research-oriented codebase. It separates the main LLM workflow into clear modules:

- tokenizer
- model architecture
- training
- inference
- LoRA fine-tuning
- evaluation
- optimization
- quantization
- ONNX export

## Project Status

Experimental.

This project is still under active development and may change frequently.

## Main Features

- Custom tokenizer module
- Tokenizer training utilities
- Custom model architecture
- Mamba / SSM-style model modules
- Training pipeline
- Checkpoint handling
- Inference generator
- Chat interface
- API inference module
- Sampling utilities
- LoRA fine-tuning
- LoRA merge support
- Evaluation suite
- Perplexity evaluation
- Generation benchmark
- Indonesian evaluation module
- KV cache optimization
- CPU threading optimization
- ONNX export
- Quantization and calibration tools
- Tokenizer tests

## Folder Structure

```txt
SIGER_LLM/
├── checkpoints/
│   └── tokenizer/
│       └── tokenizer_config.json
├── config/
│   └── model_config.py
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks.py
│   ├── generation.py
│   ├── indo_eval.py
│   ├── perplexity.py
│   ├── report.py
│   ├── run_eval.py
│   └── runner.py
├── inference/
│   ├── __init__.py
│   ├── api.py
│   ├── chat.py
│   ├── generator.py
│   └── sampler.py
├── lora/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── layer.py
│   ├── merge.py
│   ├── model.py
│   ├── run_lora.py
│   └── trainer.py
├── model/
│   ├── mamba_model.py
│   ├── ssm_block.py
│   └── ssm_core.py
├── optimization/
│   ├── benchmark.py
│   ├── kvcache.py
│   ├── cpu/
│   │   └── threading.py
│   ├── onnx/
│   │   └── export.py
│   └── quantization/
│       ├── calibrate.py
│       └── quantize.py
├── tokenizer/
│   ├── __init__.py
│   ├── sample_texts.py
│   ├── special_tokens.py
│   ├── tokenizer.py
│   ├── trainer.py
│   ├── vocab_extender.py
│   └── tests/
│       ├── sample_texts.py
│       └── test_tokenizer.py
├── training/
│   ├── __init__.py
│   ├── checkpoint.py
│   ├── dataset.py
│   ├── logger.py
│   ├── optimizer.py
│   └── trainer.py
├── AGENTS.md
├── main.py
└── README.md
````

## File: tokenizer/__init__.py
````python

````

## File: tokenizer/special_tokens.py
````python
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

    # Structural
    "<|bos|>":           100268,   # Beginning of sequence
    "<|eos|>":           100269,   # End of sequence
    "<|sep|>":           100270,   # Separator antar segment
}

# Reverse mapping: id → token string
ID_TO_SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}

# Token yang TIDAK boleh di-split saat encoding
ALLOWED_SPECIAL_IN_TEXT = {"<|endoftext|>"}
````

## File: tokenizer/tests/sample_texts.py
````python

````

## File: tokenizer/tests/test_tokenizer.py
````python
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
````

## File: tokenizer/tokenizer.py
````python
# tokenizer/tokenizer.py
import tiktoken
import json
import os
from pathlib import Path
from typing import List, Union, Optional
from .special_tokens import SPECIAL_TOKENS, ID_TO_SPECIAL, ALLOWED_SPECIAL_IN_TEXT


class MultilingualTokenizer:
    """
    Tiktoken-based multilingual tokenizer.
    Base: cl100k_base (GPT-4 encoding) — 100k vocab, handles UTF-8 multilingual
    Extended: tambah special tokens untuk chat, lang-id, dsb.
    """

    BASE_ENCODING = "cl100k_base"  # atau "o200k_base" (GPT-4o, lebih gede)

    def __init__(self, custom_vocab_path: Optional[str] = None):
        self.special_tokens = SPECIAL_TOKENS.copy()
        self.id_to_special = ID_TO_SPECIAL.copy()

        # Load custom vocab tambahan kalau ada
        if custom_vocab_path and os.path.exists(custom_vocab_path):
            self._load_custom_vocab(custom_vocab_path)

        # Build tiktoken encoder dengan special tokens
        self._build_encoder()

        # Shortcuts buat training
        self.pad_id  = self.special_tokens["<|pad|>"]
        self.eos_id  = self.special_tokens["<|eos|>"]
        self.bos_id  = self.special_tokens["<|bos|>"]
        self.unk_id  = self.special_tokens["<|unk|>"]

    def _build_encoder(self):
        base = tiktoken.get_encoding(self.BASE_ENCODING)

        # Extend encoder base dengan special tokens lo
        self.encoder = tiktoken.Encoding(
            name="multilingual_llm",
            pat_str=base._pat_str,           # regex split pattern aslinya
            mergeable_ranks=base._mergeable_ranks,  # BPE merge rules
            special_tokens=self.special_tokens
        )

        self.vocab_size = self.encoder.n_vocab
        print(f"✅ Tokenizer ready | vocab_size={self.vocab_size}")

    # ─────────────────────────────────────────
    # ENCODE
    # ─────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,  # "id", "en", "code"
    ) -> List[int]:
        """
        Encode teks → list of token IDs.

        Args:
            text     : Input string
            add_bos  : Prepend <|bos|>
            add_eos  : Append <|eos|>
            lang     : Prepend language tag token

        Returns:
            List[int] token IDs
        """
        tokens = []

        if add_bos:
            tokens.append(self.bos_id)

        # Prepend language tag
        if lang:
            lang_token = f"<|{lang}|>"
            if lang_token in self.special_tokens:
                tokens.append(self.special_tokens[lang_token])

        # Encode teks utama
        # allowed_special: token spesial yang boleh muncul di teks input
        encoded = self.encoder.encode(
            text,
            allowed_special=ALLOWED_SPECIAL_IN_TEXT,
            disallowed_special=()  # jangan raise error kalau ketemu token aneh
        )
        tokens.extend(encoded)

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def encode_batch(
        self,
        texts: List[str],
        add_bos: bool = False,
        add_eos: bool = False,
        lang: Optional[str] = None,
    ) -> List[List[int]]:
        """Encode multiple texts sekaligus."""
        return [self.encode(t, add_bos, add_eos, lang) for t in texts]

    # ─────────────────────────────────────────
    # DECODE
    # ─────────────────────────────────────────

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode list of token IDs → string.

        Args:
            token_ids           : List of IDs
            skip_special_tokens : Kalau True, buang special tokens dari output
        """
        if skip_special_tokens:
            token_ids = [
                t for t in token_ids
                if t not in self.id_to_special
            ]

        return self.encoder.decode(token_ids)

    def decode_batch(
        self,
        batch: List[List[int]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        return [self.decode(ids, skip_special_tokens) for ids in batch]

    # ─────────────────────────────────────────
    # PADDING & TRUNCATION (buat DataLoader)
    # ─────────────────────────────────────────

    def pad_sequence(
        self,
        token_ids: List[int],
        max_length: int,
        pad_left: bool = False,     # False = pad kanan (standard)
        truncate: bool = True,
    ) -> List[int]:
        """Pad atau truncate sequence ke max_length."""
        if truncate and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]

        pad_len = max_length - len(token_ids)
        padding = [self.pad_id] * pad_len

        return padding + token_ids if pad_left else token_ids + padding

    def pad_batch(
        self,
        batch: List[List[int]],
        max_length: Optional[int] = None,
        pad_left: bool = False,
    ) -> tuple:
        """
        Pad seluruh batch ke length yang sama.
        Return: (padded_batch, attention_mask)
        """
        if max_length is None:
            max_length = max(len(ids) for ids in batch)

        padded, masks = [], []
        for ids in batch:
            original_len = min(len(ids), max_length)
            padded_ids = self.pad_sequence(ids, max_length, pad_left)
            padded.append(padded_ids)

            # Attention mask: 1 = real token, 0 = padding
            if pad_left:
                mask = [0] * (max_length - original_len) + [1] * original_len
            else:
                mask = [1] * original_len + [0] * (max_length - original_len)
            masks.append(mask)

        return padded, masks

    # ─────────────────────────────────────────
    # UTILITIES
    # ─────────────────────────────────────────

    def token_to_id(self, token: str) -> int:
        if token in self.special_tokens:
            return self.special_tokens[token]
        ids = self.encoder.encode(token, allowed_special="all")
        return ids[0] if ids else self.unk_id

    def id_to_token(self, token_id: int) -> str:
        if token_id in self.id_to_special:
            return self.id_to_special[token_id]
        return self.encoder.decode([token_id])

    def count_tokens(self, text: str) -> int:
        """Hitung jumlah token tanpa buat list penuh — efisien."""
        return len(self.encoder.encode(text, allowed_special="all"))

    def _load_custom_vocab(self, path: str):
        """Load custom tokens tambahan dari JSON."""
        with open(path) as f:
            custom = json.load(f)
        self.special_tokens.update(custom)
        self.id_to_special = {v: k for k, v in self.special_tokens.items()}
        print(f"📦 Loaded {len(custom)} custom tokens from {path}")

    def save_config(self, save_dir: str):
        """Simpan konfigurasi tokenizer."""
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        config = {
            "base_encoding": self.BASE_ENCODING,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(f"{save_dir}/tokenizer_config.json", "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"💾 Tokenizer config saved to {save_dir}/")

    def __repr__(self):
        return (
            f"MultilingualTokenizer("
            f"base={self.BASE_ENCODING}, "
            f"vocab_size={self.vocab_size}, "
            f"special_tokens={len(self.special_tokens)})"
        )
````

## File: tokenizer/trainer.py
````python

````

## File: tokenizer/vocab_extender.py
````python

````

## File: training/__init__.py
````python

````

## File: training/checkpoint.py
````python
# training/checkpoint.py
import torch
import os
import json
from pathlib import Path
from datetime import datetime


class CheckpointManager:
    """
    Save & load model checkpoint.
    Kayak git commit — tiap N step lo save state.
    """
    def __init__(self, save_dir: str, keep_last: int = 3):
        self.save_dir  = Path(save_dir)
        self.keep_last = keep_last
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.history: list[str] = []

    def save(
        self,
        model,
        optimizer,
        scheduler,
        step: int,
        loss: float,
        config: dict,
    ) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"step_{step:07d}_{timestamp}.pt"
        ckpt_path = self.save_dir / ckpt_name

        torch.save({
            "step":            step,
            "loss":            loss,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_step":  scheduler.current_step,
            "config":          config,
        }, ckpt_path)

        # Simpan metadata
        meta_path = self.save_dir / "latest.json"
        with open(meta_path, "w") as f:
            json.dump({"latest": ckpt_name, "step": step, "loss": loss}, f)

        self.history.append(str(ckpt_path))
        print(f"💾 Saved checkpoint: {ckpt_name} | loss={loss:.4f}")

        # Hapus checkpoint lama
        self._cleanup()

        return str(ckpt_path)

    def load(self, model, optimizer=None, scheduler=None, path: str = None):
        """Load checkpoint. Kalau path=None, load yang paling baru."""
        if path is None:
            meta_path = self.save_dir / "latest.json"
            if not meta_path.exists():
                print("⚠️  No checkpoint found, starting fresh.")
                return 0, float("inf")
            with open(meta_path) as f:
                meta = json.load(f)
            path = self.save_dir / meta["latest"]

        ckpt = torch.load(path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

        if optimizer and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])

        if scheduler and "scheduler_step" in ckpt:
            scheduler.current_step = ckpt["scheduler_step"]

        step = ckpt.get("step", 0)
        loss = ckpt.get("loss", float("inf"))
        print(f"✅ Loaded checkpoint: step={step} | loss={loss:.4f}")
        return step, loss

    def _cleanup(self):
        if len(self.history) > self.keep_last:
            old = self.history.pop(0)
            if os.path.exists(old):
                os.remove(old)
````

## File: training/dataset.py
````python
# training/dataset.py
import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer import MultilingualTokenizer

class TextDataset(Dataset):
    """
    Sliding window dataset untuk language modeling.
    Mirip kayak chunking teks di Laravel pagination — potong-potong.
    """
    def __init__(self, texts: list[str], tokenizer: MultilingualTokenizer,
                 max_seq_len: int = 2048, stride: int = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len  # non-overlapping default

        # Tokenize semua teks, gabung jadi satu stream
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_eos=True)
            all_ids.extend(ids)

        # Sliding window: potong jadi chunk max_seq_len + 1
        # +1 karena target = input shifted by 1
        self.chunks = []
        for i in range(0, len(all_ids) - max_seq_len, self.stride):
            chunk = all_ids[i : i + max_seq_len + 1]
            self.chunks.append(chunk)

        print(f"📚 Dataset: {len(texts)} docs → {len(self.chunks)} chunks")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Input: token[0..n-1], Target: token[1..n] (next token prediction)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y
````

## File: training/logger.py
````python
# training/logger.py
import time
from collections import deque


class TrainingLogger:
    """
    Simple logger buat track metrics.
    Bisa diextend ke WandB/TensorBoard nanti.
    """
    def __init__(self, log_interval: int = 10, window: int = 100):
        self.log_interval = log_interval
        self.loss_window  = deque(maxlen=window)  # rolling average
        self.start_time   = time.time()
        self.step_times   = deque(maxlen=50)
        self._last_time   = time.time()

    def log(self, step: int, loss: float, lr: float, tokens_per_sec: float = 0):
        self.loss_window.append(loss)
        now = time.time()
        self.step_times.append(now - self._last_time)
        self._last_time = now

        if step % self.log_interval == 0:
            avg_loss   = sum(self.loss_window) / len(self.loss_window)
            avg_step_t = sum(self.step_times) / len(self.step_times)
            elapsed    = now - self.start_time
            perplexity = min(2 ** avg_loss, 99999)  # PPL = 2^loss

            print(
                f"step={step:>7,} | "
                f"loss={loss:.4f} | "
                f"avg_loss={avg_loss:.4f} | "
                f"ppl={perplexity:.1f} | "
                f"lr={lr:.2e} | "
                f"tok/s={tokens_per_sec:,.0f} | "
                f"elapsed={elapsed/60:.1f}m"
            )

    def summary(self, total_steps: int):
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.loss_window) / max(len(self.loss_window), 1)
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Steps   : {total_steps:,}")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  PPL     : {2**avg_loss:.2f}")
        print(f"  Time    : {elapsed/3600:.2f}h")
        print(f"{'='*60}")
````

## File: training/optimizer.py
````python
# training/optimizer.py
import torch
import math
from torch.optim import AdamW

def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 0.1):
    """
    AdamW dengan weight decay selective.
    Parameter 1D (bias, layernorm) → NO weight decay.
    Parameter 2D+ (weight matrices) → weight decay.
    Ini trick standar dari GPT-2 paper.
    """
    decay_params, no_decay_params = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:  # bias, layernorm weights
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = AdamW(param_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model params: {total:,} total | {trainable:,} trainable")

    return optimizer


class CosineScheduler:
    """
    Cosine decay dengan warmup.
    Analoginya: panas dulu (warmup) → dingin pelan-pelan (cosine).
    """
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        max_lr: float = 3e-4,
        min_lr: float = 3e-5,   # biasanya 10% dari max_lr
    ):
        self.optimizer   = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps   = max_steps
        self.max_lr      = max_lr
        self.min_lr      = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        step = self.current_step

        # Phase 1: Linear warmup
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps

        # Phase 2: Cosine decay
        if step >= self.max_steps:
            return self.min_lr

        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + coeff * (self.max_lr - self.min_lr)

    def step(self):
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.current_step += 1
        return lr
````

## File: training/trainer.py
````python
# training/trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from .optimizer   import build_optimizer, CosineScheduler
from .checkpoint  import CheckpointManager
from .logger      import TrainingLogger
from .dataset     import TextDataset
from tokenizer.tokenizer import MultilingualTokenizer


class Trainer:
    def __init__(self, model, config: dict, device: str = None):
        self.model  = model
        self.config = config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        print(f"🖥️  Device: {self.device}")

        # Komponen training
        self.optimizer = build_optimizer(
            model,
            lr=config["max_lr"],
            weight_decay=config.get("weight_decay", 0.1)
        )
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup_steps=config["warmup_steps"],
            max_steps=config["max_steps"],
            max_lr=config["max_lr"],
            min_lr=config.get("min_lr", config["max_lr"] / 10),
        )
        self.ckpt_manager = CheckpointManager(
            save_dir=config["checkpoint_dir"],
            keep_last=config.get("keep_last_checkpoints", 3),
        )
        self.logger = TrainingLogger(
            log_interval=config.get("log_interval", 10)
        )

        self.grad_clip   = config.get("grad_clip", 1.0)
        self.accum_steps = config.get("grad_accum_steps", 1)  # gradient accumulation

    def _build_dataloader(self, dataset: TextDataset) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config.get("num_workers", 2),
            pin_memory=(self.device == "cuda"),
            drop_last=True,
        )

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single forward + backward step."""
        x = x.to(self.device)
        y = y.to(self.device)

        # Mixed precision (otomatis kalau CUDA)
        with torch.autocast(device_type=self.device, dtype=torch.float16,
                            enabled=(self.device == "cuda")):
            _, loss = self.model(x, targets=y)
            loss = loss / self.accum_steps  # scale buat grad accum

        loss.backward()
        return loss.item() * self.accum_steps  # return unscaled loss

    def train(
        self,
        dataset: TextDataset,
        resume: bool = True,
    ):
        """
        Main training loop.
        Analoginya ke Laravel queue worker:
        while ada job → proses → simpan state.
        """
        dataloader = self._build_dataloader(dataset)
        scaler     = torch.cuda.GradScaler(enabled=(self.device == "cuda"))

        global_step = 0
        best_loss   = float("inf")

        # Resume dari checkpoint kalau ada
        if resume:
            global_step, best_loss = self.ckpt_manager.load(
                self.model, self.optimizer, self.scheduler
            )

        self.model.train()
        max_steps = self.config["max_steps"]

        print(f"\n🚀 Training starts | max_steps={max_steps:,}")
        print(f"   batch_size={self.config['batch_size']} | "
              f"grad_accum={self.accum_steps} | "
              f"effective_batch={self.config['batch_size'] * self.accum_steps}\n")

        # ── Training Loop ──────────────────────────────────────
        self.optimizer.zero_grad()
        epoch = 0

        while global_step < max_steps:
            epoch += 1

            for batch_idx, (x, y) in enumerate(dataloader):
                if global_step >= max_steps:
                    break

                # Forward + backward
                loss = self.train_step(x, y)

                # Update weights setiap accum_steps
                if (batch_idx + 1) % self.accum_steps == 0:

                    # Gradient clipping — cegah exploding gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # LR scheduler step
                    lr = self.scheduler.step()

                    # Hitung throughput
                    tokens_per_step = (
                        self.config["batch_size"]
                        * self.config["max_seq_len"]
                        * self.accum_steps
                    )

                    # Logging
                    self.logger.log(global_step, loss, lr, tokens_per_step)

                    # Checkpoint
                    save_every = self.config.get("save_every", 500)
                    if global_step > 0 and global_step % save_every == 0:
                        self.ckpt_manager.save(
                            self.model, self.optimizer, self.scheduler,
                            step=global_step, loss=loss,
                            config=self.config,
                        )
                        if loss < best_loss:
                            best_loss = loss
                            self._save_best()

                    global_step += 1

        self.logger.summary(global_step)
        # Final save
        self.ckpt_manager.save(
            self.model, self.optimizer, self.scheduler,
            step=global_step, loss=loss, config=self.config,
        )

    def _save_best(self):
        """Simpan model terbaik terpisah."""
        best_path = f"{self.config['checkpoint_dir']}/best_model.pt"
        torch.save(self.model.state_dict(), best_path)
        print(f"🏆 Best model saved!")
````
