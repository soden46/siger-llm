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