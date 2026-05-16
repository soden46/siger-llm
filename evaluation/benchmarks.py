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