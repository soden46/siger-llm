from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chat_cli import load_model
from evaluation.lampung_lookup_eval import LampungLookupEvaluator
from inference.generator import Generator
from inference.lampung_pipeline import LampungPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SigerLM Lampung lookup-first pipeline."
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lora/model_lampung_merged.pt",
        help="Path ke checkpoint merged/base model.",
    )
    parser.add_argument(
        "--dataset",
        default="data/lampung/final/valid_augmented_instruction.jsonl",
        help="Instruction JSONL untuk evaluasi.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=200,
        help="Jumlah maksimum contoh yang dievaluasi. Pakai -1 untuk semua.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/results/eval_lampung_lookup.json",
        help="Path JSON output.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.checkpoint)
    generator = Generator(model, tokenizer, device="cpu")
    pipeline = LampungPipeline(generator, tokenizer)
    evaluator = LampungLookupEvaluator(pipeline)

    max_cases = None if args.max_cases < 0 else args.max_cases
    evaluator.evaluate_file(
        dataset_path=args.dataset,
        max_cases=max_cases,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
