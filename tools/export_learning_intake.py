from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export approved SigerLM learning-intake rows into a training JSONL.",
    )
    parser.add_argument(
        "--input",
        default="data/intake/approved_training.jsonl",
        help="Approved intake JSONL.",
    )
    parser.add_argument(
        "--output",
        default="data/corpus/learning_intake_approved_train.jsonl",
        help="Output training JSONL.",
    )
    parser.add_argument(
        "--format",
        choices=["instruction_jsonl", "text_completion"],
        default="instruction_jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    skipped = 0

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("approved_for_training"):
                skipped += 1
                continue
            if row.get("privacy", {}).get("has_high") or row.get("privacy", {}).get("has_critical"):
                skipped += 1
                continue

            payload = row.get("payload") or {}
            if args.format == "text_completion":
                text = str(payload.get("text") or "").strip()
                if not text:
                    skipped += 1
                    continue
                out = {
                    "text": text,
                    "source": "approved_learning_intake",
                    "type": "text_completion",
                    "domain": row.get("domain", "general"),
                    "language": row.get("language", ""),
                    "intake_id": row.get("intake_id", ""),
                }
            else:
                instruction = str(payload.get("instruction") or "").strip()
                output = str(payload.get("output") or "").strip()
                text = str(payload.get("text") or "").strip()
                if not instruction and text:
                    instruction = "Pelajari dan jelaskan informasi berikut secara ringkas."
                    output = text
                if not instruction or not output:
                    skipped += 1
                    continue
                out = {
                    "instruction": instruction,
                    "input": str(payload.get("input") or ""),
                    "output": output,
                    "source": "approved_learning_intake",
                    "type": "instruction",
                    "domain": row.get("domain", "general"),
                    "language": row.get("language", ""),
                    "intake_id": row.get("intake_id", ""),
                }

            dst.write(json.dumps(out, ensure_ascii=False) + "\n")
            exported += 1

    print(json.dumps({"exported": exported, "skipped": skipped, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
