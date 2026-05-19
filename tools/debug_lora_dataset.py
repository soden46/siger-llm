import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Iterable


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lora.dataset import InstructionDataset, format_local_instruction
from tokenizer.hybrid_tokenizer import build_tokenizer


STANDARD_KEYS = {"instruction", "input", "output", "system", "source", "type", "reasoning"}
REQUIRED_SPECIAL_TOKENS = (
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end_turn|>",
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
)


def iter_jsonl(path: Path, limit: int) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as f:
        seen = 0
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON line {line_number}: {exc}")
                continue
            yield line_number, row
            seen += 1
            if seen >= limit:
                break


def compact(text: str, limit: int = 500) -> str:
    text = text.replace("\r", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def supervised_token_count(labels: list[int], ignore_index: int) -> int:
    return sum(1 for label in labels if label != ignore_index)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, max(0, int(len(values) * q)))
    return sorted(values)[index]


def print_special_token_report(tokenizer) -> None:
    print("\n=== Special Token Check ===")
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.special_tokens.get(token)
        encoded = tokenizer.encode(token)
        ok = token_id is not None and encoded == [token_id]
        status = "OK" if ok else "BROKEN"
        print(f"{token:<14} id={str(token_id):>8} encode={encoded} {status}")


def build_labels(tokenizer, input_ids: list[int]) -> list[int]:
    mask_builder = InstructionDataset.__new__(InstructionDataset)
    mask_builder.tokenizer = tokenizer
    return mask_builder._build_labels(input_ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect LoRA instruction formatting and assistant-only loss masks."
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="data/corpus/indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl",
        help="Path to local instruction JSONL.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of rows to inspect.")
    parser.add_argument("--stats-limit", type=int, default=500, help="Rows used for distribution stats.")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Training truncation length.")
    parser.add_argument(
        "--tokenizer",
        default="auto",
        choices=["auto", "hf_bpe", "tiktoken"],
        help="Tokenizer backend.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    tokenizer = build_tokenizer(args.tokenizer)

    assistant_id = tokenizer.special_tokens.get("<|assistant|>")
    end_turn_id = tokenizer.special_tokens.get("<|end_turn|>")
    print(f"Tokenizer backend: {getattr(tokenizer, 'backend', args.tokenizer)}")
    print(f"Vocab size       : {tokenizer.vocab_size}")
    print(f"Assistant token  : {assistant_id}")
    print(f"End-turn token   : {end_turn_id}")
    print(f"Dataset          : {dataset_path}")
    print_special_token_report(tokenizer)

    total_supervised = 0
    total_tokens = 0
    valid_rows = 0

    for index, (line_number, row) in enumerate(iter_jsonl(dataset_path, args.limit)):
        formatted = format_local_instruction(row)
        if not formatted:
            print(f"\n=== Example {index} | line {line_number} ===")
            print(f"Keys              : {sorted(row.keys())}")
            print("WARNING: row does not format into a valid instruction example.")
            continue

        input_ids = tokenizer.encode(formatted, add_bos=True, add_eos=True)
        truncated = len(input_ids) > args.max_seq_len
        input_ids = input_ids[: args.max_seq_len]
        labels = build_labels(tokenizer, input_ids)
        supervised = supervised_token_count(labels, InstructionDataset.IGNORE_INDEX)
        shifted_labels = labels[1:]
        loss_supervised = supervised_token_count(
            shifted_labels,
            InstructionDataset.IGNORE_INDEX,
        )

        supervised_ids = [
            label for label in labels if label != InstructionDataset.IGNORE_INDEX
        ]
        loss_supervised_ids = [
            label
            for label in shifted_labels
            if label != InstructionDataset.IGNORE_INDEX
        ]
        supervised_text = tokenizer.decode(supervised_ids, skip_special_tokens=False)
        loss_supervised_text = tokenizer.decode(
            loss_supervised_ids,
            skip_special_tokens=False,
        )

        total_supervised += loss_supervised
        total_tokens += len(input_ids)
        valid_rows += 1

        output = str(row.get("output", "")).strip()
        instruction = str(row.get("instruction", "")).strip()
        input_text = str(row.get("input", "")).strip()

        print(f"\n=== Example {index} | line {line_number} ===")
        print(f"Keys              : {sorted(row.keys())}")
        for key in sorted(set(row.keys()) - STANDARD_KEYS):
            print(f"Non-standard key  : {key}={compact(str(row[key]), 80)}")
        print(f"Raw has assistant : {'<|assistant|>' in json.dumps(row, ensure_ascii=False)}")
        print(f"Formatted special : assistant={'<|assistant|>' in formatted}, end_turn={'<|end_turn|>' in formatted}")
        print(f"Instruction chars : {len(instruction)}")
        print(f"Input chars       : {len(input_text)}")
        print(f"Output chars      : {len(output)}")
        print(f"Token length      : {len(input_ids)} / {args.max_seq_len} truncated={truncated}")
        print(f"Label tokens      : {supervised}")
        print(f"Loss tokens       : {loss_supervised}")
        print("Loss alignment    : logits[:, :-1] predicts labels[:, 1:]")

        if loss_supervised == 0:
            print("WARNING: no assistant tokens reach the shifted loss for this row.")

        print("\nFormatted preview:")
        print(compact(formatted))
        print("\nLabel preview:")
        print(compact(supervised_text))
        print("\nLoss target preview:")
        print(compact(loss_supervised_text))

    if valid_rows:
        ratio = total_supervised / max(total_tokens, 1)
        avg_supervised = total_supervised / valid_rows
        print("\n=== Summary ===")
        print(f"Rows inspected       : {valid_rows}")
        print(f"Avg supervised tokens: {avg_supervised:.1f}")
        print(f"Supervised/token ratio: {ratio:.2%}")

    print(f"\n=== Distribution On First {args.stats_limit} Rows ===")
    lengths: list[int] = []
    truncated_count = 0
    empty_format_count = 0
    zero_loss_count = 0
    loss_counts: list[int] = []

    for _, row in iter_jsonl(dataset_path, args.stats_limit):
        formatted = format_local_instruction(row)
        if not formatted:
            empty_format_count += 1
            continue

        full_ids = tokenizer.encode(formatted, add_bos=True, add_eos=True)
        input_ids = full_ids[: args.max_seq_len]
        labels = build_labels(tokenizer, input_ids)
        loss_count = supervised_token_count(labels[1:], InstructionDataset.IGNORE_INDEX)

        lengths.append(len(full_ids))
        loss_counts.append(loss_count)
        if len(full_ids) > args.max_seq_len:
            truncated_count += 1
        if loss_count == 0:
            zero_loss_count += 1

    if lengths:
        total = len(lengths)
        print(f"Formatted rows       : {total}")
        print(f"Empty formatted rows : {empty_format_count}")
        print(
            "Seq length           : "
            f"median={statistics.median(lengths):.0f}, "
            f"p95={percentile(lengths, 0.95)}, max={max(lengths)}"
        )
        print(
            "Loss tokens          : "
            f"median={statistics.median(loss_counts):.0f}, "
            f"min={min(loss_counts)}, p95={percentile(loss_counts, 0.95)}, "
            f"max={max(loss_counts)}"
        )
        print(f"Zero loss rows       : {zero_loss_count}/{total} ({zero_loss_count / total:.1%})")
        print(f"Truncated rows       : {truncated_count}/{total} ({truncated_count / total:.1%})")
    else:
        print("No valid formatted rows found for distribution stats.")


if __name__ == "__main__":
    main()
