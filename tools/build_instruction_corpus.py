from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset_registry import DatasetRegistry, DatasetSource, iter_jsonl, read_text_chunks


DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, ringkas, dan akurat. "
    "Jawab sesuai instruksi user."
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def instruction_row(
    instruction: str,
    output: str,
    input_text: str = "",
    system_prompt: str | None = None,
    source: str = "",
    task_type: str = "instruction",
) -> dict[str, Any]:
    row = {
        "instruction": normalize_text(instruction),
        "input": normalize_text(input_text),
        "output": normalize_text(output),
        "source": source,
        "type": task_type,
    }
    if system_prompt:
        row["system"] = normalize_text(system_prompt)
    return row


def normalize_instruction_record(
    raw: dict[str, Any],
    *,
    source_name: str,
    system_prompt: str | None,
    task_type: str,
) -> dict[str, Any] | None:
    instruction = normalize_text(raw.get("instruction"))
    output = normalize_text(raw.get("output"))
    input_text = normalize_text(raw.get("input"))

    if not instruction or not output:
        return None

    row = {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "system": normalize_text(raw.get("system") or system_prompt or DEFAULT_SYSTEM_PROMPT),
        "source": normalize_text(raw.get("source") or source_name),
        "type": normalize_text(raw.get("type") or task_type),
    }
    return row


def convert_instruction_jsonl(source: DatasetSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for raw in iter_jsonl(source.path):
        instruction = normalize_text(raw.get("instruction"))
        output = normalize_text(raw.get("output"))
        input_text = normalize_text(raw.get("input"))

        if not instruction or not output:
            continue

        row = normalize_instruction_record(
            raw,
            source_name=source.name,
            system_prompt=source.system_prompt,
            task_type=source.metadata.get("type", "instruction"),
        )
        if row:
            rows.append(row)

    return rows


def convert_chat_jsonl(source: DatasetSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for raw in iter_jsonl(source.path):
        messages = raw.get("messages")
        if not isinstance(messages, list):
            continue

        pending_user = ""
        for message in messages:
            role = message.get("role")
            content = normalize_text(message.get("content"))
            if not content:
                continue

            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user:
                rows.append(
                    instruction_row(
                        instruction=pending_user,
                        output=content,
                        system_prompt=source.system_prompt,
                        source=source.name,
                        task_type=source.metadata.get("type", "chat"),
                    )
                )
                pending_user = ""

    return rows


def convert_text_completion(source: DatasetSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for chunk in read_text_chunks(source.path):
        words = chunk.split()
        if len(words) < 20:
            continue

        split_at = max(8, int(len(words) * 0.35))
        prompt = " ".join(words[:split_at])
        continuation = " ".join(words[split_at:])

        rows.append(
            instruction_row(
                instruction="Lanjutkan teks berikut secara natural.",
                input_text=prompt,
                output=continuation,
                system_prompt=source.system_prompt,
                source=source.name,
                task_type=source.metadata.get("type", "text_completion"),
            )
        )

    return rows


def convert_source(source: DatasetSource) -> list[dict[str, Any]]:
    if source.format == "instruction_jsonl":
        rows = convert_instruction_jsonl(source)
    elif source.format == "chat_jsonl":
        rows = convert_chat_jsonl(source)
    elif source.format == "text_completion":
        rows = convert_text_completion(source)
    else:
        raise ValueError(f"Unsupported dataset source format: {source.format}")

    if source.max_items is not None:
        rows = rows[: source.max_items]

    weighted: list[dict[str, Any]] = []
    for row in rows:
        for _ in range(source.weight):
            weighted.append(dict(row))

    print(
        f"{source.name}: {len(rows)} rows"
        + (f" x{source.weight} => {len(weighted)}" if source.weight > 1 else "")
    )
    return weighted


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[dict[str, Any]] = []

    for row in rows:
        key = (
            normalize_text(row.get("instruction")).lower(),
            normalize_text(row.get("input")).lower(),
            normalize_text(row.get("output")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)

    return deduped


def build_corpus(registry: DatasetRegistry) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in registry.sources:
        rows.extend(convert_source(source))

    rows = dedupe(rows)
    random.Random(registry.shuffle_seed).shuffle(rows)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified instruction corpus.")
    parser.add_argument(
        "--registry",
        default="configs/datasets/general_instruction.json",
        help="Path to dataset registry JSON.",
    )
    args = parser.parse_args()

    registry = DatasetRegistry.from_json(args.registry)
    rows = build_corpus(registry)
    write_jsonl(registry.output_path, rows)

    print(f"\nBuilt corpus: {registry.name}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {registry.output_path}")


if __name__ == "__main__":
    main()
