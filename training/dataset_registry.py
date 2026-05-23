from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: str
    format: str = "instruction_jsonl"
    weight: int = 1
    system_prompt: str | None = None
    max_items: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MixTarget:
    name: str
    fraction: float


@dataclass(frozen=True)
class DatasetRegistry:
    name: str
    output_path: str
    sources: list[DatasetSource]
    shuffle_seed: int = 42
    target_total_rows: int | None = None
    target_mix: list[MixTarget] = field(default_factory=list)
    max_oversample_factor: int = 5

    @classmethod
    def from_json(cls, path: str | Path) -> "DatasetRegistry":
        registry_path = Path(path)
        with registry_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        sources = [
            DatasetSource(
                name=str(item["name"]),
                path=str(item["path"]),
                format=str(item.get("format", "instruction_jsonl")),
                weight=max(1, int(item.get("weight", 1))),
                system_prompt=item.get("system_prompt"),
                max_items=item.get("max_items"),
                metadata=dict(item.get("metadata", {})),
            )
            for item in data.get("sources", [])
        ]

        if not sources:
            raise ValueError(f"Dataset registry has no sources: {registry_path}")

        mix_data = data.get("target_mix", {})
        if isinstance(mix_data, dict):
            target_total_rows = mix_data.get("total_rows")
            raw_groups = mix_data.get("groups", [])
        else:
            target_total_rows = None
            raw_groups = []

        target_mix = [
            MixTarget(
                name=str(item["name"]),
                fraction=float(item["fraction"]),
            )
            for item in raw_groups
        ]
        if target_mix:
            fraction_sum = sum(item.fraction for item in target_mix)
            if not 0.99 <= fraction_sum <= 1.01:
                raise ValueError(
                    f"target_mix fractions in {registry_path} must sum to 1.0, got {fraction_sum:.4f}"
                )

        return cls(
            name=str(data.get("name", registry_path.stem)),
            output_path=str(data["output_path"]),
            sources=sources,
            shuffle_seed=int(data.get("shuffle_seed", 42)),
            target_total_rows=int(target_total_rows) if target_total_rows is not None else None,
            target_mix=target_mix,
            max_oversample_factor=max(1, int(mix_data.get("max_oversample_factor", 5))) if isinstance(mix_data, dict) else 5,
        )


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset source not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON {file_path}:{line_number}: {exc}")
                continue

            if isinstance(row, dict):
                yield row


def read_text_chunks(path: str | Path, max_chars: int = 1200) -> list[str]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Text source not found: {file_path}")

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = [" ".join(part.split()) for part in text.split("\n\n")]
    paragraphs = [part for part in paragraphs if part]

    if len(paragraphs) <= 1:
        paragraphs = [" ".join(part.split()) for part in text.splitlines()]
        paragraphs = [part for part in paragraphs if part]

    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}"
        else:
            chunks.append(current)
            current = paragraph

    if current:
        chunks.append(current)

    return chunks
