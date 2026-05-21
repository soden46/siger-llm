from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_for_match(value: Any) -> str:
    return normalize_text(value).lower()


def rough_token_count(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    word_like = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    byte_like = max(1, len(text) // 4)
    return max(word_like, byte_like)


def row_text(row: dict[str, Any]) -> str:
    return "\n".join(
        normalize_text(row.get(field))
        for field in ("system", "instruction", "input", "output", "reasoning")
        if normalize_text(row.get(field))
    )


def fingerprint(row: dict[str, Any]) -> str:
    text = normalize_for_match(row_text(row))
    text = re.sub(r"[^\w\s<>/]+", "", text, flags=re.UNICODE)
    return text[:20000]


def iter_jsonl(path: str | Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, None, str(exc)
                continue
            if not isinstance(row, dict):
                yield line_number, None, "row is not a JSON object"
                continue
            yield line_number, row, None


def repetition_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 1.0
    return 1.0 - (len(set(words)) / len(words))


def contains_all(text: str, needles: list[str]) -> bool:
    haystack = normalize_for_match(text)
    return all(normalize_for_match(needle) in haystack for needle in needles)


def contains_none(text: str, needles: list[str]) -> bool:
    haystack = normalize_for_match(text)
    return all(normalize_for_match(needle) not in haystack for needle in needles)


def metadata_license(path: str | Path) -> str:
    metadata_path = Path(path)
    if not metadata_path.exists():
        return ""
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return normalize_text(data.get("license"))
