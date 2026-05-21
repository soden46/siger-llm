from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BATAK_TOBA_MARKERS = {
    "adong",
    "akka",
    "dohot",
    "ibana",
    "molo",
    "ndang",
    "sian",
}

WEB_ARTIFACT_PATTERNS = (
    re.compile(r"\bpencarian populer\b", re.IGNORECASE),
    re.compile(r"\bdaftar presiden\b", re.IGNORECASE),
    re.compile(r"\bprivacy policy\b", re.IGNORECASE),
    re.compile(r"\bterms of service\b", re.IGNORECASE),
    re.compile(r"\bcookie\b", re.IGNORECASE),
    re.compile(r"\bklik di sini\b", re.IGNORECASE),
    re.compile(r"\bpowered by\b", re.IGNORECASE),
)

ENGLISH_TRANSLATION_HINTS = (
    "bahasa inggris",
    "english",
)

TRANSLATION_HINTS = (
    "terjemahkan",
    "translate",
)

LAMPUNG_HINTS = (
    "lampung",
    "dialek o",
    "dialek nyo",
)

INDONESIAN_COMMON_WORDS = {
    "aku",
    "anda",
    "apa",
    "bahasa",
    "dalam",
    "dan",
    "dengan",
    "di",
    "ini",
    "itu",
    "ke",
    "kamu",
    "karena",
    "makan",
    "saya",
    "sebagai",
    "untuk",
    "yang",
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def lower_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower()))


def row_context(row: dict[str, Any]) -> str:
    fields = [
        row.get("system"),
        row.get("instruction"),
        row.get("input"),
        row.get("output"),
        row.get("source"),
        row.get("type"),
    ]
    return "\n".join(normalize_text(field) for field in fields if field)


def looks_lampung_row(row: dict[str, Any]) -> bool:
    metadata = " ".join(
        normalize_text(row.get(key)).lower()
        for key in ("system", "instruction", "source", "type", "dialect")
    )
    return any(hint in metadata for hint in LAMPUNG_HINTS)


def has_batak_toba_noise(row: dict[str, Any]) -> bool:
    if not looks_lampung_row(row):
        return False
    words = lower_words(row_context(row))
    return bool(words & BATAK_TOBA_MARKERS)


def has_web_artifact(row: dict[str, Any]) -> bool:
    text = row_context(row)
    return any(pattern.search(text) for pattern in WEB_ARTIFACT_PATTERNS)


def has_instruction_mismatch(row: dict[str, Any]) -> bool:
    instruction = normalize_text(row.get("instruction")).lower()
    input_text = normalize_text(row.get("input"))
    row_type = normalize_text(row.get("type")).lower()
    if not any(hint in instruction for hint in TRANSLATION_HINTS):
        return False
    if "classification" in row_type or "label" in row_type:
        return True
    if not any(hint in instruction for hint in ENGLISH_TRANSLATION_HINTS):
        return False

    words = lower_words(input_text)
    if not words:
        return False
    indonesian_hits = len(words & INDONESIAN_COMMON_WORDS)
    return indonesian_hits >= 2


def issue_labels(row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if has_batak_toba_noise(row):
        labels.append("lampung_batak_toba_noise")
    if has_instruction_mismatch(row):
        labels.append("instruction_mismatch")
    if has_web_artifact(row):
        labels.append("web_artifact")
    return labels


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                yield line_number, {"_json_error": str(exc), "_raw": line[:300]}


def summarize_row(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    return {
        "instruction": normalize_text(row.get("instruction"))[:max_chars],
        "input": normalize_text(row.get("input"))[:max_chars],
        "output": normalize_text(row.get("output"))[:max_chars],
        "source": normalize_text(row.get("source")),
        "type": normalize_text(row.get("type")),
    }


def audit(path: Path, *, max_examples: int, max_chars: int) -> dict[str, Any]:
    issue_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_rows = 0

    for line_number, row in iter_jsonl(path):
        total_rows += 1
        if "_json_error" in row:
            labels = ["json_error"]
        else:
            labels = issue_labels(row)

        for label in labels:
            issue_counts[label] += 1
            source_counts[f"{label}::{normalize_text(row.get('source')) or 'unknown'}"] += 1
            type_counts[f"{label}::{normalize_text(row.get('type')) or 'unknown'}"] += 1
            if len(examples[label]) < max_examples:
                examples[label].append(
                    {
                        "line": line_number,
                        **summarize_row(row, max_chars=max_chars),
                    }
                )

    return {
        "path": str(path),
        "total_rows": total_rows,
        "issue_counts": dict(issue_counts),
        "top_sources": dict(source_counts.most_common(30)),
        "top_types": dict(type_counts.most_common(30)),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit SigerLM instruction JSONL noise.")
    parser.add_argument("path", help="Instruction JSONL file to audit.")
    parser.add_argument("--report", help="Optional JSON report output path.")
    parser.add_argument("--max-examples", type=int, default=5)
    parser.add_argument("--max-chars", type=int, default=220)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {path}")

    report = audit(path, max_examples=args.max_examples, max_chars=args.max_chars)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
