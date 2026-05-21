from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit_instruction_corpus import issue_labels


DEFAULT_EXCLUDE_LABELS = {
    "lampung_batak_toba_noise",
    "instruction_mismatch",
    "web_artifact",
    "json_error",
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            raw_line = line.rstrip("\n")
            if not raw_line.strip():
                continue
            try:
                yield line_number, json.loads(raw_line), raw_line, []
            except json.JSONDecodeError as exc:
                yield line_number, {}, raw_line, ["json_error"]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_corpus(
    input_path: Path,
    output_path: Path,
    *,
    quarantine_path: Path | None,
    exclude_labels: set[str],
) -> dict[str, Any]:
    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    issue_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    total_rows = 0

    for line_number, row, raw_line, parse_labels in iter_jsonl(input_path):
        total_rows += 1
        labels = parse_labels or issue_labels(row)
        matched = sorted(set(labels) & exclude_labels)
        if matched:
            issue_counts.update(matched)
            source = str(row.get("source") or "unknown")
            for label in matched:
                source_counts[f"{label}::{source}"] += 1
            if quarantine_path is not None:
                quarantined.append(
                    {
                        "_line": line_number,
                        "_issues": matched,
                        "_raw": raw_line if parse_labels else None,
                        **row,
                    }
                )
            continue
        kept.append(row)

    write_jsonl(output_path, kept)
    if quarantine_path is not None:
        write_jsonl(quarantine_path, quarantined)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "quarantine_path": str(quarantine_path) if quarantine_path else None,
        "total_rows": total_rows,
        "kept_rows": len(kept),
        "quarantined_rows": len(quarantined) if quarantine_path else total_rows - len(kept),
        "exclude_labels": sorted(exclude_labels),
        "issue_counts": dict(issue_counts),
        "top_sources": dict(source_counts.most_common(30)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter noisy rows from a SigerLM instruction JSONL corpus.")
    parser.add_argument("input", help="Input instruction JSONL.")
    parser.add_argument("--output", required=True, help="Clean output JSONL.")
    parser.add_argument("--quarantine", help="Optional JSONL of removed rows with issue labels.")
    parser.add_argument("--report", help="Optional JSON report path.")
    parser.add_argument(
        "--exclude-label",
        action="append",
        choices=sorted(DEFAULT_EXCLUDE_LABELS),
        help="Issue label to exclude. Repeatable. Defaults to all supported labels.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    exclude_labels = set(args.exclude_label or DEFAULT_EXCLUDE_LABELS)
    report = filter_corpus(
        input_path,
        Path(args.output),
        quarantine_path=Path(args.quarantine) if args.quarantine else None,
        exclude_labels=exclude_labels,
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
