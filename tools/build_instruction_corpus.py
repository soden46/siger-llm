from __future__ import annotations

import argparse
import html
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.dataset_registry import DatasetRegistry, DatasetSource, iter_jsonl, read_text_chunks
from tools.cot_formatter import maybe_apply_cot


DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, ringkas, dan akurat. "
    "Jawab sesuai instruksi user."
)

HTML_FIELD_NAMES = ("system", "instruction", "input", "output")
WEB_HTML_RE = re.compile(
    r"</?(?:p|br|div|span|ul|ol|li|a|strong|b|em|i|blockquote|h[1-6]|table|tr|td|th)\b[^>]*>|href\s*=",
    flags=re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>\n]{1,500}>")
HTML_HREF_ATTR_RE = re.compile(r"\s+href\s*=\s*([\"']).*?\1", flags=re.IGNORECASE)
URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
CODE_CONTEXT_RE = re.compile(
    r"```|<\?php|</?(?:html|head|body|script|style|template|form|input|button|select|option|textarea|svg|canvas)\b|"
    r"\b(?:function|class|const|let|var|public|private|protected|import|export)\b",
    flags=re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def is_code_like_row(row: dict[str, Any]) -> bool:
    source = normalize_text(row.get("source")).lower()
    task_type = normalize_text(row.get("type")).lower()
    content = row_text(row)
    return (
        "code" in source
        or "code" in task_type
        or "laravel" in source
        or "laravel" in task_type
        or bool(CODE_CONTEXT_RE.search(content))
    )


def strip_web_html(text: str) -> tuple[str, bool]:
    if not text:
        return text, False

    decoded = html.unescape(text).replace("\xa0", " ")
    looks_like_web_html = bool(WEB_HTML_RE.search(decoded))
    if not looks_like_web_html:
        normalized = normalize_text(decoded)
        return normalized, normalized != text

    without_href = HTML_HREF_ATTR_RE.sub("", decoded)
    without_tags = HTML_TAG_RE.sub(" ", without_href)
    without_urls = URL_RE.sub("", without_tags)
    cleaned = normalize_text(without_urls)
    return cleaned, cleaned != normalize_text(text)


def sanitize_web_html_row(row: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if is_code_like_row(row):
        return row, 0

    fixed = dict(row)
    changed = 0
    for field in HTML_FIELD_NAMES:
        if field not in fixed:
            continue
        cleaned, did_change = strip_web_html(normalize_text(fixed.get(field)))
        if did_change:
            fixed[field] = cleaned
            changed += 1
    return fixed, changed


def row_text(row: dict[str, Any]) -> str:
    return "\n".join(
        normalize_text(row.get(field))
        for field in ("system", "instruction", "input", "output")
        if normalize_text(row.get(field))
    )


def rough_token_count(text: str) -> int:
    text = normalize_text(text)
    if not text:
        return 0
    word_like = len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))
    byte_like = max(1, len(text) // 4)
    return max(word_like, byte_like)


def row_token_count(row: dict[str, Any]) -> int:
    return rough_token_count(row_text(row))


def dedupe_fingerprint(row: dict[str, Any]) -> str:
    text = row_text(row).lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\w\s<>/]+", "", text, flags=re.UNICODE)
    return text[:20000]


def is_laravel_row(row: dict[str, Any]) -> bool:
    source = normalize_text(row.get("source")).lower()
    task_type = normalize_text(row.get("type")).lower()
    content = row_text(row).lower()
    return (
        "laravel" in source
        or "laravel" in task_type
        or "santrikoding" in source
        or "<?php" in content
        or "artisan" in content
    )


def repair_markdown_fences(text: str) -> tuple[str, bool]:
    if text.count("```") % 2 == 0:
        return text, False
    return text.rstrip() + "\n```", True


def sanitize_laravel_php(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats = {"laravel_fences_repaired": 0, "laravel_php_malformed": 0}
    if not is_laravel_row(row):
        return row, stats

    fixed = dict(row)
    for field in ("instruction", "input", "output"):
        value = normalize_text(fixed.get(field))
        if not value:
            continue
        if "?>" in value and "<?php" not in value:
            stats["laravel_php_malformed"] += 1
            return None, stats
        repaired, did_repair = repair_markdown_fences(value)
        if did_repair:
            fixed[field] = repaired
            stats["laravel_fences_repaired"] += 1

    return fixed, stats


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


def convert_mined_parallel_jsonl(source: DatasetSource) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for raw in iter_jsonl(source.path):
        text_id = normalize_text(raw.get("text_id") or raw.get("indonesian"))
        text_lam_o = normalize_text(raw.get("text_lam_o") or raw.get("lampung_o") or raw.get("lampung"))
        text_en = normalize_text(raw.get("text_en") or raw.get("english"))
        question_id = normalize_text(raw.get("question_id"))
        question_lam_o = normalize_text(raw.get("question_lam_o"))
        question_en = normalize_text(raw.get("question_en"))
        answer_id = normalize_text(raw.get("answer_id"))
        answer_lam_o = normalize_text(raw.get("answer_lam_o"))
        answer_en = normalize_text(raw.get("answer_en"))
        category = normalize_text(raw.get("_category"))
        system_prompt = source.system_prompt or DEFAULT_SYSTEM_PROMPT

        if question_id and answer_id:
            rows.append(
                instruction_row(
                    instruction=question_id,
                    output=answer_id,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "qa_id"),
                )
            )

        if question_lam_o and answer_id:
            rows.append(
                instruction_row(
                    instruction="Jawab pertanyaan Lampung Dialek O berikut dalam Bahasa Indonesia.",
                    input_text=question_lam_o,
                    output=answer_id,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "lampung_o_qa_to_id"),
                )
            )

        if question_en and answer_en:
            rows.append(
                instruction_row(
                    instruction=question_en,
                    output=answer_en,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "qa_en"),
                )
            )

        if answer_lam_o and answer_id:
            rows.append(
                instruction_row(
                    instruction="Terjemahkan teks Lampung Dialek O berikut ke Bahasa Indonesia.",
                    input_text=answer_lam_o,
                    output=answer_id,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "lampung_o_to_id"),
                )
            )

        if text_lam_o and text_id:
            rows.append(
                instruction_row(
                    instruction="Terjemahkan teks Lampung Dialek O berikut ke Bahasa Indonesia.",
                    input_text=text_lam_o,
                    output=text_id,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "lampung_o_to_id"),
                )
            )
            continue

        if text_lam_o and text_en:
            rows.append(
                instruction_row(
                    instruction="Translate the following Lampung Dialect O text into English.",
                    input_text=text_lam_o,
                    output=text_en,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "lampung_o_to_en"),
                )
            )

        if text_id:
            rows.append(
                instruction_row(
                    instruction=f"Tulis teks informatif dalam Bahasa Indonesia tentang: {category or 'pengetahuan umum'}.",
                    output=text_id,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "general_id_text"),
                )
            )
            continue

        if text_en:
            rows.append(
                instruction_row(
                    instruction=f"Write an informative text in English about: {category or 'general knowledge'}.",
                    output=text_en,
                    system_prompt=system_prompt,
                    source=source.name,
                    task_type=source.metadata.get("type", "general_en_text"),
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
    elif source.format == "mined_parallel_jsonl":
        rows = convert_mined_parallel_jsonl(source)
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


def filter_quality(
    rows: list[dict[str, Any]],
    *,
    max_row_tokens: int = 2048,
    strict_laravel_php: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats = {
        "filtered_too_long": 0,
        "filtered_laravel_php_malformed": 0,
        "laravel_fences_repaired": 0,
        "html_fields_sanitized": 0,
    }
    filtered: list[dict[str, Any]] = []

    for row in rows:
        row, html_fields_sanitized = sanitize_web_html_row(row)
        stats["html_fields_sanitized"] += html_fields_sanitized

        if max_row_tokens > 0 and row_token_count(row) > max_row_tokens:
            stats["filtered_too_long"] += 1
            continue

        if strict_laravel_php:
            sanitized, php_stats = sanitize_laravel_php(row)
            stats["filtered_laravel_php_malformed"] += php_stats["laravel_php_malformed"]
            stats["laravel_fences_repaired"] += php_stats["laravel_fences_repaired"]
            if sanitized is None:
                continue
            row = sanitized

        filtered.append(row)

    return filtered, stats


def dedupe_with_report(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    duplicates = 0

    for row in rows:
        key = dedupe_fingerprint(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        deduped.append(row)

    return deduped, duplicates


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped, _ = dedupe_with_report(rows)
    return deduped


def build_corpus_with_report(
    registry: DatasetRegistry,
    *,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
    max_row_tokens: int = 2048,
    strict_laravel_php: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "registry": registry.name,
        "output_path": str(registry.output_path),
        "max_row_tokens": max_row_tokens,
        "strict_laravel_php": strict_laravel_php,
        "sources": [],
    }
    for source in registry.sources:
        source_rows = convert_source(source)
        report["sources"].append(
            {
                "name": source.name,
                "format": source.format,
                "weight": source.weight,
                "rows_after_weight": len(source_rows),
            }
        )
        rows.extend(source_rows)

    if cot_ratio > 0:
        rows = [maybe_apply_cot(row, ratio=cot_ratio, mode=cot_mode) for row in rows]

    report["rows_before_quality"] = len(rows)
    rows, quality_stats = filter_quality(
        rows,
        max_row_tokens=max_row_tokens,
        strict_laravel_php=strict_laravel_php,
    )
    report.update(quality_stats)
    report["rows_after_quality"] = len(rows)

    rows, duplicates_removed = dedupe_with_report(rows)
    report["duplicates_removed"] = duplicates_removed
    report["rows_after_dedupe"] = len(rows)

    random.Random(registry.shuffle_seed).shuffle(rows)
    return rows, report


def build_corpus(
    registry: DatasetRegistry,
    *,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
    max_row_tokens: int = 2048,
    strict_laravel_php: bool = True,
) -> list[dict[str, Any]]:
    rows, _ = build_corpus_with_report(
        registry,
        cot_ratio=cot_ratio,
        cot_mode=cot_mode,
        max_row_tokens=max_row_tokens,
        strict_laravel_php=strict_laravel_php,
    )
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified instruction corpus.")
    parser.add_argument(
        "--registry",
        default="configs/datasets/general_instruction.json",
        help="Path to dataset registry JSON.",
    )
    parser.add_argument(
        "--cot-ratio",
        type=float,
        default=0.0,
        help="Convert this deterministic fraction of rows to <thought>...</thought> CoT format.",
    )
    parser.add_argument(
        "--cot-mode",
        choices=["auto", "minimal"],
        default="auto",
        help="CoT reasoning template style.",
    )
    parser.add_argument(
        "--max-row-tokens",
        type=int,
        default=2048,
        help="Drop rows whose rough token count exceeds this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-laravel-php-check",
        action="store_true",
        help="Disable lightweight Laravel/PHP snippet sanity checks.",
    )
    parser.add_argument(
        "--quality-report",
        default=None,
        help="Optional output path for corpus quality report JSON.",
    )
    args = parser.parse_args()

    registry = DatasetRegistry.from_json(args.registry)
    rows, report = build_corpus_with_report(
        registry,
        cot_ratio=args.cot_ratio,
        cot_mode=args.cot_mode,
        max_row_tokens=args.max_row_tokens,
        strict_laravel_php=not args.skip_laravel_php_check,
    )
    write_jsonl(registry.output_path, rows)
    report_path = args.quality_report or str(Path(registry.output_path).with_suffix(".report.json"))
    write_report(report_path, report)

    print(f"\nBuilt corpus: {registry.name}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {registry.output_path}")
    print(f"Quality report: {report_path}")
    if report["filtered_too_long"] or report["duplicates_removed"] or report["html_fields_sanitized"]:
        print(
            "Quality gate: "
            f"filtered_too_long={report['filtered_too_long']}, "
            f"html_fields_sanitized={report['html_fields_sanitized']}, "
            f"duplicates_removed={report['duplicates_removed']}"
        )


if __name__ == "__main__":
    main()
