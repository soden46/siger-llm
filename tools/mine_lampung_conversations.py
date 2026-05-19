from __future__ import annotations

import argparse
import glob
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


API_URL = "https://backend.translatelampung.com/api/translate"
DEFAULT_OUTPUT_FILE = "data/mined/lampung/lampung_conversations_translated.jsonl"
DEFAULT_CHECKPOINT_FILE = "data/mined/lampung/lampung_conversations_checkpoint.json"
DEFAULT_REPORT_FILE = "data/mined/lampung/lampung_conversations_report.json"
DEFAULT_INPUT_GLOBS = [
    "data/mined/instruction/*.jsonl",
    "data/mined/hf_indonesia/*.jsonl",
    "data/kaggle/*.jsonl",
    "data/corpus/*_train.jsonl",
]

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; SigerLM-Dataset-Miner/1.0)",
}

SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten umum yang juga memahami percakapan "
    "Bahasa Indonesia dan Bahasa Lampung."
)

SPEAKER_RE = re.compile(r"^\s*([A-Za-zÀ-ÿ0-9_. \-]{1,32})\s*[:：]\s*(.+?)\s*$")
INSTRUCTION_PAIR_HINTS = (
    "qa",
    "qna",
    "question",
    "answer",
    "chat",
    "conversation",
    "dialog",
    "customer",
    "support",
)
INSTRUCTION_PAIR_BLOCKLIST = (
    "translation",
    "translate",
    "vocab",
    "slang",
    "reasoning",
    "software",
    "code",
    "text_completion",
)


@dataclass(frozen=True)
class ConversationCandidate:
    source_path: str
    row_index: int
    dialog: str


@dataclass(frozen=True)
class DialectSpec:
    code: str
    label: str
    instruction: str


DIALECTS = {
    "abl": DialectSpec(
        code="abl",
        label="Dialek O (Nyo)",
        instruction="Terjemahkan percakapan Bahasa Indonesia berikut ke Bahasa Lampung Dialek O (Nyo).",
    ),
    "ljp": DialectSpec(
        code="ljp",
        label="Dialek A (Api)",
        instruction="Terjemahkan percakapan Bahasa Indonesia berikut ke Bahasa Lampung Dialek A (Api).",
    ),
}


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def normalize_multiline(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = [normalize_space(line) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSON {path}:{line_number}: {exc}")
                continue
            if isinstance(row, dict):
                yield row


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()

    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).exists():
            matches = [pattern]

        for match in matches:
            path = Path(match)
            key = str(path.resolve())
            if path.is_file() and key not in seen:
                paths.append(path)
                seen.add(key)

    return paths


def role_to_speaker(role: str, turn_index: int) -> str:
    role_lower = role.lower().strip()
    if role_lower in {"assistant", "bot", "model"}:
        return "B"
    if role_lower in {"user", "human"}:
        return "A"
    return "A" if turn_index % 2 == 0 else "B"


def dialog_from_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    lines: list[str] = []
    for turn_index, item in enumerate(messages):
        if not isinstance(item, dict):
            continue
        content = normalize_multiline(
            item.get("content")
            or item.get("value")
            or item.get("text")
            or item.get("message")
        )
        if not content:
            continue
        speaker = role_to_speaker(str(item.get("role") or item.get("from") or ""), turn_index)
        lines.append(f"{speaker}: {content}")

    return "\n".join(lines)


def should_use_instruction_pair(row: dict[str, Any], allow_generic: bool) -> bool:
    if allow_generic:
        return True

    source_text = " ".join(
        normalize_space(row.get(key)).lower()
        for key in ("source", "type", "task_type", "category", "dataset")
    )
    if any(token in source_text for token in INSTRUCTION_PAIR_BLOCKLIST):
        return False
    return any(token in source_text for token in INSTRUCTION_PAIR_HINTS)


def dialog_from_instruction(row: dict[str, Any], allow_generic: bool = False) -> str:
    if not should_use_instruction_pair(row, allow_generic):
        return ""

    instruction = normalize_space(row.get("instruction") or row.get("question") or row.get("prompt"))
    input_text = normalize_space(row.get("input") or row.get("context"))
    output = normalize_space(row.get("output") or row.get("answer") or row.get("response"))

    if not instruction or not output:
        return ""

    user_text = f"{instruction}\n\n{input_text}".strip() if input_text else instruction
    return f"A: {user_text}\nB: {output}"


def looks_like_dialog(text: str) -> bool:
    lines = [line for line in normalize_multiline(text).splitlines() if line]
    speaker_lines = [line for line in lines if SPEAKER_RE.match(line)]
    return len(speaker_lines) >= 2


def normalize_dialog(text: str) -> str:
    lines: list[str] = []
    for raw_line in normalize_multiline(text).splitlines():
        match = SPEAKER_RE.match(raw_line)
        if not match:
            continue
        speaker = normalize_space(match.group(1))
        sentence = normalize_space(match.group(2))
        if speaker and sentence:
            lines.append(f"{speaker}: {sentence}")
    return "\n".join(lines)


def extract_dialog(row: dict[str, Any], allow_generic_instruction: bool = False) -> str:
    for key in ("messages", "conversations"):
        dialog = dialog_from_messages(row.get(key))
        if looks_like_dialog(dialog):
            return normalize_dialog(dialog)

    for key in ("dialog", "dialogue", "conversation", "text", "input", "output"):
        text = normalize_multiline(row.get(key))
        if looks_like_dialog(text):
            return normalize_dialog(text)

    return normalize_dialog(dialog_from_instruction(row, allow_generic=allow_generic_instruction))


def iter_candidates(
    input_paths: list[Path],
    max_source_rows: int | None = None,
    allow_generic_instruction: bool = False,
) -> list[ConversationCandidate]:
    candidates: list[ConversationCandidate] = []
    seen_dialogs: set[str] = set()

    for path in input_paths:
        for row_index, row in enumerate(iter_jsonl(path)):
            if max_source_rows is not None and row_index >= max_source_rows:
                break

            dialog = extract_dialog(row, allow_generic_instruction=allow_generic_instruction)
            if not dialog:
                continue

            line_count = len(dialog.splitlines())
            char_count = len(dialog)
            if line_count < 2 or char_count < 24 or char_count > 3000:
                continue

            key = normalize_space(dialog.lower())
            if key in seen_dialogs:
                continue

            candidates.append(
                ConversationCandidate(
                    source_path=str(path),
                    row_index=row_index,
                    dialog=dialog,
                )
            )
            seen_dialogs.add(key)

    return candidates


def read_checkpoint(path: Path) -> dict[str, int]:
    if not path.exists():
        return {"candidate_index": 0, "success_count": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"candidate_index": 0, "success_count": 0}
    return {
        "candidate_index": int(data.get("candidate_index", 0)),
        "success_count": int(data.get("success_count", 0)),
    }


def write_checkpoint(path: Path, candidate_index: int, success_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "candidate_index": candidate_index,
                "success_count": success_count,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def extract_translation(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""

    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("translated", "translation", "text", "result"):
            value = normalize_space(data.get(key))
            if value:
                return value

    for key in ("translated", "translation", "text", "result"):
        value = normalize_space(payload.get(key))
        if value:
            return value

    return ""


def hit_api(
    text: str,
    dialect: str,
    *,
    timeout: float,
    max_retries: int,
    cooldown_seconds: float,
    dry_run: bool = False,
) -> str | None:
    if dry_run:
        return f"[{dialect}] {text}"

    payload = {"text": text, "from": "id", "to": dialect}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(API_URL, json=payload, headers=HEADERS, timeout=timeout)
        except requests.RequestException as exc:
            print(f"Request failed ({type(exc).__name__}), attempt {attempt + 1}/{max_retries + 1}")
            time.sleep(cooldown_seconds)
            continue

        if response.status_code == 200:
            translated = extract_translation(response.json())
            return translated or None

        if response.status_code == 429:
            print(f"Rate limited by API. Cooling down {cooldown_seconds:.1f}s.")
            time.sleep(cooldown_seconds)
            continue

        if response.status_code >= 500:
            return None

        print(f"Unexpected status {response.status_code}: {response.text[:120]}")
        return None

    return None


def translate_dialog(
    dialog: str,
    dialect: str,
    *,
    line_delay_seconds: float,
    timeout: float,
    max_retries: int,
    cooldown_seconds: float,
    dry_run: bool,
) -> tuple[str, int, int]:
    translated_lines: list[str] = []
    ok_count = 0
    fallback_count = 0

    for line in dialog.splitlines():
        match = SPEAKER_RE.match(line)
        if not match:
            continue

        speaker = normalize_space(match.group(1))
        sentence = normalize_space(match.group(2))
        if not sentence:
            continue

        translated = hit_api(
            sentence,
            dialect,
            timeout=timeout,
            max_retries=max_retries,
            cooldown_seconds=cooldown_seconds,
            dry_run=dry_run,
        )
        if translated:
            translated_lines.append(f"{speaker}: {translated}")
            ok_count += 1
        else:
            translated_lines.append(f"{speaker}: {sentence}")
            fallback_count += 1

        if line_delay_seconds > 0:
            time.sleep(line_delay_seconds)

    return "\n".join(translated_lines), ok_count, fallback_count


def make_output_row(
    candidate: ConversationCandidate,
    dialect: DialectSpec,
    translated_dialog: str,
) -> dict[str, Any]:
    return {
        "instruction": dialect.instruction,
        "input": candidate.dialog,
        "output": translated_dialog,
        "system": SYSTEM_PROMPT,
        "source": "translatelampung_conversation_mining",
        "type": "lampung_conversation_translation",
        "dialect": dialect.code,
        "dialect_label": dialect.label,
        "source_path": candidate.source_path,
        "source_row_index": candidate.row_index,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine Lampung conversation translation rows from already-mined local "
            "SigerLM JSONL datasets."
        )
    )
    parser.add_argument("--input", nargs="*", default=DEFAULT_INPUT_GLOBS, help="Input JSONL files or glob patterns.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT_FILE)
    parser.add_argument("--report", default=DEFAULT_REPORT_FILE)
    parser.add_argument("--target-count", type=int, default=1000, help="Number of source conversations to process.")
    parser.add_argument("--dialects", nargs="+", default=["abl", "ljp"], choices=sorted(DIALECTS))
    parser.add_argument("--max-source-rows", type=int, default=None)
    parser.add_argument("--line-delay", type=float, default=0.4)
    parser.add_argument("--conversation-delay", type=float, default=1.5)
    parser.add_argument("--cooldown", type=float, default=15.0)
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--fresh", action="store_true", help="Ignore checkpoint and overwrite output.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call API; write placeholder translations.")
    parser.add_argument(
        "--include-any-instruction-pairs",
        action="store_true",
        help=(
            "Also turn generic instruction/output rows into two-turn dialogs. "
            "Default only uses Q&A/chat/customer-support-like instruction rows."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_paths = expand_input_paths(args.input)
    if not input_paths:
        raise FileNotFoundError(f"No input JSONL files matched: {args.input}")

    print("Input files:")
    for path in input_paths:
        print(f"  - {path}")

    candidates = iter_candidates(
        input_paths,
        max_source_rows=args.max_source_rows,
        allow_generic_instruction=args.include_any_instruction_pairs,
    )
    print(f"Conversation candidates: {len(candidates):,}")

    if args.fresh:
        checkpoint = {"candidate_index": 0, "success_count": 0}
        output_path.write_text("", encoding="utf-8")
    else:
        checkpoint = read_checkpoint(checkpoint_path)

    start_index = checkpoint["candidate_index"]
    success_count = checkpoint["success_count"]
    mode = "a" if output_path.exists() and not args.fresh else "w"

    translated_lines = 0
    fallback_lines = 0
    written_rows = 0

    with output_path.open(mode, encoding="utf-8") as f_out:
        for candidate_index in range(start_index, len(candidates)):
            if success_count >= args.target_count:
                break

            candidate = candidates[candidate_index]
            print(
                f"Processing candidate {candidate_index + 1}/{len(candidates)} "
                f"| success {success_count}/{args.target_count}"
            )

            rows_for_candidate: list[dict[str, Any]] = []
            candidate_ok_lines = 0
            candidate_fallback_lines = 0

            for dialect_code in args.dialects:
                dialect = DIALECTS[dialect_code]
                translated_dialog, ok_count, fallback_count = translate_dialog(
                    candidate.dialog,
                    dialect.code,
                    line_delay_seconds=args.line_delay,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                    cooldown_seconds=args.cooldown,
                    dry_run=args.dry_run,
                )
                if not translated_dialog:
                    continue

                rows_for_candidate.append(make_output_row(candidate, dialect, translated_dialog))
                candidate_ok_lines += ok_count
                candidate_fallback_lines += fallback_count

            if rows_for_candidate:
                for row in rows_for_candidate:
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                f_out.flush()
                success_count += 1
                written_rows += len(rows_for_candidate)
                translated_lines += candidate_ok_lines
                fallback_lines += candidate_fallback_lines

            write_checkpoint(checkpoint_path, candidate_index + 1, success_count)

            if args.conversation_delay > 0:
                time.sleep(args.conversation_delay)

    report = {
        "schema_version": "siger_lampung_conversation_mining_v1",
        "api_url": API_URL,
        "input_files": [str(path) for path in input_paths],
        "output": str(output_path),
        "checkpoint": str(checkpoint_path),
        "candidate_count": len(candidates),
        "target_count": args.target_count,
        "success_count": success_count,
        "written_rows": written_rows,
        "dialects": args.dialects,
        "translated_lines": translated_lines,
        "fallback_lines": fallback_lines,
        "dry_run": bool(args.dry_run),
    }
    write_report(report_path, report)

    print(f"Done. Conversations: {success_count}")
    print(f"Rows written this run: {written_rows}")
    print(f"Output: {output_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
