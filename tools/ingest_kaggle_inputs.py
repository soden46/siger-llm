from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


TEXT_KEYS = [
    "text",
    "content",
    "sentence",
    "paragraph",
    "article",
    "body",
    "berita",
    "kalimat",
    "kata",
    "slang",
    "formal",
    "normal",
    "normalized",
    "translation",
    "terjemah",
    "question",
    "answer",
    "instruction",
    "input",
    "output",
    "response",
]
INSTRUCTION_KEYS = ["instruction", "prompt", "question", "query"]
INPUT_KEYS = ["input", "context", "passage", "article", "source"]
OUTPUT_KEYS = ["output", "response", "answer", "target", "label", "completion"]

DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, akurat, dan ringkas. "
    "Jawab dalam Bahasa Indonesia kecuali user meminta bahasa lain."
)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def flatten_texts(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        text = clean_text(value)
        if text:
            yield text
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from flatten_texts(item)
        return
    if isinstance(value, list):
        for item in value:
            yield from flatten_texts(item)
        return
    text = normalize_text(value)
    if text:
        yield text


def first_value(row: dict[str, Any], keys: list[str]) -> Any:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for key in keys:
        actual = lower_map.get(key.lower())
        if actual is not None and row.get(actual) not in (None, ""):
            return row[actual]
    return None


def has_template_placeholder(*values: Any) -> bool:
    return any("{{" in str(value or "") or "}}" in str(value or "") for value in values)


def label_to_hoax_text(value: Any) -> str | None:
    text = normalize_text(value).lower()
    if text in {"1", "true", "hoax", "hoaks", "fake", "false news"}:
        return "hoaks"
    if text in {"0", "false", "valid", "real", "non-hoax", "non hoax", "bukan hoax", "bukan hoaks"}:
        return "bukan hoaks"
    return None


def row_to_instruction(row: dict[str, Any], source: str) -> dict[str, Any] | None:
    slang = normalize_text(first_value(row, ["slang", "tidak_baku", "nonformal", "informal"]))
    formal = normalize_text(first_value(row, ["formal", "normal", "normalized", "baku"]))
    if slang and formal and slang != formal and not has_template_placeholder(slang, formal):
        return {
            "instruction": "Normalisasikan kata tidak baku berikut ke Bahasa Indonesia baku.",
            "input": slang,
            "output": formal,
            "system": DEFAULT_SYSTEM_PROMPT,
            "source": source,
            "type": "kaggle_normalization",
        }

    user_text = clean_text(first_value(row, ["you", "user", "pertama"]))
    assistant_text = clean_text(first_value(row, ["eliana", "assistant", "bot", "kedua"]))
    if user_text and assistant_text and not has_template_placeholder(user_text, assistant_text):
        return {
            "instruction": "Balas percakapan pengguna berikut secara natural dalam Bahasa Indonesia.",
            "input": user_text,
            "output": assistant_text,
            "system": DEFAULT_SYSTEM_PROMPT,
            "source": source,
            "type": "kaggle_conversation",
        }

    if "hoaks" in source or "hoax" in source:
        article = clean_text(first_value(row, ["clean_text", "narasi", "article", "berita", "text"]))
        title = normalize_text(first_value(row, ["judul", "title"]))
        label_value = first_value(row, ["hoax", "hoaks", "label"])
        label_text = label_to_hoax_text(label_value)
        if article and label_text and not has_template_placeholder(title, article, label_text):
            input_text = f"Judul: {title}\n\nIsi: {article}" if title else article
            return {
                "instruction": "Klasifikasikan apakah berita berikut hoaks atau bukan hoaks.",
                "input": input_text,
                "output": label_text,
                "system": DEFAULT_SYSTEM_PROMPT,
                "source": source,
                "type": "kaggle_hoax_classification",
            }

    instruction = normalize_text(first_value(row, INSTRUCTION_KEYS))
    output = clean_text(first_value(row, OUTPUT_KEYS))
    input_text = clean_text(first_value(row, INPUT_KEYS))

    if not instruction or not output:
        return None

    if input_text == instruction:
        input_text = ""

    if has_template_placeholder(instruction, input_text, output):
        return None

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "system": DEFAULT_SYSTEM_PROMPT,
        "source": source,
        "type": "kaggle_local_instruction",
    }


def row_to_text(row: dict[str, Any]) -> str:
    chunks: list[str] = []
    lower_map = {str(key).lower(): key for key in row.keys()}
    for key in TEXT_KEYS:
        actual = lower_map.get(key.lower())
        if actual is None:
            continue
        chunks.extend(flatten_texts(row.get(actual)))

    if not chunks:
        chunks.extend(flatten_texts(row))

    return "\n".join(text for text in chunks if len(text) >= 2)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skip invalid JSONL {path}:{line_number}: {exc}")
                continue
            if isinstance(row, dict):
                yield row


def iter_json(path: Path) -> Iterable[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig", errors="ignore"))
    except json.JSONDecodeError as exc:
        print(f"Skip invalid JSON {path}: {exc}")
        return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        emitted = False
        for key in ["data", "rows", "train", "validation", "valid", "test", "items"]:
            value = data.get(key)
            if isinstance(value, list):
                emitted = True
                for item in value:
                    if isinstance(item, dict):
                        yield item
        if not emitted:
            yield data


def iter_csv(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        try:
            yield from csv.DictReader(f)
        except csv.Error as exc:
            print(f"Skip invalid CSV {path}: {exc}")


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from iter_jsonl(path)
    elif suffix == ".json":
        yield from iter_json(path)
    elif suffix == ".csv":
        yield from iter_csv(path)


def safe_source_name(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path.name
    return "kaggle__" + re.sub(r"[^A-Za-z0-9]+", "_", str(rel)).strip("_").lower()


def discover_files(input_dir: Path) -> list[Path]:
    suffixes = {".txt", ".csv", ".json", ".jsonl"}
    return sorted(path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def write_registry(registry_path: Path, instruction_output: Path, text_output: Path) -> None:
    registry = {
        "name": "kaggle_local_inputs",
        "output_path": "data/corpus/kaggle_local_inputs_train.jsonl",
        "shuffle_seed": 42,
        "sources": [
            {
                "name": "kaggle_local_instruction",
                "path": str(instruction_output).replace("\\", "/"),
                "format": "instruction_jsonl",
                "weight": 1,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "metadata": {"type": "kaggle_local_instruction"},
            },
            {
                "name": "kaggle_local_text",
                "path": str(text_output).replace("\\", "/"),
                "format": "text_completion",
                "weight": 1,
                "system_prompt": DEFAULT_SYSTEM_PROMPT,
                "metadata": {"type": "kaggle_local_text_completion"},
            },
        ],
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def ingest(
    *,
    input_dir: Path,
    text_output: Path,
    instruction_output: Path,
    registry_output: Path,
    min_text_chars: int,
    max_files: int | None,
) -> dict[str, Any]:
    files = discover_files(input_dir)
    if max_files is not None:
        files = files[:max_files]

    text_output.parent.mkdir(parents=True, exist_ok=True)
    instruction_output.parent.mkdir(parents=True, exist_ok=True)

    stats = {
        "input_dir": str(input_dir),
        "files": 0,
        "text_blocks": 0,
        "instruction_rows": 0,
        "skipped_short_text": 0,
        "by_suffix": {},
    }
    seen_text: set[str] = set()
    seen_instruction: set[tuple[str, str, str]] = set()

    with text_output.open("w", encoding="utf-8") as text_f, instruction_output.open("w", encoding="utf-8") as inst_f:
        for path in files:
            stats["files"] += 1
            stats["by_suffix"][path.suffix.lower()] = stats["by_suffix"].get(path.suffix.lower(), 0) + 1
            source = safe_source_name(path, input_dir)

            if path.suffix.lower() == ".txt":
                text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
                if len(text) < min_text_chars:
                    stats["skipped_short_text"] += 1
                    continue
                key = normalize_text(text).lower()
                if key not in seen_text:
                    seen_text.add(key)
                    text_f.write(text + "\n\n")
                    stats["text_blocks"] += 1
                continue

            for row in iter_records(path):
                text = row_to_text(row)
                if len(text) >= min_text_chars:
                    key = normalize_text(text).lower()
                    if key not in seen_text:
                        seen_text.add(key)
                        text_f.write(text + "\n\n")
                        stats["text_blocks"] += 1
                else:
                    stats["skipped_short_text"] += 1

                instruction = row_to_instruction(row, source)
                if instruction:
                    key = (
                        normalize_text(instruction["instruction"]).lower(),
                        normalize_text(instruction["input"]).lower(),
                        normalize_text(instruction["output"]).lower(),
                    )
                    if key not in seen_instruction:
                        seen_instruction.add(key)
                        inst_f.write(json.dumps(instruction, ensure_ascii=False) + "\n")
                        stats["instruction_rows"] += 1

    write_registry(registry_output, instruction_output, text_output)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Kaggle Add Input datasets into SigerLM local data files.")
    parser.add_argument("--input-dir", default="/kaggle/input")
    parser.add_argument("--text-output", default="data/kaggle/kaggle_extra_text.txt")
    parser.add_argument("--instruction-output", default="data/kaggle/kaggle_extra_instruction.jsonl")
    parser.add_argument("--registry-output", default="configs/datasets/kaggle_local_inputs.json")
    parser.add_argument("--report-output", default="data/kaggle/kaggle_ingest_report.json")
    parser.add_argument("--min-text-chars", type=int, default=40)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Kaggle input dir not found: {input_dir}")

    stats = ingest(
        input_dir=input_dir,
        text_output=Path(args.text_output),
        instruction_output=Path(args.instruction_output),
        registry_output=Path(args.registry_output),
        min_text_chars=args.min_text_chars,
        max_files=args.max_files,
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Kaggle input ingest complete")
    print(f"  files           : {stats['files']}")
    print(f"  text blocks     : {stats['text_blocks']}")
    print(f"  instruction rows: {stats['instruction_rows']}")
    print(f"  text output     : {args.text_output}")
    print(f"  instruction out : {args.instruction_output}")
    print(f"  registry        : {args.registry_output}")
    print(f"  report          : {args.report_output}")


if __name__ == "__main__":
    main()
