from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cot_formatter import maybe_apply_cot


DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, akurat, dan ringkas. "
    "Jawab dalam Bahasa Indonesia kecuali user meminta bahasa lain."
)
CUSTOMER_SUPPORT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten customer support Bahasa Indonesia yang ramah, "
    "jelas, dan membantu menyelesaikan masalah pengguna."
)
TEXT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten Bahasa Indonesia yang memahami teks umum "
    "dan melanjutkan tulisan secara natural."
)
TRANSLATION_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten penerjemah Indonesia-Inggris yang akurat "
    "dan menjaga makna asli."
)
QA_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten tanya jawab Bahasa Indonesia. Jawab akurat "
    "berdasarkan pertanyaan dan konteks yang tersedia."
)


@dataclass(frozen=True)
class HFDatasetSpec:
    name: str
    kind: str
    config: str | None = None
    split: str | None = None
    max_items: int | None = None


DEFAULT_SOURCES: list[HFDatasetSpec] = [
    HFDatasetSpec("indonesian-nlp/wikipedia-id", "text"),
    HFDatasetSpec("Lyon28/Corpus-Indonesia", "text"),
    HFDatasetSpec("Hemgg/indonesian2english-dataset", "translation"),
    HFDatasetSpec("hndrbrm/indonesia_vocabulary", "vocab"),
    HFDatasetSpec("abid/indonesia-medical-qna", "qa"),
    HFDatasetSpec("morissu/indonesian_corpus", "text"),
    HFDatasetSpec("IndonesiaAI/translated-samples", "instruction"),
    HFDatasetSpec("kaitchup/opus-Indonesian-to-English", "translation"),
    HFDatasetSpec("akahana/english-indonesia-wikimatrix", "translation"),
    HFDatasetSpec("akahana/english-indonesia", "translation"),
    HFDatasetSpec("ermandmand/indonesian-simple-instruction-dataset", "instruction"),
    HFDatasetSpec("IndonesiaAI/sft-dataset", "instruction"),
    HFDatasetSpec("audichandra/bitext_customer_support_llm_dataset_indonesian", "instruction"),
    HFDatasetSpec("LorthGyu/indonesian-qa", "qa"),
    HFDatasetSpec("theonlydo/indonesia-slang", "vocab"),
    HFDatasetSpec("nahiar/indonesia-slang", "vocab"),
]

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


def parse_mapping_string(value: Any) -> dict[str, Any] | None:
    """Parse fields that store a dict as a string, e.g. '{"id": "...", "en": "..."}'."""
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text or text[0] not in "{[":
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                parsed = parser(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def split_parallel_text(value: Any) -> tuple[str, str]:
    """Split translation pairs stored in one text field."""
    text = clean_text(value)
    if not text:
        return "", ""

    delimiters = ["###>", "##>", "|||", "\t"]
    for delimiter in delimiters:
        if delimiter in text:
            left, right = text.split(delimiter, 1)
            return clean_text(left), clean_text(right)

    return "", ""


def first_value(row: dict[str, Any], keys: list[str]) -> Any:
    lower_map = {str(key).lower(): key for key in row.keys()}
    for key in keys:
        current: Any = row
        ok = True
        for part in key.split("."):
            if isinstance(current, dict):
                current_lower = {str(k).lower(): k for k in current.keys()}
                actual_key = current_lower.get(part.lower())
                if actual_key and actual_key in current:
                    current = current[actual_key]
                elif part in current:
                    current = current[part]
                else:
                    ok = False
                    break
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def ensure_siger_instruction_row(row: dict[str, Any]) -> dict[str, Any] | None:
    instruction = normalize_text(row.get("instruction"))
    output = clean_text(row.get("output"))
    if not instruction or not output:
        return None
    return {
        "instruction": instruction,
        "input": clean_text(row.get("input")),
        "output": output,
        "system": normalize_text(row.get("system")) or DEFAULT_SYSTEM_PROMPT,
        "source": normalize_text(row.get("source")) or "unknown",
        "type": normalize_text(row.get("type")) or "instruction",
    }


def instruction_row(
    instruction: str,
    output: str,
    *,
    input_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    source: str,
    task_type: str,
) -> dict[str, Any] | None:
    instruction = normalize_text(instruction)
    input_text = clean_text(input_text)
    output = clean_text(output)
    if not instruction or not output:
        return None
    return ensure_siger_instruction_row({
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "system": system_prompt,
        "source": source,
        "type": task_type,
    })


def pair_from_known_keys(row: dict[str, Any], left_keys: list[str], right_keys: list[str]) -> tuple[str, str]:
    left = first_value(row, left_keys)
    right = first_value(row, right_keys)
    return clean_text(left), clean_text(right)


def first_two_text_columns(row: dict[str, Any]) -> tuple[str, str]:
    values: list[str] = []
    for key, value in row.items():
        key_l = str(key).lower()
        if key_l.startswith("_"):
            continue
        text = clean_text(value)
        if len(text) >= 2:
            values.append(text)
        if len(values) >= 2:
            break
    if len(values) >= 2:
        return values[0], values[1]
    return "", ""


def row_to_instruction(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    messages = first_value(row, ["messages", "conversations"])
    if isinstance(messages, list):
        output: list[dict[str, Any]] = []
        pending_user = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = normalize_text(message.get("role") or message.get("from")).lower()
            content = clean_text(message.get("content") or message.get("text") or message.get("value"))
            if role in {"user", "human", "instruction"}:
                pending_user = content
            elif role in {"assistant", "gpt", "bot"} and pending_user and content:
                built = instruction_row(
                    pending_user,
                    content,
                    source=source,
                    task_type="general_chat",
                )
                if built:
                    output.append(built)
                pending_user = ""
        if output:
            return output

    instruction = first_value(row, [
        "instruction", "prompt", "question", "query", "user", "customer", "customer_message",
        "user_message", "input", "text_input", "problem", "issue", "column3",
    ])
    input_text = first_value(row, ["context", "input_text", "source", "text", "conversation", "history", "column1", "column2"])
    answer = first_value(row, [
        "output", "response", "answer", "completion", "target", "assistant", "assistant_message",
        "agent", "agent_response", "reply", "label", "response_j", "column4",
    ])
    if normalize_text(instruction).lower() in {"instruction", "prompt", "question"}:
        return []
    if normalize_text(answer).lower() in {"response", "answer", "output"}:
        return []
    system_prompt = CUSTOMER_SUPPORT_SYSTEM_PROMPT if "customer" in source.lower() or "support" in source.lower() else DEFAULT_SYSTEM_PROMPT
    built = instruction_row(
        normalize_text(instruction),
        clean_text(answer),
        input_text=clean_text(input_text) if input_text != instruction else "",
        system_prompt=system_prompt,
        source=source,
        task_type="customer_support" if system_prompt == CUSTOMER_SUPPORT_SYSTEM_PROMPT else "general_instruction",
    )
    return [built] if built else []


def row_to_qa(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    question = first_value(row, ["question", "query", "instruction", "prompt"])
    answer = first_value(row, ["answer", "answers", "response", "output", "completion"])
    context = first_value(row, ["context", "passage", "article", "text"])
    built = instruction_row(
        normalize_text(question),
        " ".join(flatten_texts(answer)),
        input_text=clean_text(context),
        system_prompt=QA_SYSTEM_PROMPT,
        source=source,
        task_type="indonesian_qa",
    )
    return [built] if built else []


def row_to_translation(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    translation = first_value(row, ["translation"])
    translation_mapping = parse_mapping_string(translation)
    if translation_mapping:
        id_text = first_value(translation_mapping, ["id", "indonesian", "indonesia", "indo", "id_id", "source"])
        en_text = first_value(translation_mapping, ["en", "english", "eng", "en_uk", "en_us", "target"])
    else:
        id_text, en_text = pair_from_known_keys(
            row,
            [
                "id", "indonesian", "indonesia", "indo", "bahasa_indonesia", "text_id", "id_text",
                "sentence_id", "source_id", "source", "input", "text_1", "text1", "kalimat_indonesia",
                "question", "prompt",
            ],
            [
                "en", "english", "eng", "bahasa_inggris", "text_en", "en_text", "sentence_en",
                "target_en", "target", "output", "text_2", "text2", "kalimat_inggris",
                "response", "answer", "completion",
            ],
        )

        if not id_text or not en_text:
            pairs = [
                ("indonesian", "translation"),
                ("translation_id", "translation_en"),
                ("src", "tgt"),
                ("source_text", "target_text"),
                ("input_text", "output_text"),
                ("question", "answer"),
                ("question", "response"),
            ]
            for left_key, right_key in pairs:
                id_text, en_text = pair_from_known_keys(row, [left_key], [right_key])
                if id_text and en_text:
                    break

        if not id_text or not en_text:
            for key in ("text", "translation", "content", "sentence"):
                id_text, en_text = split_parallel_text(first_value(row, [key]))
                if id_text and en_text:
                    break

        if not id_text or not en_text:
            left, right = first_two_text_columns(row)
            split_left, split_right = split_parallel_text(left)
            id_text, en_text = (split_left, split_right) if split_left and split_right else (left, right)

    id_text = clean_text(id_text)
    en_text = clean_text(en_text)
    rows: list[dict[str, Any]] = []
    if id_text and en_text:
        id_to_en = instruction_row(
            "Terjemahkan teks berikut dari Bahasa Indonesia ke Bahasa Inggris.",
            en_text,
            input_text=id_text,
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            source=source,
            task_type="id_en_translation",
        )
        en_to_id = instruction_row(
            "Terjemahkan teks berikut dari Bahasa Inggris ke Bahasa Indonesia.",
            id_text,
            input_text=en_text,
            system_prompt=TRANSLATION_SYSTEM_PROMPT,
            source=source,
            task_type="en_id_translation",
        )
        rows.extend(row for row in [id_to_en, en_to_id] if row)
    return rows


def row_to_vocab(row: dict[str, Any], source: str) -> list[dict[str, Any]]:
    word = first_value(row, ["word", "kata", "slang", "term", "token", "input", "alay", "informal", "text"])
    meaning = first_value(row, ["meaning", "arti", "definition", "formal", "normalized", "normal", "baku", "output"])
    example = first_value(row, ["example", "contoh", "sentence", "context"])
    word_text = normalize_text(word)
    if not word_text or word_text.lower() in {"text", "word", "kata"}:
        return []
    input_text = clean_text(example)
    if not clean_text(meaning):
        built = instruction_row(
            "Catat kosakata Bahasa Indonesia berikut.",
            word_text,
            input_text=word_text,
            source=source,
            task_type="indonesian_vocabulary_entry",
        )
        return [built] if built else []

    built = instruction_row(
        f"Jelaskan arti kata atau ungkapan berikut: {word_text}",
        clean_text(meaning),
        input_text=input_text,
        source=source,
        task_type="indonesian_vocabulary",
    )
    return [built] if built else []


def row_to_text(row: dict[str, Any], source: str, *, min_chars: int) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[str] = []
    for key in ["text", "content", "article", "paragraph", "sentence", "translation", "terjemah", "body"]:
        value = first_value(row, [key])
        candidates.extend(flatten_texts(value))
    if not candidates:
        candidates.extend(flatten_texts(row))

    rows: list[dict[str, Any]] = []
    texts: list[str] = []
    for text in candidates:
        text = clean_text(text)
        if len(text) < min_chars:
            id_text, en_text = split_parallel_text(text)
            if id_text and en_text:
                rows.extend(row_to_translation({"id": id_text, "en": en_text}, source))
            continue

        id_text, en_text = split_parallel_text(text)
        if id_text and en_text:
            rows.extend(row_to_translation({"id": id_text, "en": en_text}, source))
            texts.extend([id_text, en_text])
            continue

        texts.append(text)
        words = text.split()
        if len(words) < 24:
            continue
        split_at = max(8, int(len(words) * 0.35))
        built = instruction_row(
            "Lanjutkan teks berikut secara natural.",
            " ".join(words[split_at:]),
            input_text=" ".join(words[:split_at]),
            system_prompt=TEXT_SYSTEM_PROMPT,
            source=source,
            task_type="text_completion",
        )
        if built:
            rows.append(built)
    return rows, texts


def convert_row(
    row: dict[str, Any],
    spec: HFDatasetSpec,
    *,
    min_text_chars: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = spec.name.replace("/", "__")
    if spec.kind == "instruction":
        return row_to_instruction(row, source), []
    if spec.kind == "qa":
        return row_to_qa(row, source), []
    if spec.kind == "translation":
        return row_to_translation(row, source), []
    if spec.kind == "vocab":
        return row_to_vocab(row, source), []
    if spec.kind == "text":
        return row_to_text(row, source, min_chars=min_text_chars)
    raise ValueError(f"Unsupported dataset kind: {spec.kind}")


def iter_hf_rows(spec: HFDatasetSpec, *, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import DatasetDict, IterableDatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install datasets") from exc

    kwargs: dict[str, Any] = {"trust_remote_code": True, "streaming": streaming}
    if spec.split:
        kwargs["split"] = spec.split

    dataset = load_dataset(spec.name, spec.config, **kwargs)
    if isinstance(dataset, (DatasetDict, IterableDatasetDict)) or isinstance(dataset, dict):
        preferred_splits = ["train", "validation", "valid", "test"]
        for split_name in preferred_splits:
            if split_name not in dataset:
                continue
            for row in dataset[split_name]:
                yield dict(row)
        return

    for row in dataset:
        yield dict(row)


def parse_source(value: str) -> HFDatasetSpec:
    parts = value.split(":")
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("Use dataset:kind[:split[:config[:max_items]]]")
    max_items = int(parts[4]) if len(parts) >= 5 and parts[4] else None
    return HFDatasetSpec(
        name=parts[0],
        kind=parts[1],
        split=parts[2] if len(parts) >= 3 and parts[2] else None,
        config=parts[3] if len(parts) >= 4 and parts[3] else None,
        max_items=max_items,
    )


def dedupe_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        normalize_text(row.get("instruction")).lower(),
        normalize_text(row.get("input")).lower(),
        normalize_text(row.get("output")).lower(),
    )


def sample_row_preview(row: dict[str, Any], *, max_chars: int = 120) -> dict[str, str]:
    preview: dict[str, str] = {}
    for key, value in row.items():
        text = clean_text(value)
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        preview[str(key)] = text
    return preview


def mine_sources(
    sources: list[HFDatasetSpec],
    *,
    instruction_output: Path,
    text_output: Path | None,
    max_items_per_source: int | None,
    min_text_chars: int,
    streaming: bool,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
) -> dict[str, Any]:
    instruction_output.parent.mkdir(parents=True, exist_ok=True)
    if text_output:
        text_output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[tuple[str, str, str]] = set()
    report: dict[str, Any] = {
        "schema_version": "siger_instruction_v1",
        "cot_ratio": cot_ratio,
        "cot_mode": cot_mode,
        "sources": [],
        "total_instruction_rows": 0,
        "total_text_rows": 0,
    }

    with instruction_output.open("w", encoding="utf-8") as out_jsonl:
        text_handle = text_output.open("w", encoding="utf-8") if text_output else None
        try:
            for spec in sources:
                source_limit = spec.max_items if spec.max_items is not None else max_items_per_source
                source_rows = 0
                source_texts = 0
                scanned = 0
                sample_keys: list[str] = []
                sample_row: dict[str, str] = {}
                print(f"Loading {spec.name} ({spec.kind})")
                try:
                    for raw in iter_hf_rows(spec, streaming=streaming):
                        if not sample_keys:
                            sample_keys = sorted(str(key) for key in raw.keys())
                            sample_row = sample_row_preview(raw)
                        rows, texts = convert_row(raw, spec, min_text_chars=min_text_chars)
                        scanned += 1
                        for row in rows:
                            row = maybe_apply_cot(row, ratio=cot_ratio, mode=cot_mode)
                            key = dedupe_key(row)
                            if key in seen:
                                continue
                            seen.add(key)
                            out_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                            source_rows += 1
                        if text_handle:
                            for text in texts:
                                text_handle.write(text.replace("\r\n", "\n").strip() + "\n\n")
                                source_texts += 1
                        if source_limit is not None and source_rows >= source_limit:
                            break
                except Exception as exc:
                    print(f"Skip {spec.name}: {exc}")
                    report["sources"].append(
                        {
                            "name": spec.name,
                            "kind": spec.kind,
                            "rows": source_rows,
                            "texts": source_texts,
                            "scanned": scanned,
                            "sample_keys": sample_keys,
                            "sample_row": sample_row,
                            "error": str(exc),
                        }
                    )
                    continue

                print(f"{spec.name}: {source_rows} instruction rows, {source_texts} text rows")
                report["sources"].append(
                    {
                        "name": spec.name,
                        "kind": spec.kind,
                        "rows": source_rows,
                        "texts": source_texts,
                        "scanned": scanned,
                        "sample_keys": sample_keys,
                        "sample_row": sample_row if source_rows == 0 and source_texts == 0 else {},
                    }
                )
                report["total_instruction_rows"] += source_rows
                report["total_text_rows"] += source_texts
        finally:
            if text_handle:
                text_handle.close()

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine Indonesian Hugging Face datasets into SigerLM instruction JSONL and optional raw text."
    )
    parser.add_argument("--instruction-output", default="data/mined/hf_indonesia/indonesian_hf_mix_instruction.jsonl")
    parser.add_argument("--text-output", default="data/indonesian_hf_mix.txt")
    parser.add_argument("--report-output", default="data/mined/hf_indonesia/hf_mix_report.json")
    parser.add_argument("--max-items-per-source", type=int, default=50000)
    parser.add_argument("--min-text-chars", type=int, default=120)
    parser.add_argument("--no-streaming", action="store_true", help="Download full datasets instead of streaming rows.")
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        default=[],
        help="Extra source: dataset:kind[:split[:config[:max_items]]]. Kind: text, instruction, qa, translation, vocab.",
    )
    parser.add_argument("--only-custom-sources", action="store_true")
    parser.add_argument("--cot-ratio", type=float, default=0.0, help="Convert this deterministic fraction of mined rows to CoT format.")
    parser.add_argument("--cot-mode", choices=["auto", "minimal"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_sources = list(DEFAULT_SOURCES)
    sources = list(args.source) if args.only_custom_sources else [*default_sources, *args.source]
    report = mine_sources(
        sources,
        instruction_output=Path(args.instruction_output),
        text_output=Path(args.text_output) if args.text_output else None,
        max_items_per_source=args.max_items_per_source,
        min_text_chars=args.min_text_chars,
        streaming=not args.no_streaming,
        cot_ratio=args.cot_ratio,
        cot_mode=args.cot_mode,
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. Instruction: {args.instruction_output}")
    if args.text_output:
        print(f"Raw text   : {args.text_output}")
    print(f"Report     : {args.report_output}")


if __name__ == "__main__":
    main()
