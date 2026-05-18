from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.cot_formatter import maybe_apply_cot


USER_AGENT = (
    "SIGER-LLM-DatasetMiner/1.0 "
    "(educational dataset curation; contact via project repository)"
)
DEFAULT_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, akurat, dan ringkas. "
    "Jawab dalam Bahasa Indonesia kecuali user meminta bahasa lain."
)
QA_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten tanya jawab Bahasa Indonesia. "
    "Jawab berdasarkan konteks jika konteks tersedia, dan jangan mengarang fakta."
)
LARAVEL_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten programming yang membantu Laravel versi 9 sampai 13. "
    "Jawab praktis, akurat, dan sebutkan versi Laravel jika relevan."
)
CODE_SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten programming yang menulis kode benar, aman, "
    "teruji, dan mudah dirawat. Jelaskan asumsi penting jika relevan."
)

HF_QA_SOURCES = ["SEACrowd/indoqa"]
LARGE_HF_QA_SOURCES = ["SEACrowd/tydiqa_id"]
HF_INSTRUCTION_SOURCES = [
    "togethercomputer/MoAA-SFT",
]
GATED_HF_INSTRUCTION_SOURCES = ["Iftitahu/indonesian_instruct_stories"]
HF_COMMERCIAL_SAFE_INSTRUCTION_SOURCES = [
    ("QuixiAI/dolphin", "flan1m-alpaca-uncensored", None),
]
HF_COMMERCIAL_SAFE_REASONING_SOURCES = [
    ("microsoft/orca-math-word-problems-200k", "qa", None, None),
    ("openbmb/UltraInteract_pair", "preference", None, None),
    (
        "xTayyub/High-Quality-Synthetic-Python-Dataset-with-Reasoning-Traces-Chain-of-Thought-for-LLM-Fine-Tuning",
        "code",
        None,
        None,
    ),
    ("HuggingFaceTB/cosmopedia", "text", "stories", "train"),
]
HF_CODE_EVAL_SOURCES = [
    "openai/openai_humaneval",
    "loubnabnl/humaneval_infilling",
]
HF_LARAVEL_DATASET_SOURCES = [
    "nqhung97/docs-laravel-v13",
    "fchis/laravel-buildspec-training",
    "fchis/Laravel-13x-Planner-Instructions",
    "fchis/Laravel-13x-Code-Instructions",
    "patelakshay3943/laravel12-dataset-cp",
    "patelakshay3943/laravel12-dataset",
    "brijmansuriya/web-beast-laravel",
    "codeXpedite/Laravel",
    "relai-ai/laravel-reasoning",
]


@dataclass(frozen=True)
class MineStats:
    source: str
    rows: int
    output_path: str


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\xa0", " ")).strip()


def clean_text(value: Any) -> str:
    text = str(value or "").replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def instruction_row(
    instruction: str,
    output: str,
    *,
    input_text: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    source: str,
    task_type: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    instruction = normalize_text(instruction)
    input_text = clean_text(input_text)
    output = clean_text(output)
    if not instruction or not output or len(output) < 2:
        return None

    row: dict[str, Any] = {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "system": system_prompt,
        "source": source,
        "type": task_type,
    }
    if metadata:
        row.update(metadata)
    return row


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def apply_cot_to_rows(rows: Iterable[dict[str, Any]], *, cot_ratio: float, cot_mode: str) -> list[dict[str, Any]]:
    return [maybe_apply_cot(row, ratio=cot_ratio, mode=cot_mode) for row in rows]


def touch_jsonl(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        key = (
            normalize_text(row.get("instruction")).lower(),
            normalize_text(row.get("input")).lower(),
            normalize_text(row.get("output")).lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def first_value(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        current: Any = row
        ok = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                ok = False
                break
        if ok and current not in (None, ""):
            return current
    return None


def extract_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        for item in value:
            answer = extract_answer(item)
            if answer:
                return answer
        return ""
    if isinstance(value, dict):
        for key in ["content", "text", "answer", "answers", "value", "response", "output"]:
            answer = extract_answer(value.get(key))
            if answer:
                return answer
    return normalize_text(value)


def row_to_qa_instruction(row: dict[str, Any], source_name: str) -> dict[str, Any] | None:
    question = normalize_text(first_value(row, ["question", "query"]))
    context = clean_text(first_value(row, ["context", "passage", "paragraph", "article", "text"]) or "")
    answer = extract_answer(first_value(row, ["answer", "answers", "label"]))
    if not question or not answer:
        return None

    if context:
        instruction = "Jawab pertanyaan berdasarkan konteks berikut."
        input_text = f"Konteks:\n{context}\n\nPertanyaan:\n{question}"
    else:
        instruction = question
        input_text = ""

    return instruction_row(
        instruction,
        answer,
        input_text=input_text,
        system_prompt=QA_SYSTEM_PROMPT,
        source=source_name,
        task_type="indonesian_qa",
    )


def messages_to_instruction(messages: Any, source_name: str) -> list[dict[str, Any]]:
    if not isinstance(messages, list):
        return []
    rows: list[dict[str, Any]] = []
    pending_user = ""
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role") or message.get("from")).lower()
        content = clean_text(message.get("content") or message.get("text") or message.get("value") or "")
        if not content:
            continue
        if role in {"user", "human", "instruction"}:
            pending_user = content
        elif role in {"assistant", "gpt", "bot"} and pending_user:
            row = instruction_row(pending_user, content, source=source_name, task_type="general_chat")
            if row:
                rows.append(row)
            pending_user = ""
    return rows


def row_to_instruction(row: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    for key in ["messages", "conversations"]:
        if key in row:
            rows = messages_to_instruction(row[key], source_name)
            if rows:
                return rows

    instruction = normalize_text(first_value(row, ["instruction", "prompt", "question", "input", "query"]))
    input_text = clean_text(first_value(row, ["context", "input", "source"]) or "")
    output = clean_text(first_value(row, ["output", "response", "answer", "completion", "target", "chosen"]) or "")
    built = instruction_row(
        instruction,
        output,
        input_text=input_text if input_text != instruction else "",
        source=source_name,
        task_type="general_instruction",
    )
    return [built] if built else []


def row_to_text_completion(row: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    text = clean_text(
        first_value(
            row,
            ["text", "content", "article", "document", "story", "markdown", "prompt", "output", "response"],
        )
        or ""
    )
    words = text.split()
    if len(words) < 40:
        return []
    split_at = max(12, int(len(words) * 0.35))
    prompt = " ".join(words[:split_at])
    continuation = " ".join(words[split_at:])
    built = instruction_row(
        "Lanjutkan teks berikut secara natural.",
        continuation,
        input_text=prompt,
        source=source_name,
        task_type="commercial_safe_text_completion",
    )
    return [built] if built else []


def row_to_preference_instruction(row: dict[str, Any], source_name: str) -> list[dict[str, Any]]:
    instruction = normalize_text(
        first_value(row, ["instruction", "prompt", "question", "query", "task", "input"])
    )
    input_text = clean_text(first_value(row, ["context", "input", "source"]) or "")
    output_value = first_value(
        row,
        [
            "chosen",
            "chosen_response",
            "accepted",
            "winner",
            "response_j",
            "good_response",
            "target",
            "output",
            "answer",
        ],
    )
    output = clean_text(extract_answer(output_value))
    if not output and output_value is not None:
        output = clean_text(output_value)
    built = instruction_row(
        instruction,
        output,
        input_text=input_text if input_text != instruction else "",
        source=source_name,
        task_type="commercial_safe_preference_chosen",
    )
    return [built] if built else []


def row_to_code_instruction(row: dict[str, Any], source_name: str, *, laravel: bool = False) -> list[dict[str, Any]]:
    for key in ["messages", "conversations"]:
        if key in row:
            rows = messages_to_instruction(row[key], source_name)
            if rows:
                system_prompt = LARAVEL_SYSTEM_PROMPT if laravel else CODE_SYSTEM_PROMPT
                task_type = "laravel_instruction" if laravel else "code_instruction"
                for item in rows:
                    item["system"] = system_prompt
                    item["type"] = task_type
                return rows

    prompt = clean_text(
        first_value(
            row,
            [
                "prompt",
                "instruction",
                "question",
                "task",
                "problem",
                "description",
                "spec",
                "buildspec",
                "request",
                "input",
                "text",
                "content",
                "doc",
                "documentation",
            ],
        )
        or ""
    )
    context = clean_text(
        first_value(
            row,
            [
                "context",
                "input",
                "starter_code",
                "declaration",
                "signature",
                "imports",
                "prefix",
                "before",
                "source",
            ],
        )
        or ""
    )
    output = clean_text(
        first_value(
            row,
            [
                "output",
                "response",
                "answer",
                "completion",
                "canonical_solution",
                "solution",
                "code_solution",
                "code",
                "target",
                "accepted_answer",
                "final_answer",
                "assistant",
                "suffix",
                "after",
            ],
        )
        or ""
    )
    thought = clean_text(first_value(row, ["thought_process", "reasoning", "cot", "chain_of_thought", "analysis"]) or "")
    explanation = clean_text(first_value(row, ["explanation", "final_answer", "summary"]) or "")
    if thought and output and "<thought>" not in output and "<think>" not in output:
        tail = output
        if explanation and explanation not in tail:
            tail = f"{tail}\n\nPenjelasan:\n{explanation}"
        output = f"<thought> {normalize_text(thought)} </thought>\n{tail}"
    tests = clean_text(first_value(row, ["test", "tests", "unit_tests", "example_test"]) or "")
    entry_point = normalize_text(first_value(row, ["entry_point", "function_name", "name"]) or "")

    if not prompt and context:
        prompt, context = context, ""
    if not output:
        return []

    input_parts: list[str] = []
    if context and context != prompt:
        input_parts.append(context)
    if tests:
        input_parts.append(f"Tests:\n{tests}")
    if entry_point:
        input_parts.append(f"Entry point: {entry_point}")

    if "humaneval" in source_name.lower():
        instruction = "Lengkapi fungsi Python berikut agar lolos unit test."
        input_text = "\n\n".join(part for part in [prompt, *input_parts] if part)
        task_type = "code_humaneval"
    elif laravel:
        instruction = prompt or "Buat solusi Laravel sesuai spesifikasi berikut."
        input_text = "\n\n".join(input_parts)
        task_type = "laravel_instruction"
    else:
        instruction = prompt or "Selesaikan tugas pemrograman berikut."
        input_text = "\n\n".join(input_parts)
        task_type = "code_instruction"

    built = instruction_row(
        instruction,
        output,
        input_text=input_text,
        system_prompt=LARAVEL_SYSTEM_PROMPT if laravel else CODE_SYSTEM_PROMPT,
        source=source_name,
        task_type=task_type,
    )
    return [built] if built else []


def iter_hf_rows(dataset_name: str, *, config_name: str | None = None, split: str | None = None):
    try:
        from datasets import DatasetDict, __version__ as datasets_version, load_dataset
    except ImportError as exc:
        raise RuntimeError("Install dependency: pip install datasets") from exc

    major_version = int(datasets_version.split(".", 1)[0])
    if dataset_name.startswith("SEACrowd/") and major_version >= 3:
        raise RuntimeError(
            f"{dataset_name} needs HuggingFace datasets 2.x. "
            'Run: pip install "datasets>=2.18,<3" --force-reinstall'
        )

    kwargs: dict[str, Any] = {"trust_remote_code": True, "streaming": True}
    if split:
        kwargs["split"] = split

    try:
        dataset = load_dataset(dataset_name, config_name, **kwargs)
    except Exception as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message or "trust_remote_code" in message:
            raise RuntimeError(
                f"{dataset_name} could not be loaded with datasets {datasets_version}. "
                'Run: pip install "datasets>=2.18,<3" --force-reinstall'
            ) from exc
        raise

    if isinstance(dataset, DatasetDict) or isinstance(dataset, dict):
        for split_name, split_dataset in dataset.items():
            for row in split_dataset:
                row = dict(row)
                row["_split"] = split_name
                yield row
    else:
        for row in dataset:
            yield dict(row)


def mine_hf_dataset(
    dataset_name: str,
    output_path: Path,
    *,
    kind: str,
    config_name: str | None = None,
    split: str | None = None,
    max_items: int | None = None,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
) -> MineStats:
    rows: list[dict[str, Any]] = []
    source_name = dataset_name.replace("/", "__")
    try:
        for raw in iter_hf_rows(dataset_name, config_name=config_name, split=split):
            if kind == "qa":
                converted = row_to_qa_instruction(raw, source_name)
                if converted:
                    rows.append(converted)
            elif kind == "instruction":
                rows.extend(row_to_instruction(raw, source_name))
            elif kind == "code":
                rows.extend(row_to_code_instruction(raw, source_name, laravel=False))
            elif kind == "laravel":
                rows.extend(row_to_code_instruction(raw, source_name, laravel=True))
            elif kind == "text":
                rows.extend(row_to_text_completion(raw, source_name))
            elif kind == "preference":
                rows.extend(row_to_preference_instruction(raw, source_name))
            else:
                raise ValueError(f"Unsupported HF mining kind: {kind}")

            if max_items is not None and len(rows) >= max_items:
                break
    except OSError as exc:
        if getattr(exc, "errno", None) == 28:
            print(f"Skip {dataset_name}: disk full while downloading/extracting.")
            return MineStats(dataset_name, 0, str(output_path))
        print(f"Skip {dataset_name}: {exc}")
        touch_jsonl(output_path)
        return MineStats(dataset_name, 0, str(output_path))
    except Exception as exc:
        print(f"Skip {dataset_name}: {exc}")
        touch_jsonl(output_path)
        return MineStats(dataset_name, 0, str(output_path))

    rows = dedupe_rows(rows[:max_items] if max_items else rows)
    rows = apply_cot_to_rows(rows, cot_ratio=cot_ratio, cot_mode=cot_mode)
    count = append_jsonl(output_path, rows)
    print(f"{dataset_name}: {count} rows -> {output_path}")
    return MineStats(dataset_name, count, str(output_path))


def iter_local_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            yield from (row for row in data if isinstance(row, dict))
        elif isinstance(data, dict):
            for key in ["data", "rows", "train", "validation", "test"]:
                value = data.get(key)
                if isinstance(value, list):
                    yield from (row for row in value if isinstance(row, dict))
        return
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)
        return
    raise ValueError(f"Unsupported local file format: {path}")


def mine_local_qa_file(
    path: Path,
    output_path: Path,
    *,
    max_items: int | None = None,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
) -> MineStats:
    rows: list[dict[str, Any]] = []
    for raw in iter_local_records(path):
        converted = row_to_qa_instruction(raw, path.stem)
        if converted:
            rows.append(converted)
        if max_items is not None and len(rows) >= max_items:
            break
    rows = dedupe_rows(rows)
    rows = apply_cot_to_rows(rows, cot_ratio=cot_ratio, cot_mode=cot_mode)
    count = append_jsonl(output_path, rows)
    print(f"{path}: {count} rows -> {output_path}")
    return MineStats(str(path), count, str(output_path))


def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "id-ID,id;q=0.9,en;q=0.8"})
    return session


def fetch_html(session: requests.Session, url: str, timeout: int = 30) -> str | None:
    try:
        response = session.get(url, timeout=timeout)
        if response.status_code != 200:
            print(f"Skip {url} -> HTTP {response.status_code}")
            return None
        return response.text
    except requests.RequestException as exc:
        print(f"Request failed: {url} -> {exc}")
        return None


def same_domain(url: str, domain: str) -> bool:
    return domain in urlparse(url).netloc


def extract_title(soup: BeautifulSoup) -> str:
    node = soup.select_one("h1") or soup.select_one("title")
    return normalize_text(node.get_text(" ", strip=True)) if node else "Untitled"


def extract_article_text(soup: BeautifulSoup) -> str:
    content = soup.select_one("main") or soup.select_one("article") or soup.body
    if content is None:
        return ""
    for unwanted in content.select("script, style, nav, header, footer"):
        unwanted.decompose()
    blocks: list[str] = []
    for node in content.find_all(["h1", "h2", "h3", "p", "li", "pre", "code"]):
        text = node.get_text("\n" if node.name in {"pre", "code"} else " ", strip=True)
        text = clean_text(text)
        if text and len(text) > 20:
            blocks.append(text)
    return "\n\n".join(blocks)


def chunk_text(text: str, *, max_chars: int = 1800) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not current:
            current = part
        elif len(current) + len(part) + 2 <= max_chars:
            current = f"{current}\n\n{part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def collect_laravel_doc_urls(session: requests.Session, version: str, *, max_pages: int) -> list[str]:
    version_path = version if version.endswith(".x") else f"{version}.x"
    start_url = f"https://laravel.com/docs/{version_path}"
    html = fetch_html(session, start_url)
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    urls = [start_url]
    seen = {start_url}
    for link in soup.select('a[href*="/docs/"]'):
        absolute = urljoin(start_url, link.get("href", "")).split("#", 1)[0]
        if not same_domain(absolute, "laravel.com") or f"/docs/{version_path}" not in absolute:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= max_pages:
            break
    return urls


def mine_laravel_docs(
    output_path: Path,
    *,
    versions: list[str],
    max_pages_per_version: int,
    delay: float,
    max_chunks_per_page: int,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
) -> MineStats:
    session = build_session()
    rows: list[dict[str, Any]] = []
    for version in versions:
        urls = collect_laravel_doc_urls(session, version, max_pages=max_pages_per_version)
        print(f"Laravel {version}: {len(urls)} doc pages")
        for url in urls:
            html = fetch_html(session, url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            title = extract_title(soup)
            for index, chunk in enumerate(chunk_text(extract_article_text(soup))[:max_chunks_per_page], start=1):
                row = instruction_row(
                    f"Jelaskan dokumentasi Laravel {version}: {title}",
                    chunk,
                    input_text=f"URL: {url}\nBagian: {index}",
                    system_prompt=LARAVEL_SYSTEM_PROMPT,
                    source=f"laravel_docs_{version}",
                    task_type="laravel_docs",
                    metadata={"source_url": url, "laravel_version": version},
                )
                if row:
                    rows.append(row)
            time.sleep(delay)
    rows = dedupe_rows(rows)
    rows = apply_cot_to_rows(rows, cot_ratio=cot_ratio, cot_mode=cot_mode)
    count = append_jsonl(output_path, rows)
    print(f"Laravel docs: {count} rows -> {output_path}")
    return MineStats("laravel_docs", count, str(output_path))


def collect_santrikoding_urls(session: requests.Session, *, max_pages: int) -> list[str]:
    start_url = "https://santrikoding.com/tag/laravel"
    pending = [start_url]
    seen_pages: set[str] = set()
    article_urls: list[str] = []
    seen_articles: set[str] = set()
    while pending and len(seen_pages) < max_pages:
        url = pending.pop(0)
        if url in seen_pages:
            continue
        seen_pages.add(url)
        html = fetch_html(session, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.select("a[href]"):
            absolute = urljoin(url, link.get("href", "")).split("#", 1)[0]
            if not same_domain(absolute, "santrikoding.com"):
                continue
            lower = absolute.lower()
            text = normalize_text(link.get_text(" ", strip=True)).lower()
            if ("laravel" in lower or "laravel" in text) and "/tag/" not in lower and absolute not in seen_articles:
                seen_articles.add(absolute)
                article_urls.append(absolute)
            if (
                ("/tag/laravel" in lower or "page" in lower)
                and absolute not in seen_pages
                and absolute not in pending
                and len(seen_pages) + len(pending) < max_pages
            ):
                pending.append(absolute)
        time.sleep(1.0)
    return article_urls


def infer_laravel_version(title: str, url: str, text: str) -> str | None:
    haystack = f"{title} {url} {text[:1000]}".lower()
    for version in ["13", "12", "11", "10", "9"]:
        if re.search(rf"laravel\s+{version}\b|laravel-{version}\b|laravel/{version}\b", haystack):
            return version
    return None


def mine_santrikoding_laravel(
    output_path: Path,
    *,
    max_pages: int,
    max_articles: int,
    delay: float,
    max_chunks_per_article: int,
    cot_ratio: float = 0.0,
    cot_mode: str = "auto",
) -> MineStats:
    session = build_session()
    urls = collect_santrikoding_urls(session, max_pages=max_pages)[:max_articles]
    rows: list[dict[str, Any]] = []
    print(f"SantriKoding Laravel candidate articles: {len(urls)}")
    for url in urls:
        html = fetch_html(session, url)
        if not html:
            continue
        soup = BeautifulSoup(html, "html.parser")
        title = extract_title(soup)
        text = extract_article_text(soup)
        version = infer_laravel_version(title, url, text)
        if version is None:
            continue
        for index, chunk in enumerate(chunk_text(text)[:max_chunks_per_article], start=1):
            row = instruction_row(
                f"Jelaskan tutorial Laravel {version} dari SantriKoding: {title}",
                chunk,
                input_text=f"URL: {url}\nBagian: {index}",
                system_prompt=LARAVEL_SYSTEM_PROMPT,
                source="santrikoding_laravel",
                task_type="laravel_tutorial_id",
                metadata={"source_url": url, "laravel_version": version},
            )
            if row:
                rows.append(row)
        time.sleep(delay)
    rows = dedupe_rows(rows)
    rows = apply_cot_to_rows(rows, cot_ratio=cot_ratio, cot_mode=cot_mode)
    count = append_jsonl(output_path, rows)
    print(f"SantriKoding Laravel: {count} rows -> {output_path}")
    return MineStats("santrikoding_laravel", count, str(output_path))


def reset_outputs(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def write_report(path: Path, stats: list[MineStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"outputs": [stat.__dict__ for stat in stats], "total_rows": sum(stat.rows for stat in stats)}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine Q&A, instruction, code, and Laravel sources into instruction JSONL.")
    parser.add_argument("--preset", choices=["all", "qa", "instruction", "code", "reasoning", "laravel"], default="all")
    parser.add_argument("--output-dir", default="data/mined/instruction")
    parser.add_argument("--max-items", type=int, default=None, help="Limit rows per HF/local source.")
    parser.add_argument("--hf-source", action="append", default=[], help="Extra HF dataset name.")
    parser.add_argument("--hf-kind", choices=["qa", "instruction", "code", "reasoning", "text", "preference", "laravel"], default="instruction")
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument(
        "--include-gated",
        action="store_true",
        help="Also try gated HuggingFace sources. Requires accepted access and huggingface-cli login.",
    )
    parser.add_argument(
        "--include-large-qa",
        action="store_true",
        help="Also mine large QA sources such as SEACrowd/tydiqa_id. Requires more disk space.",
    )
    parser.add_argument("--local-qa-file", action="append", default=[], help="Local JSON/JSONL/CSV QA file.")
    parser.add_argument("--laravel-versions", nargs="+", default=["9", "10", "11", "12", "13"])
    parser.add_argument("--max-laravel-pages", type=int, default=80)
    parser.add_argument("--max-doc-chunks", type=int, default=6)
    parser.add_argument("--max-santrikoding-pages", type=int, default=8)
    parser.add_argument("--max-santrikoding-articles", type=int, default=120)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--cot-ratio", type=float, default=0.0, help="Convert this deterministic fraction of mined rows to CoT format.")
    parser.add_argument("--cot-mode", choices=["auto", "minimal"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    qa_output = output_dir / "indonesian_qa_instruction.jsonl"
    instruction_output = output_dir / "indonesian_general_instruction.jsonl"
    code_output = output_dir / "code_instruction.jsonl"
    reasoning_output = output_dir / "commercial_safe_reasoning_instruction.jsonl"
    laravel_output = output_dir / "laravel_instruction.jsonl"
    report_output = output_dir / "mining_report.json"
    stats: list[MineStats] = []

    reset_outputs([qa_output, instruction_output, code_output, reasoning_output, laravel_output])
    for output_path in [qa_output, instruction_output, code_output, reasoning_output, laravel_output]:
        touch_jsonl(output_path)

    if args.preset in {"all", "qa"}:
        qa_sources = list(HF_QA_SOURCES)
        if args.include_large_qa:
            qa_sources.extend(LARGE_HF_QA_SOURCES)
        for source in qa_sources:
            stats.append(mine_hf_dataset(source, qa_output, kind="qa", max_items=args.max_items, cot_ratio=args.cot_ratio, cot_mode=args.cot_mode))
        for local_path in args.local_qa_file:
            stats.append(mine_local_qa_file(Path(local_path), qa_output, max_items=args.max_items, cot_ratio=args.cot_ratio, cot_mode=args.cot_mode))

    if args.preset in {"all", "instruction"}:
        instruction_sources = list(HF_INSTRUCTION_SOURCES)
        if args.include_gated:
            instruction_sources.extend(GATED_HF_INSTRUCTION_SOURCES)
        for source in instruction_sources:
            stats.append(mine_hf_dataset(source, instruction_output, kind="instruction", max_items=args.max_items, cot_ratio=args.cot_ratio, cot_mode=args.cot_mode))
        for source, config_name, split in HF_COMMERCIAL_SAFE_INSTRUCTION_SOURCES:
            stats.append(
                mine_hf_dataset(
                    source,
                    instruction_output,
                    kind="instruction",
                    config_name=config_name,
                    split=split,
                    max_items=args.max_items,
                    cot_ratio=args.cot_ratio,
                    cot_mode=args.cot_mode,
                )
            )

    if args.preset in {"all", "code"}:
        for source in HF_CODE_EVAL_SOURCES:
            stats.append(
                mine_hf_dataset(
                    source,
                    code_output,
                    kind="code",
                    max_items=args.max_items,
                    cot_ratio=args.cot_ratio,
                    cot_mode=args.cot_mode,
                )
            )

    if args.preset in {"all", "reasoning"}:
        for source, kind, config_name, split in HF_COMMERCIAL_SAFE_REASONING_SOURCES:
            stats.append(
                mine_hf_dataset(
                    source,
                    reasoning_output,
                    kind=kind,
                    config_name=config_name,
                    split=split,
                    max_items=args.max_items,
                    cot_ratio=args.cot_ratio,
                    cot_mode=args.cot_mode,
                )
            )

    for source in args.hf_source:
        if args.hf_kind == "qa":
            output = qa_output
        elif args.hf_kind == "code":
            output = code_output
        elif args.hf_kind in {"reasoning", "text", "preference"}:
            output = reasoning_output
        elif args.hf_kind == "laravel":
            output = laravel_output
        else:
            output = instruction_output
        kind = "qa" if args.hf_kind == "reasoning" else args.hf_kind
        stats.append(
            mine_hf_dataset(
                source,
                output,
                kind=kind,
                config_name=args.hf_config,
                split=args.hf_split,
                max_items=args.max_items,
                cot_ratio=args.cot_ratio,
                cot_mode=args.cot_mode,
            )
        )

    if args.preset in {"all", "laravel"}:
        for source in HF_LARAVEL_DATASET_SOURCES:
            stats.append(
                mine_hf_dataset(
                    source,
                    laravel_output,
                    kind="laravel",
                    max_items=args.max_items,
                    cot_ratio=args.cot_ratio,
                    cot_mode=args.cot_mode,
                )
            )
        stats.append(
            mine_laravel_docs(
                laravel_output,
                versions=args.laravel_versions,
                max_pages_per_version=args.max_laravel_pages,
                delay=args.delay,
                max_chunks_per_page=args.max_doc_chunks,
                cot_ratio=args.cot_ratio,
                cot_mode=args.cot_mode,
            )
        )
        stats.append(
            mine_santrikoding_laravel(
                laravel_output,
                max_pages=args.max_santrikoding_pages,
                max_articles=args.max_santrikoding_articles,
                delay=args.delay,
                max_chunks_per_article=args.max_doc_chunks,
                cot_ratio=args.cot_ratio,
                cot_mode=args.cot_mode,
            )
        )

    write_report(report_output, stats)
    print(f"\nMining complete. Report: {report_output}")
    print(f"Total rows: {sum(stat.rows for stat in stats)}")


if __name__ == "__main__":
    main()
