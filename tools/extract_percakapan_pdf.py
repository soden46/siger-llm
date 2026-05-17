from pathlib import Path
import json
import re

import fitz


PDF_PATH = Path("data/lampung/raw/dataset_1000_percakapan_lampung_dialek_o.pdf")
OUT_PATH = Path("data/lampung/processed/percakapan_1000_pairs.jsonl")
TEXT_OUT_PATH = Path("data/lampung/processed/percakapan_1000_text.txt")

SOURCE = "dataset_1000_percakapan_lampung_dialek_o"
SPEAKER_RE = re.compile(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'. -]{1,30}:")
NUMBER_RE = re.compile(r"^\d{1,4}$")

NOISE_PREFIXES = (
    "Dataset Sintetik Percakapan Lampung Dialek O",
    "Dataset Sintetik 1000 Percakapan",
    "Bahasa Lampung Dialek O",
    "Disusun dari",
    "Tanggal penyusunan",
    "Catatan penting",
    "Berikut 100 dialog",
    "Percakapan Lampung Dialek O",
    "Terjemahan Indonesia",
    "Jumlah dialog",
    "Total dialog",
)

NOISE_EXACT = {
    "No.",
    "Kategori",
    "4.1 Sapaan dan kabar",
    "4.2 Perkenalan",
    "4.3 Rencana pergi",
    "4.4 Sekolah",
    "4.5 Belanja di pasar",
    "4.6 Makan dan rumah",
    "4.7 Kesehatan",
    "4.8 Meminta bantuan",
    "4.9 Kunjungan dan sosial",
    "4.10 Aktivitas harian",
}


def clean_line(line: str) -> str:
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def is_noise(line: str) -> bool:
    if not line:
        return True
    if line in NOISE_EXACT:
        return True
    if any(line.startswith(prefix) for prefix in NOISE_PREFIXES):
        return True
    if re.match(r"^\d+\.\s+", line):
        return True
    if line.startswith("•"):
        return True
    return False


def iter_clean_lines(doc) -> list[str]:
    lines: list[str] = []
    raw_text_parts: list[str] = []

    for page_number, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        raw_text_parts.append(f"\n\n===== PAGE {page_number} =====\n{page_text}")

        for raw_line in page_text.splitlines():
            line = clean_line(raw_line)
            if is_noise(line):
                continue
            lines.append(line)

    TEXT_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TEXT_OUT_PATH.write_text("".join(raw_text_parts), encoding="utf-8")
    return lines


def collect_blocks(lines: list[str]) -> dict[int, list[str]]:
    blocks: dict[int, list[str]] = {}
    current_no: int | None = None

    for line in lines:
        if NUMBER_RE.fullmatch(line):
            number = int(line)
            if 1 <= number <= 1000:
                current_no = number
                blocks.setdefault(current_no, [])
            continue

        if current_no is None:
            continue

        blocks[current_no].append(line)

    return blocks


def merge_continuations(lines: list[str]) -> list[str]:
    merged: list[str] = []

    for line in lines:
        if SPEAKER_RE.match(line):
            merged.append(line)
        elif merged:
            merged[-1] = f"{merged[-1]} {line}"

    return merged


def split_dialogue(lines: list[str]) -> tuple[list[str], list[str]] | None:
    merged = merge_continuations(lines)
    if len(merged) < 2 or len(merged) % 2 != 0:
        return None

    mid = len(merged) // 2
    lampung_lines = merged[:mid]
    indonesian_lines = merged[mid:]

    if len(lampung_lines) != len(indonesian_lines):
        return None

    return lampung_lines, indonesian_lines


def make_rows(dialogue_no: int, lampung_lines: list[str], indonesian_lines: list[str]) -> list[dict]:
    lampung = "\n".join(lampung_lines)
    indonesian = "\n".join(indonesian_lines)

    rows = [
        {
            "dialect": "o",
            "lampung": lampung,
            "indonesian": indonesian,
            "english": "",
            "source": SOURCE,
            "type": "daily_conversation",
            "dialogue_no": dialogue_no,
            "synthetic": True,
        }
    ]

    for turn_index, (lo, indo) in enumerate(zip(lampung_lines, indonesian_lines), start=1):
        rows.append(
            {
                "dialect": "o",
                "lampung": lo,
                "indonesian": indo,
                "english": "",
                "source": SOURCE,
                "type": "conversation_turn",
                "dialogue_no": dialogue_no,
                "turn_index": turn_index,
                "synthetic": True,
            }
        )

    return rows


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    lines = iter_clean_lines(doc)
    blocks = collect_blocks(lines)

    rows: list[dict] = []
    failed: list[int] = []

    for dialogue_no in range(1, 1001):
        split = split_dialogue(blocks.get(dialogue_no, []))
        if split is None:
            failed.append(dialogue_no)
            continue

        lampung_lines, indonesian_lines = split
        rows.extend(make_rows(dialogue_no, lampung_lines, indonesian_lines))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    full_dialogues = sum(1 for row in rows if row["type"] == "daily_conversation")
    turns = sum(1 for row in rows if row["type"] == "conversation_turn")

    print(f"Saved {len(rows)} rows to {OUT_PATH}")
    print(f"Dialogues: {full_dialogues}")
    print(f"Turns    : {turns}")
    print(f"Failed   : {len(failed)}")
    if failed:
        print(f"Failed dialogue numbers preview: {failed[:30]}")

    print("\nPreview:")
    for row in rows[:6]:
        print(f"- [{row['type']}] {row['lampung']} -> {row['indonesian']}")


if __name__ == "__main__":
    main()
