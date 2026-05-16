from pathlib import Path
import json
import random
import re
import unicodedata


# ============================================================
# PATH CONFIG
# ============================================================
BASE_DIR = Path("data/lampung")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed"
FINAL_DIR = BASE_DIR / "final"

FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT SOURCES
# ============================================================
INPUT_FILES = [
    # Hasil ekstraksi Kamus Budaya Lampung Dialek O
    PROCESSED_DIR / "kamus_pairs.jsonl",

    #Hasil scarpe web ranjotuho
    RAW_DIR / "rajotuho_pairs.jsonl",

    # Pasangan kalimat Nyo dari paper SMT
    RAW_DIR / "smt_pairs.jsonl",

    # Pasangan manual tervalidasi
    RAW_DIR / "manual_pairs.jsonl",

    # Dataset trilingual awal jika ada
    FINAL_DIR / "lampung_o_trilingual_normalized.jsonl",
]


# ============================================================
# HELPERS
# ============================================================
def normalize_text(text: str) -> str:
    """
    Normalisasi teks ringan agar konsisten.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def read_jsonl(path: Path) -> list[dict]:
    """
    Baca file JSONL. Kalau file tidak ada, skip dengan aman.
    """
    if not path.exists():
        print(f"⚠️  Skip missing file: {path}")
        return []

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠️  Invalid JSON skipped: {path}:{line_number} -> {e}")

    print(f"📥 Loaded {len(rows)} rows from {path}")
    return rows


def standardize_row(row: dict) -> dict | None:
    """
    Menyamakan format semua sumber menjadi:
    {
      dialect,
      lampung,
      indonesian,
      english,
      source,
      type,
      pos?
    }
    """

    dialect = normalize_text(row.get("dialect", "o")).lower()

    # Support format lama dan baru
    lampung = normalize_text(
        row.get("lampung", row.get("lampung_o", ""))
    )

    indonesian = normalize_text(
        row.get("indonesian", "")
    )

    english = normalize_text(
        row.get("english", "")
    )

    source = normalize_text(
        row.get("source", "unknown")
    )

    row_type = normalize_text(
        row.get("type", "sentence_pair")
    )

    pos = normalize_text(
        row.get("pos", "")
    )

    # Skip data invalid
    if not lampung or not indonesian:
        return None

    standardized = {
        "dialect": dialect,
        "lampung": lampung,
        "indonesian": indonesian,
        "english": english,
        "source": source,
        "type": row_type,
    }

    if pos:
        standardized["pos"] = pos

    return standardized


def deduplicate_rows(rows: list[dict]) -> list[dict]:
    """
    Hilangkan duplikasi berdasarkan pasangan inti.
    """
    cleaned = []
    seen = set()

    for row in rows:
        normalized = standardize_row(row)

        if normalized is None:
            continue

        key = (
            normalized["dialect"].lower(),
            normalized["lampung"].lower(),
            normalized["indonesian"].lower(),
            normalized["english"].lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(normalized)

    return cleaned


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """
    Simpan list dict ke JSONL.
    """
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# MAIN
# ============================================================
def main():
    all_rows = []

    print("🚧 Building Lampung dataset...\n")

    for file_path in INPUT_FILES:
        all_rows.extend(read_jsonl(file_path))

    cleaned_rows = deduplicate_rows(all_rows)

    random.seed(42)
    random.shuffle(cleaned_rows)

    total = len(cleaned_rows)

    train_end = int(total * 0.8)
    valid_end = int(total * 0.9)

    train_rows = cleaned_rows[:train_end]
    valid_rows = cleaned_rows[train_end:valid_end]
    test_rows = cleaned_rows[valid_end:]

    write_jsonl(FINAL_DIR / "train.jsonl", train_rows)
    write_jsonl(FINAL_DIR / "valid.jsonl", valid_rows)
    write_jsonl(FINAL_DIR / "test.jsonl", test_rows)

    print("\n✅ Dataset build complete!")
    print(f"Total: {total}")
    print(f"Train: {len(train_rows)}")
    print(f"Valid: {len(valid_rows)}")
    print(f"Test : {len(test_rows)}")

    print("\n📁 Output:")
    print(f"- {FINAL_DIR / 'train.jsonl'}")
    print(f"- {FINAL_DIR / 'valid.jsonl'}")
    print(f"- {FINAL_DIR / 'test.jsonl'}")


if __name__ == "__main__":
    main()