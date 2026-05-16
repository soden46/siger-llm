from pathlib import Path
import json
import re
import fitz


# ============================================================
# PATH CONFIG
# ============================================================
PDF_PATH = Path("data/lampung/raw/kamus_lampung_o.pdf")
OUT_PATH = Path("data/lampung/processed/kamus_pairs.jsonl")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

# Entri kamus mulai efektif di halaman PDF ke-7
# PyMuPDF index 0-based, jadi 6 = halaman 7
PAGE_START_INDEX = 6

# POS yang muncul di kamus:
# n. noun
# v. verb
# a. adjective
# dst.
POS_PATTERN = r"(?:n|v|a|adj|adv|num|pron|p|prep|konj|interj)"

# Pattern entri:
# "apeng n. kain pembatas ..."
# "akuk way v. permainan ..."
#
# Headword boleh mengandung:
# - huruf
# - angka
# - spasi
# - tanda kurung
# - slash
# - tanda petik
# - hyphen
ENTRY_PATTERN = re.compile(
    rf"""
    (?P<headword>
        [A-Za-zÀ-ÖØ-öø-ÿ0-9'()/\- ]{{1,80}}?
    )
    \s+
    (?P<pos>{POS_PATTERN})
    \.
    \s*
    (?P<definition>
        .*?
    )
    (?=
        \s+
        [A-Za-zÀ-ÖØ-öø-ÿ0-9'()/\- ]{{1,80}}?
        \s+
        {POS_PATTERN}
        \.
        |
        $
    )
    """,
    re.VERBOSE | re.IGNORECASE | re.DOTALL,
)


# ============================================================
# CLEANING HELPERS
# ============================================================

def clean_text(text: str) -> str:
    """
    Normalisasi teks hasil ekstraksi PDF.
    """
    text = text.replace("\u00a0", " ")
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_field(text: str) -> str:
    text = clean_text(text)
    text = text.strip(" ,;:")
    return text


def remove_page_noise(text: str) -> str:
    """
    Hapus artefak header/footer/nomor halaman yang sering ikut masuk.
    """
    patterns = [
        r"\d{1,2}/\d{1,2}/\d{4}\s+\d+",
        r"buah penyaro buho",
        r"kamus budaya lampung indonesia",
        r"lampung - indonesia",
        r"dialek o",
    ]

    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_definition(text: str) -> str:
    """
    Rapikan definisi kamus.
    """
    text = clean_field(text)

    # Hapus sisa nomor halaman yang kadang nyelip
    text = re.sub(r"\b\d{1,2}/\d{1,2}/\d{4}\s+\d+\b", " ", text)

    # Rapikan spasi
    text = re.sub(r"\s+", " ", text).strip()

    return text


def is_valid_row(row: dict) -> bool:
    lampung = row["lampung"].strip()
    indonesian = row["indonesian"].strip()

    if not lampung or not indonesian:
        return False

    if len(lampung) < 2:
        return False

    if len(lampung) > 80:
        return False

    if indonesian in {"?", "-", "—"}:
        return False

    # Skip kemungkinan noise heading
    noise_headwords = {
        "kamus",
        "korpus",
        "dialek",
        "lampung indonesia",
    }

    if lampung.lower() in noise_headwords:
        return False

    return True


# ============================================================
# MAIN EXTRACTOR
# ============================================================

def extract_entries_from_page(page_text: str, page_number: int) -> list[dict]:
    page_text = clean_text(page_text)
    page_text = remove_page_noise(page_text)

    rows = []

    for match in ENTRY_PATTERN.finditer(page_text):
        headword = clean_field(match.group("headword")).lower()
        pos = clean_field(match.group("pos")).lower()
        definition = normalize_definition(match.group("definition")).lower()

        row = {
            "dialect": "o",
            "lampung": headword,
            "indonesian": definition,
            "english": "",
            "pos": pos,
            "source": "kamus_budaya_lampung_o",
            "type": "dictionary",
            "page": page_number,
        }

        if is_valid_row(row):
            rows.append(row)

    return rows


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF tidak ditemukan: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)

    all_rows = []
    seen = set()

    for page_index in range(PAGE_START_INDEX, len(doc)):
        page_number = page_index + 1
        page = doc[page_index]

        page_rows = extract_entries_from_page(
            page.get_text("text"),
            page_number,
        )

        for row in page_rows:
            key = (
                row["dialect"],
                row["lampung"],
                row["indonesian"],
            )

            if key in seen:
                continue

            seen.add(key)
            all_rows.append(row)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(all_rows)} cleaned rows to {OUT_PATH}")

    print("\nPreview 30 rows pertama:")
    for row in all_rows[:30]:
        print(
            f"- {row['lampung']} [{row['pos']}.] "
            f"-> {row['indonesian']}"
        )


if __name__ == "__main__":
    main()