from pathlib import Path
import json
import re
import unicodedata

INPUT_PATH = Path("data/lampung/final/lampung_o_trilingual.jsonl")
OUTPUT_PATH = Path("data/lampung/final/lampung_o_trilingual_normalized.jsonl")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Lowercase
    text = text.lower()

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Normalize quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # Remove weird spaces
    text = text.strip()

    return text


rows = []

with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)

        row["lampung_o"] = normalize_text(
            row.get("lampung_o", "")
        )

        row["indonesian"] = normalize_text(
            row.get("indonesian", "")
        )

        row["english"] = normalize_text(
            row.get("english", "")
        )

        rows.append(row)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(
            json.dumps(row, ensure_ascii=False) + "\n"
        )

print(f"Normalized {len(rows)} rows")
print(f"Saved to: {OUTPUT_PATH}")