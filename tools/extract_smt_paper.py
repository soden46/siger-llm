from pathlib import Path
import fitz

PDF_PATH = Path("data/lampung/raw/smt_lampung_nyo_paper.pdf")
OUT_PATH = Path("data/lampung/processed/smt_paper_text.txt")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

doc = fitz.open(PDF_PATH)
texts = []

for page_num, page in enumerate(doc, start=1):
    texts.append(f"\n\n===== PAGE {page_num} =====\n")
    texts.append(page.get_text("text"))

OUT_PATH.write_text("".join(texts), encoding="utf-8")

print(f"Saved to {OUT_PATH}")