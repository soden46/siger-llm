# tools/scrape_rajotuho.py
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIG
# ============================================================
CATEGORY_URL = "https://rajotuho.com/category/bahasa-lampung/"
BASE_DOMAIN = "rajotuho.com"

OUT_JSONL = Path("data/lampung/raw/rajotuho_pairs.jsonl")
OUT_REPORT = Path("data/lampung/processed/rajotuho_scrape_report.json")

MAX_CATEGORY_PAGES = 10
REQUEST_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = 30

USER_AGENT = (
    "SIGER-LLM-DatasetBuilder/1.0 "
    "(educational dataset curation; contact via project repository)"
)

# Scraper difokuskan ke Dialek O.
# Artikel yang jelas campur A/O tetap bisa diparse, tapi yang diambil hanya bagian (O).
ONLY_DIALECT_O = True


# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class PairRecord:
    dialect: str
    lampung: str
    indonesian: str
    english: str
    source: str
    source_url: str
    type: str
    topic: str
    extraction_method: str
    title: str

    def to_dict(self) -> dict:
        return {
            "dialect": self.dialect,
            "lampung": self.lampung,
            "indonesian": self.indonesian,
            "english": self.english,
            "source": self.source,
            "source_url": self.source_url,
            "type": self.type,
            "topic": self.topic,
            "extraction_method": self.extraction_method,
            "title": self.title,
        }


# ============================================================
# HTTP
# ============================================================
def build_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "id-ID,id;q=0.9,en;q=0.8",
        }
    )
    return session


def fetch_html(session: requests.Session, url: str) -> str | None:
    try:
        response = session.get(url, timeout=TIMEOUT_SECONDS)
        if response.status_code != 200:
            print(f"⚠️  Skip {url} -> HTTP {response.status_code}")
            return None
        return response.text
    except requests.RequestException as exc:
        print(f"⚠️  Request failed: {url} -> {exc}")
        return None


# ============================================================
# TEXT HELPERS
# ============================================================
def normalize_space(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_dialect_tag_only(text: str) -> bool:
    value = normalize_space(text).upper().replace(" ", "")
    return value in {
        "O",
        "A",
        "O/A",
        "A/O",
        "(O)",
        "(A)",
        "(O/A)",
        "(A/O)",
    }


def strip_speaker_prefix(text: str) -> str:
    """
    Hapus prefix seperti:
    Rita:
    A:
    Medi:
    """
    text = normalize_space(text)
    text = re.sub(r"^[A-Za-zÀ-ÖØ-öø-ÿ0-9 .'-]{1,40}:\s*", "", text)
    return text.strip()


def clean_pair_text(text: str) -> str:
    text = strip_speaker_prefix(text)
    text = text.strip(" \t\n\r\"'“”‘’")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_good_pair(lampung: str, indonesian: str) -> bool:
    lampung = clean_pair_text(lampung)
    indonesian = clean_pair_text(indonesian)

    if not lampung or not indonesian:
        return False

    if is_dialect_tag_only(lampung) or is_dialect_tag_only(indonesian):
        return False

    if len(lampung) < 2 or len(indonesian) < 2:
        return False

    if len(lampung) > 500 or len(indonesian) > 700:
        return False

    bad_fragments = (
        "baca juga",
        "read more",
        "previous article",
        "next article",
        "share on",
        "tweet on",
    )

    merged = f"{lampung} {indonesian}".lower()
    if any(fragment in merged for fragment in bad_fragments):
        return False

    return True


def infer_topic(title: str) -> str:
    t = title.lower()

    if "pasar" in t:
        return "market"
    if "sakit" in t:
        return "health"
    if "sekolah" in t:
        return "school"
    if "salam" in t or "apa kabar" in t:
        return "greeting"
    if "perkenalan" in t:
        return "introduction"
    if "waktu" in t:
        return "time"
    if "tempat" in t:
        return "location"
    if "percakapan" in t:
        return "conversation"

    return "language_learning"


def page_mentions_dialect_o(title: str, page_text: str) -> bool:
    haystack = f"{title} {page_text}".lower()
    markers = [
        "dialek o",
        "nyow",
        "(o)",
        "bahasa lampung dialek o",
    ]
    return any(marker in haystack for marker in markers)


# ============================================================
# CATEGORY CRAWLER
# ============================================================
def category_page_url(page: int) -> str:
    if page == 1:
        return CATEGORY_URL
    return urljoin(CATEGORY_URL, f"page/{page}/")


def collect_article_urls(session: requests.Session) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()

    for page in range(1, MAX_CATEGORY_PAGES + 1):
        url = category_page_url(page)
        print(f"📚 Reading category page: {url}")

        html = fetch_html(session, url)
        if not html:
            break

        soup = BeautifulSoup(html, "html.parser")

        # Umumnya daftar artikel muncul sebagai heading link.
        candidate_links = []

        for selector in ["h2 a[href]", "h3 a[href]", "article a[href]"]:
            candidate_links.extend(soup.select(selector))

        page_new_count = 0

        for a in candidate_links:
            href = a.get("href", "").strip()
            if not href:
                continue

            absolute = urljoin(url, href)
            parsed = urlparse(absolute)

            if parsed.netloc and BASE_DOMAIN not in parsed.netloc:
                continue

            # Skip kategori, tag, author, pagination, gambar.
            if any(
                part in absolute
                for part in [
                    "/category/",
                    "/tag/",
                    "/author/",
                    "/page/",
                    "/wp-content/",
                    "#",
                ]
            ):
                continue

            if absolute in seen:
                continue

            # Fokus ke artikel bahasa Lampung
            anchor_text = normalize_space(a.get_text(" ", strip=True)).lower()
            href_lower = absolute.lower()

            if not (
                "lampung" in anchor_text
                or "lampung" in href_lower
                or "bahasa" in anchor_text
            ):
                continue

            seen.add(absolute)
            urls.append(absolute)
            page_new_count += 1

        print(f"   ↳ found {page_new_count} new article URLs")

        # Jika halaman pagination tidak menambah artikel sama sekali, stop.
        if page > 1 and page_new_count == 0:
            break

        time.sleep(REQUEST_DELAY_SECONDS)

    return urls


# ============================================================
# ARTICLE TEXT EXTRACTION
# ============================================================
def extract_article_title(soup: BeautifulSoup) -> str:
    for selector in ["h1.entry-title", "h1.post-title", "h1"]:
        node = soup.select_one(selector)
        if node:
            return normalize_space(node.get_text(" ", strip=True))
    return "Untitled Rajotuho Article"


def extract_content_blocks(soup: BeautifulSoup) -> list[str]:
    """
    Ambil paragraf/list yang mungkin berisi pasangan bahasa.
    """
    content = (
        soup.select_one(".entry-content")
        or soup.select_one(".post-content")
        or soup.select_one("article")
        or soup.body
    )

    if content is None:
        return []

    blocks: list[str] = []

    for node in content.find_all(["p", "li", "h2", "h3", "h4"]):
        text = normalize_space(node.get_text(" ", strip=True))
        if text:
            blocks.append(text)

    return blocks


# ============================================================
# EXTRACTION METHODS
# ============================================================

def extract_o_a_i_pattern(
    block: str,
    *,
    title: str,
    url: str,
    topic: str,
) -> list[PairRecord]:
    """
    Pattern:
    A: Nyow kabar meu? ... (O). Api kabar mu? ... (A). Apa kabar kamu? ... (I)

    Yang kita ambil:
    Lampung O -> Indonesia
    """
    results: list[PairRecord] = []

    pattern = re.compile(
        r"""
        (?P<lampung_o>.+?)
        \s*\(O\)\.?
        \s*
        (?:
            .+?
            \s*\(A\)\.?
            \s*
        )?
        (?P<indo>.+?)
        \s*\(I\)\.?
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for match in pattern.finditer(block):
        lampung = clean_pair_text(match.group("lampung_o"))
        indo = clean_pair_text(match.group("indo"))

        if not is_good_pair(lampung, indo):
            continue

        results.append(
            PairRecord(
                dialect="o",
                lampung=lampung,
                indonesian=indo,
                english="",
                source="rajotuho",
                source_url=url,
                type="sentence_pair",
                topic=topic,
                extraction_method="o_a_i_pattern",
                title=title,
            )
        )

    return results


def extract_indonesia_equals_o_pattern(
    block: str,
    *,
    title: str,
    url: str,
    topic: str,
) -> list[PairRecord]:
    """
    Pattern:
    Aku bertemu Jamal di sekolah = Nyak tunggo Jamal di sekulah (O). Nyak tungga ... (A).

    Yang kita ambil:
    Nyak tunggo Jamal di sekulah -> Aku bertemu Jamal di sekolah
    """
    results: list[PairRecord] = []

    pattern = re.compile(
        r"""
        (?P<indo>.+?)
        \s*=\s*
        (?P<lampung_o>.+?)
        \s*\(O\)
        """,
        re.IGNORECASE | re.VERBOSE,
    )

    for match in pattern.finditer(block):
        indo = clean_pair_text(match.group("indo"))
        lampung = clean_pair_text(match.group("lampung_o"))

        if not is_good_pair(lampung, indo):
            continue

        results.append(
            PairRecord(
                dialect="o",
                lampung=lampung,
                indonesian=indo,
                english="",
                source="rajotuho",
                source_url=url,
                type="sentence_pair",
                topic=topic,
                extraction_method="indonesia_equals_o_pattern",
                title=title,
            )
        )

    return results


def extract_lampung_parentheses_indonesia_pattern(
    block: str,
    *,
    title: str,
    url: str,
    topic: str,
    allow_untagged: bool,
) -> list[PairRecord]:
    """
    Pattern:
    Rita: Hai, Sapo namo sekam kiyay? (Hai, Siapa nama anda kiyay-abang?)

    Dipakai hanya pada halaman yang jelas Dialek O / Nyow.
    """
    if not allow_untagged:
        return []

    results: list[PairRecord] = []

    # Mengambil kalimat sebelum tanda kurung terakhir sebagai Lampung
    # dan isi tanda kurung sebagai terjemahan Indonesia.
    pattern = re.compile(
        r"""
        ^(?P<lampung>.+?)
        \s*
        \(
        (?P<indo>[^()]{2,700})
        \)
        \s*$
        """,
        re.VERBOSE,
    )

    match = pattern.match(block)
    if not match:
        return results

    lampung = clean_pair_text(match.group("lampung"))
    indo = clean_pair_text(match.group("indo"))

    if is_dialect_tag_only(indo):
        return results

    # Hindari salah parse struktur O/A/I yang sudah ditangani method khusus.
    if "(O)" in block or "(A)" in block or "(I)" in block:
        return results

    if not is_good_pair(lampung, indo):
        return results

    results.append(
        PairRecord(
            dialect="o",
            lampung=lampung,
            indonesian=indo,
            english="",
            source="rajotuho",
            source_url=url,
            type="daily_conversation",
            topic=topic,
            extraction_method="parentheses_translation_pattern",
            title=title,
        )
    )

    return results


def extract_o_equals_indonesia_pattern(
    block: str,
    *,
    title: str,
    url: str,
    topic: str,
    allow_untagged: bool,
) -> list[PairRecord]:
    """
    Pattern:
    Tiyan agow dak nei = mereka mau kesana

    Ini hanya aman dipakai di artikel yang jelas membahas Dialek O.
    """
    if not allow_untagged:
        return []

    results: list[PairRecord] = []

    if "(O)" in block or "(A)" in block or "(I)" in block:
        return results

    if "=" not in block:
        return results

    left, right = block.split("=", 1)

        # Kalau sisi kanan masih mengandung marker dialek,
    # berarti formatnya kemungkinan:
    # Indonesia = Lampung O (O), Lampung A (A)
    # dan sudah ditangani extractor lain.
    if re.search(r"\((?:O|A|O/A|A/O)\)", right, re.IGNORECASE):
        return results
    
    lampung = clean_pair_text(left)
    indo = clean_pair_text(right)

    # Heuristik: contoh dari situs biasanya Bahasa Indonesia di kanan
    # ketika artikel membahas kosakata/percakapan O.
    if not is_good_pair(lampung, indo):
        return results

    results.append(
        PairRecord(
            dialect="o",
            lampung=lampung,
            indonesian=indo,
            english="",
            source="rajotuho",
            source_url=url,
            type="sentence_pair",
            topic=topic,
            extraction_method="o_equals_indonesia_pattern",
            title=title,
        )
    )

    return results


# ============================================================
# ARTICLE PROCESSOR
# ============================================================
def extract_pairs_from_article(
    session: requests.Session,
    url: str,
) -> tuple[list[PairRecord], dict]:
    html = fetch_html(session, url)
    if not html:
        return [], {
            "url": url,
            "title": "",
            "status": "fetch_failed",
            "pairs": 0,
        }

    soup = BeautifulSoup(html, "html.parser")
    title = extract_article_title(soup)
    blocks = extract_content_blocks(soup)
    page_text = " ".join(blocks)
    topic = infer_topic(title)
    has_dialect_o_marker = page_mentions_dialect_o(title, page_text)

    records: list[PairRecord] = []

    for block in blocks:
        # Structured forms can be extracted regardless,
        # because they explicitly tag Dialek O.
        records.extend(
            extract_o_a_i_pattern(
                block,
                title=title,
                url=url,
                topic=topic,
            )
        )

        records.extend(
            extract_indonesia_equals_o_pattern(
                block,
                title=title,
                url=url,
                topic=topic,
            )
        )

        # Untagged forms only when page clearly indicates Dialek O/Nyow.
        records.extend(
            extract_lampung_parentheses_indonesia_pattern(
                block,
                title=title,
                url=url,
                topic=topic,
                allow_untagged=has_dialect_o_marker or not ONLY_DIALECT_O,
            )
        )

        records.extend(
            extract_o_equals_indonesia_pattern(
                block,
                title=title,
                url=url,
                topic=topic,
                allow_untagged=has_dialect_o_marker or not ONLY_DIALECT_O,
            )
        )

    report = {
        "url": url,
        "title": title,
        "status": "ok",
        "topic": topic,
        "dialect_o_marker": has_dialect_o_marker,
        "blocks": len(blocks),
        "pairs": len(records),
    }

    return records, report


# ============================================================
# DEDUP & SAVE
# ============================================================
def deduplicate(records: Iterable[PairRecord]) -> list[PairRecord]:
    output: list[PairRecord] = []
    seen: set[tuple[str, str, str]] = set()

    for record in records:
        key = (
            record.dialect.lower(),
            normalize_space(record.lampung).lower(),
            normalize_space(record.indonesian).lower(),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(record)

    return output


def save_jsonl(records: list[PairRecord]) -> None:
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def save_report(report: dict) -> None:
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    print("🌐 Scraping Rajotuho Bahasa Lampung category...")
    print(f"📂 Output pairs : {OUT_JSONL}")
    print(f"📋 Output report: {OUT_REPORT}\n")

    session = build_session()

    article_urls = collect_article_urls(session)
    print(f"\n🔗 Total candidate article URLs: {len(article_urls)}\n")

    all_records: list[PairRecord] = []
    article_reports: list[dict] = []

    for index, url in enumerate(article_urls, start=1):
        print(f"[{index}/{len(article_urls)}] Scraping: {url}")

        records, report = extract_pairs_from_article(session, url)
        all_records.extend(records)
        article_reports.append(report)

        print(
            f"   ↳ {report.get('title', 'Untitled')} "
            f"| pairs={report.get('pairs', 0)} "
            f"| dialect_o_marker={report.get('dialect_o_marker', False)}"
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    clean_records = deduplicate(all_records)

    save_jsonl(clean_records)
    save_report(
        {
            "category_url": CATEGORY_URL,
            "articles_found": len(article_urls),
            "raw_pairs_found": len(all_records),
            "deduplicated_pairs_saved": len(clean_records),
            "output_jsonl": str(OUT_JSONL),
            "articles": article_reports,
        }
    )

    print("\n✅ Rajotuho scraping complete!")
    print(f"   Raw pairs found       : {len(all_records)}")
    print(f"   Unique pairs saved    : {len(clean_records)}")
    print(f"   Dataset output        : {OUT_JSONL}")
    print(f"   Scrape report         : {OUT_REPORT}")

    if clean_records:
        print("\nPreview 10 rows:")
        for record in clean_records[:10]:
            print(f"- {record.lampung} -> {record.indonesian}")


if __name__ == "__main__":
    main()