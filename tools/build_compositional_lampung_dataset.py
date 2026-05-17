from pathlib import Path
import json
import itertools


OUT_PATH = Path("data/lampung/processed/compositional_pairs.jsonl")
SOURCE = "synthetic_compositional_lampung_o"


SUBJECTS = {
    "saya": ("Nyak", "i"),
    "aku": ("Nyak", "i"),
    "kamu": ("Nikeu", "you"),
    "dia": ("Yo", "he/she"),
    "kita": ("Gham", "we"),
    "kami": ("Ikam", "we"),
    "kalian": ("Mettei", "you all"),
    "mereka": ("Tiyan", "they"),
}

FOODS = {
    "nasi": ("nasi", "rice"),
    "ikan": ("punyeu", "fish"),
    "ikan bakar": ("punyeu puppul", "grilled fish"),
    "ayam": ("manuk", "chicken"),
    "durian": ("deghian", "durian"),
    "rambutan": ("rambutan", "rambutan"),
}

ITEMS = {
    "buku": ("buku", "a book"),
    "baju": ("kawai", "clothes"),
    "ikan": ("punyeu", "fish"),
    "cabai": ("cabik", "chili"),
    "air": ("air", "water"),
    "obat": ("obat", "medicine"),
}

PLACES = {
    "rumah": ("nuwo", "home"),
    "pasar": ("pasar", "the market"),
    "warung": ("warung", "the stall"),
    "sekolah": ("sekulah", "school"),
    "kelas": ("kelas", "class"),
    "masjid": ("masjid", "the mosque"),
}

NEAR_PLACES = {
    "jalan": ("jalan", "the road"),
    "rumah": ("nuwo", "home"),
    "pasar": ("pasar", "the market"),
    "sungai": ("way", "the river"),
}

TIMES = {
    "sekarang": ("tano", "now"),
    "nanti": ("na'en", "later"),
    "hari ini": ("dawah ijo", "today"),
    "besok": ("jemeh", "tomorrow"),
}


def row(indonesian: str, lampung: str, english: str, row_type: str) -> dict:
    return {
        "dialect": "o",
        "lampung": lampung,
        "indonesian": indonesian,
        "english": english,
        "source": SOURCE,
        "type": row_type,
        "synthetic": True,
    }


def want_verb(subject_en: str) -> str:
    return "wants" if subject_en == "he/she" else "want"


def build_rows() -> list[dict]:
    rows: list[dict] = []

    for subj_id, (subj_lo, subj_en) in SUBJECTS.items():
        for food_id, (food_lo, food_en) in FOODS.items():
            rows.append(
                row(
                    f"{subj_id} mau makan {food_id}",
                    f"{subj_lo} haga mengan {food_lo}",
                    f"{subj_en} {want_verb(subj_en)} to eat {food_en}",
                    "synthetic_composition",
                )
            )

            for place_id, (place_lo, place_en) in PLACES.items():
                rows.append(
                    row(
                        f"{subj_id} mau makan {food_id} di {place_id}",
                        f"{subj_lo} haga mengan {food_lo} di {place_lo}",
                        f"{subj_en} {want_verb(subj_en)} to eat {food_en} at {place_en}",
                        "synthetic_composition",
                    )
                )

                for near_id, (near_lo, near_en) in NEAR_PLACES.items():
                    if near_id == place_id:
                        continue
                    rows.append(
                        row(
                            f"{subj_id} mau makan {food_id} di {place_id} dekat {near_id}",
                            f"{subj_lo} haga mengan {food_lo} di {place_lo} paghek {near_lo}",
                            f"{subj_en} {want_verb(subj_en)} to eat {food_en} at {place_en} near {near_en}",
                            "synthetic_composition",
                        )
                    )

    for subj_id, (subj_lo, subj_en) in SUBJECTS.items():
        for item_id, (item_lo, item_en) in ITEMS.items():
            rows.append(
                row(
                    f"{subj_id} mau membeli {item_id}",
                    f"{subj_lo} ago belei {item_lo}",
                    f"{subj_en} {want_verb(subj_en)} to buy {item_en}",
                    "synthetic_composition",
                )
            )
            rows.append(
                row(
                    f"{subj_id} mau beli {item_id}",
                    f"{subj_lo} ago belei {item_lo}",
                    f"{subj_en} {want_verb(subj_en)} to buy {item_en}",
                    "synthetic_composition",
                )
            )

            for place_id, (place_lo, place_en) in PLACES.items():
                rows.append(
                    row(
                        f"{subj_id} mau membeli {item_id} di {place_id}",
                        f"{subj_lo} ago belei {item_lo} di {place_lo}",
                        f"{subj_en} {want_verb(subj_en)} to buy {item_en} at {place_en}",
                        "synthetic_composition",
                    )
                )

    for subj_id, (subj_lo, subj_en) in SUBJECTS.items():
        for place_id, (place_lo, place_en) in PLACES.items():
            for time_id, (time_lo, time_en) in TIMES.items():
                rows.append(
                    row(
                        f"{subj_id} mau ke {place_id} {time_id}",
                        f"{subj_lo} ago dak {place_lo} {time_lo}",
                        f"{subj_en} {want_verb(subj_en)} to go to {place_en} {time_en}",
                        "synthetic_composition",
                    )
                )

    return rows


def dedupe(rows: list[dict]) -> list[dict]:
    seen = set()
    deduped = []

    for item in rows:
        key = (item["lampung"].lower(), item["indonesian"].lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    return deduped


def main() -> None:
    rows = dedupe(build_rows())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows)} compositional rows to {OUT_PATH}")
    for item in itertools.islice(rows, 8):
        print(f"- {item['indonesian']} -> {item['lampung']}")


if __name__ == "__main__":
    main()
