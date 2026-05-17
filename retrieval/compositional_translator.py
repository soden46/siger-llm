import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompositionalResult:
    output: str
    source: str = "compositional rule"


class LampungCompositionalTranslator:
    """Small rule-based composer for common Lampung translation phrases."""

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

    def translate_id_to_lo(self, text: str) -> Optional[CompositionalResult]:
        normalized = self._norm(text)

        result = self._translate_want_to_eat(normalized)
        if result:
            return result

        result = self._translate_want_to_buy(normalized)
        if result:
            return result

        return self._translate_want_to_go(normalized)

    def translate_lo_to_en(self, text: str) -> Optional[CompositionalResult]:
        normalized = self._norm(text)

        result = self._translate_lo_eat_to_en(normalized)
        if result:
            return result

        result = self._translate_lo_buy_to_en(normalized)
        if result:
            return result

        return self._translate_lo_go_to_en(normalized)

    def _translate_want_to_eat(self, text: str) -> Optional[CompositionalResult]:
        pattern = (
            r"^(?P<subject>.+?) mau makan (?P<food>.+?)"
            r"(?: di (?P<place>.+?))?"
            r"(?: dekat (?P<near>.+?))?$"
        )
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_id(self.SUBJECTS, match.group("subject"))
        food = self._lookup_id(self.FOODS, match.group("food"))
        place = self._lookup_optional_id(self.PLACES, match.group("place"))
        near = self._lookup_optional_id(self.NEAR_PLACES, match.group("near"))

        if not subject or not food:
            return None

        parts = [subject, "haga", "mengan", food]
        if match.group("place"):
            if not place:
                return None
            parts.extend(["di", place])
        if match.group("near"):
            if not near:
                return None
            parts.extend(["paghek", near])

        return CompositionalResult(" ".join(parts))

    def _translate_want_to_buy(self, text: str) -> Optional[CompositionalResult]:
        pattern = r"^(?P<subject>.+?) mau (?:membeli|beli) (?P<item>.+?)(?: di (?P<place>.+?))?$"
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_id(self.SUBJECTS, match.group("subject"))
        item = self._lookup_id(self.ITEMS, match.group("item"))
        place = self._lookup_optional_id(self.PLACES, match.group("place"))

        if not subject or not item:
            return None

        parts = [subject, "ago", "belei", item]
        if match.group("place"):
            if not place:
                return None
            parts.extend(["di", place])

        return CompositionalResult(" ".join(parts))

    def _translate_want_to_go(self, text: str) -> Optional[CompositionalResult]:
        pattern = r"^(?P<subject>.+?) mau ke (?P<place>.+?)(?: (?P<time>sekarang|nanti|hari ini|besok))?$"
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_id(self.SUBJECTS, match.group("subject"))
        place = self._lookup_id(self.PLACES, match.group("place"))
        time = self._lookup_optional_id(self.TIMES, match.group("time"))

        if not subject or not place:
            return None

        parts = [subject, "ago", "dak", place]
        if match.group("time"):
            if not time:
                return None
            parts.append(time)

        return CompositionalResult(" ".join(parts))

    def _translate_lo_eat_to_en(self, text: str) -> Optional[CompositionalResult]:
        pattern = (
            r"^(?P<subject>.+?) haga mengan (?P<food>.+?)"
            r"(?: di (?P<place>.+?))?"
            r"(?: paghek (?P<near>.+?))?$"
        )
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_lo(self.SUBJECTS, match.group("subject"))
        food = self._lookup_lo(self.FOODS, match.group("food"))
        place = self._lookup_optional_lo(self.PLACES, match.group("place"))
        near = self._lookup_optional_lo(self.NEAR_PLACES, match.group("near"))

        if not subject or not food:
            return None

        output = f"{subject} {self._want_verb(subject)} to eat {food}"
        if match.group("place"):
            if not place:
                return None
            output += f" at {place}"
        if match.group("near"):
            if not near:
                return None
            output += f" near {near}"

        return CompositionalResult(output)

    def _translate_lo_buy_to_en(self, text: str) -> Optional[CompositionalResult]:
        pattern = r"^(?P<subject>.+?) ago belei (?P<item>.+?)(?: di (?P<place>.+?))?$"
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_lo(self.SUBJECTS, match.group("subject"))
        item = self._lookup_lo(self.ITEMS, match.group("item"))
        place = self._lookup_optional_lo(self.PLACES, match.group("place"))

        if not subject or not item:
            return None

        output = f"{subject} {self._want_verb(subject)} to buy {item}"
        if match.group("place"):
            if not place:
                return None
            output += f" at {place}"

        return CompositionalResult(output)

    def _translate_lo_go_to_en(self, text: str) -> Optional[CompositionalResult]:
        pattern = r"^(?P<subject>.+?) ago dak (?P<place>.+?)(?: (?P<time>tano|na'en|dawah ijo|jemeh))?$"
        match = re.fullmatch(pattern, text)
        if not match:
            return None

        subject = self._lookup_lo(self.SUBJECTS, match.group("subject"))
        place = self._lookup_lo(self.PLACES, match.group("place"))
        time = self._lookup_optional_lo(self.TIMES, match.group("time"))

        if not subject or not place:
            return None

        output = f"{subject} {self._want_verb(subject)} to go to {place}"
        if match.group("time"):
            if not time:
                return None
            output += f" {time}"

        return CompositionalResult(output)

    def _lookup_id(self, mapping: dict[str, tuple[str, str]], key: str) -> Optional[str]:
        item = mapping.get(key)
        return item[0] if item else None

    def _lookup_lo(self, mapping: dict[str, tuple[str, str]], key: str) -> Optional[str]:
        for lo, en in mapping.values():
            if lo.lower() == key:
                return en
        return None

    def _lookup_optional_id(self, mapping: dict[str, tuple[str, str]], key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        return self._lookup_id(mapping, key)

    def _lookup_optional_lo(self, mapping: dict[str, tuple[str, str]], key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        return self._lookup_lo(mapping, key)

    def _want_verb(self, subject: str) -> str:
        return "wants" if subject == "he/she" else "want"

    def _norm(self, text: str) -> str:
        return " ".join(text.strip().lower().strip(".!?").split())
