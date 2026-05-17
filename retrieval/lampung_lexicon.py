import json
import re
from pathlib import Path


class LampungLexicon:
    """Small JSONL-backed lexicon for retrieval-augmented Lampung prompts."""

    WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")

    def __init__(self, jsonl_path: str):
        self.entries = self._load(jsonl_path)

    def _load(self, path: str) -> dict[str, str]:
        file_path = Path(path)
        entries: dict[str, str] = {}

        if not file_path.exists():
            return entries

        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                lampung = str(row.get("lampung") or row.get("input") or "").strip()
                indo = str(row.get("indonesian") or row.get("output") or "").strip()
                instruction = str(row.get("instruction") or "").lower()

                if "indonesia ke lampung" in instruction:
                    continue

                if lampung and indo and len(lampung.split()) <= 4:
                    entries.setdefault(lampung.lower().strip(".!?"), indo)

        return entries

    def lookup_words(self, text: str) -> list[tuple[str, str]]:
        normalized = text.lower().strip(".!?")
        matches: list[tuple[str, str]] = []

        if normalized in self.entries:
            matches.append((normalized, self.entries[normalized]))

        for word in self.WORD_RE.findall(normalized):
            if word in self.entries:
                pair = (word, self.entries[word])
                if pair not in matches:
                    matches.append(pair)

        return matches

    def build_context(self, text: str, max_entries: int = 12) -> str:
        pairs = self.lookup_words(text)[:max_entries]
        return "\n".join(f"- {src} = {dst}" for src, dst in pairs)

