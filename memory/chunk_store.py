from dataclasses import dataclass, field
from collections import Counter
import re
from typing import List, Dict


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class MemoryChunk:
    text: str
    metadata: Dict = field(default_factory=dict)


class ChunkStore:
    def __init__(self):
        self.chunks: List[MemoryChunk] = []

    def add(self, text: str, metadata: Dict | None = None):
        text = text.strip()
        if not text:
            return
        self.chunks.append(MemoryChunk(text=text, metadata=metadata or {}))

    def add_long_text(
        self,
        text: str,
        chunk_size_words: int = 180,
        overlap_words: int = 40,
        metadata: Dict | None = None,
    ):
        words = tokenize_words(text)
        if not words:
            return

        step = max(1, chunk_size_words - overlap_words)

        for start in range(0, len(words), step):
            chunk_words = words[start : start + chunk_size_words]
            if not chunk_words:
                continue

            self.add(
                " ".join(chunk_words),
                metadata={
                    **(metadata or {}),
                    "start_word": start,
                    "end_word": start + len(chunk_words),
                },
            )

    def search(self, query: str, top_k: int = 5) -> List[MemoryChunk]:
        q_terms = tokenize_words(query)
        if not q_terms:
            return []

        q_counts = Counter(q_terms)
        scored = []

        for chunk in self.chunks:
            c_terms = tokenize_words(chunk.text)
            c_counts = Counter(c_terms)

            score = 0.0
            for term, q_count in q_counts.items():
                score += min(q_count, c_counts.get(term, 0))

            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]