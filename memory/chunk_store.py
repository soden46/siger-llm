from dataclasses import dataclass, field
from collections import Counter
import re
from typing import List, Dict


_WORD_RE = re.compile(r"\w+", re.UNICODE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}", re.UNICODE)


def tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class MemoryChunk:
    text: str
    metadata: Dict = field(default_factory=dict)
    terms: list[str] = field(default_factory=list)
    term_counts: Counter = field(default_factory=Counter)


class ChunkStore:
    def __init__(self):
        self.chunks: List[MemoryChunk] = []

    def add(self, text: str, metadata: Dict | None = None):
        text = text.strip()
        if not text:
            return
        terms = tokenize_words(text)
        self.chunks.append(
            MemoryChunk(
                text=text,
                metadata=metadata or {},
                terms=terms,
                term_counts=Counter(terms),
            )
        )

    def add_long_text(
        self,
        text: str,
        chunk_size_words: int = 180,
        overlap_words: int = 40,
        metadata: Dict | None = None,
    ):
        matches = list(_WORD_RE.finditer(text))
        if not matches:
            return

        step = max(1, chunk_size_words - overlap_words)

        for start in range(0, len(matches), step):
            chunk_matches = matches[start : start + chunk_size_words]
            if not chunk_matches:
                continue

            chunk_text = text[chunk_matches[0].start() : chunk_matches[-1].end()].strip()
            self.add(
                chunk_text,
                metadata={
                    **(metadata or {}),
                    "start_word": start,
                    "end_word": start + len(chunk_matches),
                },
            )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        source_types: set[str] | None = None,
    ) -> List[MemoryChunk]:
        q_terms = tokenize_words(query)
        if not q_terms:
            return []

        q_counts = Counter(q_terms)
        scored = []

        for index, chunk in enumerate(self.chunks):
            if source_types and str(chunk.metadata.get("type")) not in source_types:
                continue

            score = 0.0
            for term, q_count in q_counts.items():
                score += min(q_count, chunk.term_counts.get(term, 0))

            if score <= 0:
                continue

            coverage = sum(1 for term in q_counts if chunk.term_counts.get(term, 0)) / max(1, len(q_counts))
            length_norm = max(1.0, len(chunk.terms) ** 0.35)
            recency = 1.0 + (index / max(1, len(self.chunks) - 1)) * 0.05
            final_score = (score + coverage * 2.0) / length_norm * recency
            scored.append((final_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]


def extract_leading_summary(text: str, max_chars: int = 700) -> str:
    """Cheap extractive summary for huge inputs before model summarization exists."""
    cleaned = " ".join(text.strip().split())
    if len(cleaned) <= max_chars:
        return cleaned

    sentences = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    summary = ""
    for sentence in sentences:
        candidate = f"{summary} {sentence}".strip()
        if len(candidate) > max_chars:
            break
        summary = candidate

    if summary:
        return summary
    return cleaned[:max_chars].rstrip() + "..."
