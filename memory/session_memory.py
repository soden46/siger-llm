from dataclasses import dataclass
from typing import List, Dict

from memory.chunk_store import ChunkStore, extract_leading_summary


@dataclass
class Turn:
    role: str
    content: str


class SessionMemory:
    def __init__(
        self,
        max_recent_turns: int = 8,
        summary_max_chars: int = 2000,
    ):
        self.turns: List[Turn] = []
        self.chunk_store = ChunkStore()
        self.summary = ""
        self.long_context_summary = ""
        self.max_recent_turns = max_recent_turns
        self.summary_max_chars = summary_max_chars
        self.pinned_facts: List[str] = []
        self.document_count = 0
        self.long_context_chars = 0

    def add_turn(self, role: str, content: str):
        content = content.strip()
        if not content:
            return

        turn = Turn(role=role, content=content)
        self.turns.append(turn)

        self.chunk_store.add(
            f"{role}: {content}",
            metadata={"type": "turn", "role": role},
        )

        self._update_summary()

    def add_document(
        self,
        text: str,
        metadata: Dict | None = None,
        *,
        chunk_size_words: int = 180,
        overlap_words: int = 40,
    ):
        text = text.strip()
        if not text:
            return
        self.document_count += 1
        self.long_context_chars += len(text)
        self.chunk_store.add_long_text(
            text,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            metadata={
                "type": "document",
                "document_index": self.document_count,
                **(metadata or {}),
            },
        )
        self._update_long_context_summary(text, metadata or {})

    def add_pinned_fact(self, fact: str):
        fact = fact.strip()
        if fact and fact not in self.pinned_facts:
            self.pinned_facts.append(fact)

    def recent_turns(self) -> List[Turn]:
        return self.turns[-self.max_recent_turns :]

    def retrieve(self, query: str, top_k: int = 5):
        return self.chunk_store.search(query, top_k=top_k)

    def ingest_long_user_message(
        self,
        text: str,
        *,
        max_inline_chars: int = 1200,
        chunk_size_words: int = 180,
        overlap_words: int = 40,
    ) -> str:
        """Store huge user input as retrievable context and return a compact query."""
        text = text.strip()
        if len(text) <= max_inline_chars:
            return text

        summary = extract_leading_summary(text, max_chars=700)
        self.add_document(
            text,
            metadata={"source": "long_user_message"},
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )
        return (
            "User memberi konteks panjang yang sudah disimpan dalam memory retrieval. "
            f"Ringkasan awal konteks: {summary}\n\n"
            "Gunakan konteks relevan dari memory untuk menjawab permintaan user."
        )

    def _update_summary(self):
        if len(self.turns) <= self.max_recent_turns:
            return

        older = self.turns[: -self.max_recent_turns]
        summary_lines = []

        for turn in older[-20:]:
            text = turn.content.replace("\n", " ").strip()
            summary_lines.append(f"{turn.role}: {text[:180]}")

        summary = "\n".join(summary_lines)
        self.summary = summary[-self.summary_max_chars:]

    def _update_long_context_summary(self, text: str, metadata: Dict) -> None:
        label = str(metadata.get("title") or metadata.get("source") or f"document_{self.document_count}")
        summary = extract_leading_summary(text, max_chars=500)
        entry = f"[{label}] {summary}"
        combined = f"{self.long_context_summary}\n{entry}".strip()
        self.long_context_summary = combined[-self.summary_max_chars:]

    def clear(self):
        self.turns.clear()
        self.chunk_store = ChunkStore()
        self.summary = ""
        self.long_context_summary = ""
        self.pinned_facts.clear()
        self.document_count = 0
        self.long_context_chars = 0
