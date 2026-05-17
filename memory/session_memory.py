from dataclasses import dataclass, field
from typing import List, Dict

from memory.chunk_store import ChunkStore


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
        self.max_recent_turns = max_recent_turns
        self.summary_max_chars = summary_max_chars
        self.pinned_facts: List[str] = []

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

    def add_document(self, text: str, metadata: Dict | None = None):
        self.chunk_store.add_long_text(
            text,
            metadata={"type": "document", **(metadata or {})},
        )

    def add_pinned_fact(self, fact: str):
        fact = fact.strip()
        if fact and fact not in self.pinned_facts:
            self.pinned_facts.append(fact)

    def recent_turns(self) -> List[Turn]:
        return self.turns[-self.max_recent_turns :]

    def retrieve(self, query: str, top_k: int = 5):
        return self.chunk_store.search(query, top_k=top_k)

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

    def clear(self):
        self.turns.clear()
        self.chunk_store = ChunkStore()
        self.summary = ""
        self.pinned_facts.clear()