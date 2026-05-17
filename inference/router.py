from __future__ import annotations

import re
from dataclasses import dataclass

from inference.chat import ChatSession
from inference.lampung_pipeline import LampungPipeline, LampungResponse


@dataclass(frozen=True)
class RoutedResponse:
    text: str
    route: str
    source: str


class SigerRouter:
    """Route between general chat and Lampung domain tools."""

    LAMPUNG_HINTS = {
        "nyak",
        "nikeu",
        "sekam",
        "mettei",
        "gham",
        "ikam",
        "tiyan",
        "punyeu",
        "mengan",
        "haga",
        "agow",
        "ago",
        "dak",
        "dikedow",
        "paghek",
        "nuwo",
        "wawai",
        "tabik",
    }

    def __init__(self, chat: ChatSession, lampung: LampungPipeline):
        self.chat = chat
        self.lampung = lampung

    def route(self, text: str, max_new_tokens: int = 80) -> RoutedResponse:
        intent = self.detect_intent(text)

        if intent == "lampung_to_id":
            response = self.lampung.translate("Lampung O", "Bahasa Indonesia", text)
            return self._from_lampung(response, "lampung_to_id")

        if intent == "id_to_lampung":
            cleaned = self._strip_translate_request(text)
            response = self.lampung.translate("Bahasa Indonesia", "Lampung O", cleaned)
            return self._from_lampung(response, "id_to_lampung")

        if intent == "lampung_to_en":
            response = self.lampung.translate("Lampung O", "English", text)
            return self._from_lampung(response, "lampung_to_en")

        reply = self.chat.chat(
            text,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
        )
        return RoutedResponse(reply.strip(), "general_chat", "model generation")

    def detect_intent(self, text: str) -> str:
        normalized = self._norm(text)

        if re.search(r"\b(translate|terjemah(?:kan)?)\b", normalized):
            if "english" in normalized or "inggris" in normalized:
                return "lampung_to_en" if self.looks_lampung(normalized) else "general_chat"
            if "lampung" in normalized:
                return "id_to_lampung"
            if self.looks_lampung(normalized):
                return "lampung_to_id"

        if self.looks_lampung(normalized):
            return "lampung_to_id"

        return "general_chat"

    def looks_lampung(self, text: str) -> bool:
        words = set(re.findall(r"[a-z']+", text.lower()))
        return len(words & self.LAMPUNG_HINTS) >= 2

    def _strip_translate_request(self, text: str) -> str:
        cleaned = re.sub(
            r"(?i)\b(tolong\s+)?(terjemah(?:kan)?|translate)\b.*?\b(lampung|bahasa lampung)\b[:\-]?",
            "",
            text,
        ).strip()
        return cleaned or text

    def _norm(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _from_lampung(self, response: LampungResponse, route: str) -> RoutedResponse:
        return RoutedResponse(response.text, route, response.source)
