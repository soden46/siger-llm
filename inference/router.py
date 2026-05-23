from __future__ import annotations

import re
from dataclasses import dataclass

from inference.chat import ChatSession
from inference.constitutional_ai_layer import ConstitutionalAILayer
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

    def __init__(
        self,
        chat: ChatSession,
        lampung: LampungPipeline,
        safety_layer: ConstitutionalAILayer | None = None,
    ):
        self.chat = chat
        self.lampung = lampung
        self.safety_layer = safety_layer or ConstitutionalAILayer()

    def route(self, text: str, max_new_tokens: int = 80) -> RoutedResponse:
        safety = self.safety_layer.guard_prompt(text)
        if not safety.allowed:
            return RoutedResponse(safety.response, f"refusal_{safety.category}", "constitutional guardrail")

        intent = self.detect_intent(text)

        if intent == "lampung_reason":
            lampung_text = self._extract_lampung_text(text)
            response = self.lampung.reason_lo_to_id(lampung_text, max_new_tokens=max_new_tokens)
            return self._from_lampung(response, "lampung_reason")

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

        if self._is_lampung_reason_request(normalized):
            return "lampung_reason"

        if self.looks_lampung(normalized):
            return "lampung_to_id"

        return "general_chat"

    def detect_language(self, text: str) -> str:
        normalized = self._norm(text)
        if self.looks_lampung(normalized):
            return "lampung_o"

        english_markers = {
            "the",
            "and",
            "please",
            "translate",
            "explain",
            "what",
            "how",
            "why",
        }
        indonesian_markers = {
            "dan",
            "yang",
            "tolong",
            "terjemahkan",
            "jelaskan",
            "apa",
            "bagaimana",
            "kenapa",
        }
        words = set(re.findall(r"[a-z']+", normalized))
        en_score = len(words & english_markers)
        id_score = len(words & indonesian_markers)
        if en_score > id_score:
            return "en"
        return "id"

    def detect_domain(self, text: str) -> str:
        normalized = self._norm(text)
        if self.safety_layer.guard_prompt(text).allowed is False:
            return "safety"
        if re.search(r"\b(laravel|artisan|migration|eloquent|blade|controller)\b", normalized):
            return "laravel"
        if re.search(r"\b(debug|error|traceback|exception|bug|fix)\b", normalized):
            return "debug"
        if re.search(r"\b(python|javascript|php|sql|function|class|api|kode|code)\b", normalized):
            return "code"
        if re.search(r"\b(hitung|matematika|integral|turunan|aljabar|probability|equation)\b", normalized):
            return "math"
        if re.search(r"\b(translate|terjemah(?:kan)?)\b", normalized) or self.looks_lampung(normalized):
            return "translation"
        return "general"

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

    def _is_lampung_reason_request(self, text: str) -> bool:
        if not self.looks_lampung(text):
            return False

        reason_markers = (
            "jelaskan",
            "penjelasan",
            "struktur",
            "susunan",
            "pola kalimat",
            "kata per kata",
            "arti kata",
            "makna",
            "grammar",
        )
        return "lampung" in text and any(marker in text for marker in reason_markers)

    def _extract_lampung_text(self, text: str) -> str:
        for separator in (":", "\n"):
            if separator in text:
                candidate = text.rsplit(separator, 1)[-1].strip()
                if self.looks_lampung(self._norm(candidate)):
                    return candidate
        return text

    def _norm(self, text: str) -> str:
        return " ".join(text.strip().lower().split())

    def _from_lampung(self, response: LampungResponse, route: str) -> RoutedResponse:
        return RoutedResponse(response.text, route, response.source)
