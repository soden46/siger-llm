"""Lightweight critical/safety layer for SigerLM inference.

The goal is not to replace model alignment. This layer gives the router a cheap
first line of defense: refuse clearly illegal, harmful, privacy-violating, or
unrealistic requests with a calm explanation and a useful alternative.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstitutionalPrinciple:
    name: str
    description: str
    severity: str
    refusal_reason: str
    safe_alternative: str


@dataclass(frozen=True)
class GuardrailDecision:
    allowed: bool
    category: str
    reason: str
    response: str = ""
    confidence: float = 0.0


SIGER_PRINCIPLES: dict[str, ConstitutionalPrinciple] = {
    "no_illegal": ConstitutionalPrinciple(
        name="no_illegal",
        description="Tolak bantuan untuk aktivitas ilegal atau pengelakan hukum.",
        severity="hard_block",
        refusal_reason="permintaan itu dapat melanggar hukum atau membantu pelanggaran",
        safe_alternative=(
            "Saya bisa bantu jelaskan jalur legal, mitigasi risiko, atau langkah "
            "administratif yang aman."
        ),
    ),
    "no_harm": ConstitutionalPrinciple(
        name="no_harm",
        description="Tolak instruksi yang dapat melukai orang, diri sendiri, atau sistem.",
        severity="hard_block",
        refusal_reason="permintaan itu bisa membahayakan orang lain, diri sendiri, atau sistem",
        safe_alternative=(
            "Saya bisa bantu cari pendekatan yang aman, pencegahan, edukasi, atau "
            "rencana pemulihan."
        ),
    ),
    "respect_privacy": ConstitutionalPrinciple(
        name="respect_privacy",
        description="Tolak akses data pribadi tanpa izin.",
        severity="hard_block",
        refusal_reason="itu menyangkut privasi dan akses tanpa izin",
        safe_alternative=(
            "Saya bisa bantu dengan praktik keamanan akun, pemulihan akses milik sendiri, "
            "atau cara meminta data secara sah."
        ),
    ),
    "no_discrimination": ConstitutionalPrinciple(
        name="no_discrimination",
        description="Tolak ujaran kebencian dan generalisasi merendahkan kelompok.",
        severity="hard_block",
        refusal_reason="permintaan itu dapat merendahkan atau menyasar kelompok tertentu",
        safe_alternative=(
            "Saya bisa bantu membahas topik sosial secara faktual, adil, dan menghormati "
            "martabat semua pihak."
        ),
    ),
    "critical_realism": ConstitutionalPrinciple(
        name="critical_realism",
        description="Kritis terhadap permintaan yang tidak realistis atau klaim mustahil.",
        severity="soft_block",
        refusal_reason="permintaan itu tidak realistis atau tidak bisa dijamin dengan jujur",
        safe_alternative=(
            "Saya bisa bantu ubah jadi target yang masuk akal, rencana bertahap, atau "
            "analisis risiko dan batasannya."
        ),
    ),
}


class ConstitutionalAILayer:
    """Keyword and pattern based guardrail with gentle refusal templates."""

    def __init__(
        self,
        principles: Optional[dict[str, ConstitutionalPrinciple]] = None,
        enabled: bool = True,
    ):
        self.principles = principles or SIGER_PRINCIPLES
        self.enabled = enabled

    def guard_prompt(self, prompt: str) -> GuardrailDecision:
        """Decide whether a user prompt should be answered by the model."""
        if not self.enabled:
            return GuardrailDecision(True, "allowed", "guard disabled")

        normalized = self._normalize(prompt)
        category = self._classify_prompt(normalized)
        if category is None:
            return GuardrailDecision(True, "allowed", "no blocking pattern detected", confidence=0.2)

        principle = self.principles[category]
        response = self._soft_refusal(prompt, principle)
        confidence = 0.9 if principle.severity == "hard_block" else 0.75
        logger.info("Guardrail blocked prompt: category=%s confidence=%.2f", category, confidence)
        return GuardrailDecision(False, category, principle.refusal_reason, response, confidence)

    def evaluate_response(self, prompt: str, response: str) -> tuple[bool, float, list[str]]:
        """Backward-compatible response checker."""
        text = self._normalize(f"{prompt} {response}")
        violations = []
        for category in self._detect_categories(text):
            if category in self.principles:
                violations.append(category)

        score = 10.0
        for category in violations:
            severity = self.principles[category].severity
            score -= 5.0 if severity == "hard_block" else 2.0
        return not violations, max(0.0, score), violations

    def revise_response(self, prompt: str, response: str, violated_principles: list[str]) -> str:
        if not violated_principles:
            return response
        principle = self.principles.get(violated_principles[0])
        if principle is None:
            return response
        return self._soft_refusal(prompt, principle)

    def self_critique_batch(self, responses: list[dict]) -> list[dict]:
        results = []
        for item in responses:
            prompt = str(item.get("prompt", ""))
            response = str(item.get("response", ""))
            is_safe, score, violated = self.evaluate_response(prompt, response)
            results.append(
                {
                    "prompt": prompt,
                    "original_response": response,
                    "is_safe": is_safe,
                    "safety_score": score,
                    "violated_principles": violated,
                    "revised_response": response if is_safe else self.revise_response(prompt, response, violated),
                }
            )
        return results

    def _classify_prompt(self, normalized_prompt: str) -> Optional[str]:
        categories = self._detect_categories(normalized_prompt)

        # Hard blocks win over "unrealistic" because they carry legal/safety risk.
        for category in ("no_illegal", "no_harm", "respect_privacy", "no_discrimination"):
            if category in categories:
                return category
        if "critical_realism" in categories:
            return "critical_realism"
        return None

    def _detect_categories(self, text: str) -> list[str]:
        categories: list[str] = []

        illegal_patterns = (
            r"\b(cara|tutorial|bantu|tolong|ajarin|bagaimana|help|teach)\b.*\b(hack|retas|membobol|bobol|crack password|password cracker)\b",
            r"\b(bypass|menghindari)\b.*\b(pajak|hukum|polisi|deteksi|verifikasi)\b",
            r"\b(dokumen palsu|ktp palsu|ijazah palsu|uang palsu|money laundering)\b",
            r"\b(cara|tutorial|buat|bikin)\b.*\b(malware|ransomware|phishing|keylogger)\b",
        )
        harm_patterns = (
            r"\b(cara|tutorial|buat|bikin|racik)\b.*\b(racun|bom|senjata|melukai|membunuh)\b",
            r"\b(self-harm|suicide|bunuh diri|overdose)\b",
            r"\b(cara|tips)\b.*\b(membully|memeras|mengancam|doxxing)\b",
        )
        privacy_patterns = (
            r"\b(cara|bantu|tolong)\b.*\b(lihat|baca|ambil|curi|akses)\b.*\b(chat|email|password|akun|data pribadi)\b",
            r"\b(chat pribadi|email orang|akun orang|data pribadi orang)\b",
            r"\b(help|teach|show)\b.*\b(read|access|steal|get)\b.*\b(private chat|private email|password|someone else's account)\b",
        )
        discrimination_patterns = (
            r"\b(benci|hapuskan|serang|hina)\b.*\b(suku|agama|ras|gender|kelompok)\b",
            r"\b(racist|sexist|homophobic|ujaran kebencian)\b",
        )
        unrealistic_patterns = (
            r"\b(jamin|pastikan|garansi)\b.*\b(100%|pasti|selalu)\b",
            r"\b(buat|bikin|jadikan)\b.*\b(kaya mendadak|viral pasti|menang judi|mesin waktu)\b",
            r"\b(hapus|hilangkan)\b.*\b(hutang|jejak digital|catatan kriminal)\b.*\b(seketika|tanpa proses)\b",
            r"\b(prediksi|tebak)\b.*\b(angka togel|lotre|judi)\b.*\b(pasti|akurat)\b",
        )

        pattern_groups = (
            ("no_illegal", illegal_patterns),
            ("no_harm", harm_patterns),
            ("respect_privacy", privacy_patterns),
            ("no_discrimination", discrimination_patterns),
            ("critical_realism", unrealistic_patterns),
        )
        for category, patterns in pattern_groups:
            if any(re.search(pattern, text) for pattern in patterns):
                categories.append(category)

        return categories

    def _soft_refusal(self, prompt: str, principle: ConstitutionalPrinciple) -> str:
        if principle.name == "critical_realism":
            return (
                "Aku perlu jujur: permintaan itu tidak bisa kujanjikan secara realistis. "
                f"Alasannya, {principle.refusal_reason}. {principle.safe_alternative}"
            )

        return (
            "Maaf, aku tidak bisa membantu langsung dengan permintaan itu. "
            f"Alasannya, {principle.refusal_reason}. {principle.safe_alternative}"
        )

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.lower().strip().split())


def main() -> None:
    layer = ConstitutionalAILayer()
    examples = [
        "Cara hack email orang?",
        "Bikin aku kaya mendadak 100% pasti berhasil",
        "Terjemahkan: Nyak haga mengan manuk",
    ]
    for example in examples:
        decision = layer.guard_prompt(example)
        print(f"{example!r} -> allowed={decision.allowed}, category={decision.category}")
        if not decision.allowed:
            print(decision.response)


if __name__ == "__main__":
    main()
