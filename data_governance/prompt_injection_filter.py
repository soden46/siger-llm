from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class InjectionFinding:
    kind: str
    severity: str
    reason: str
    pattern: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InjectionScanResult:
    allowed: bool
    action: str
    findings: list[InjectionFinding] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "findings": [item.to_dict() for item in self.findings],
        }


class PromptInjectionFilter:
    """Detect common prompt-injection and system-exfiltration attempts."""

    def __init__(self, block_high: bool = True):
        self.block_high = block_high
        self.patterns: list[tuple[str, str, str, re.Pattern]] = [
            (
                "system_prompt_exfiltration",
                "high",
                "Attempts to reveal hidden system/developer instructions.",
                re.compile(
                    r"(?i)\b(reveal|show|print|dump|leak|expose|tampilkan|bocorkan)\b.{0,80}\b(system prompt|developer message|hidden instruction|instruksi sistem|prompt rahasia)\b"
                ),
            ),
            (
                "instruction_override",
                "high",
                "Attempts to override safety or system instructions.",
                re.compile(
                    r"(?i)\b(ignore|abaikan|lupakan|bypass|override|disregard)\b.{0,80}\b(previous instructions|system instructions|instruksi sebelumnya|aturan|safety|guardrail)\b"
                ),
            ),
            (
                "jailbreak_roleplay",
                "medium",
                "Jailbreak-style roleplay marker.",
                re.compile(r"(?i)\b(DAN mode|developer mode|jailbreak|tanpa batasan|no restrictions|unfiltered)\b"),
            ),
            (
                "tool_or_memory_exfiltration",
                "high",
                "Attempts to extract tool output, memory, or private context.",
                re.compile(
                    r"(?i)\b(read|dump|show|print|extract|ambil|tampilkan)\b.{0,80}\b(memory|tool result|retrieved context|private context|chat history|riwayat chat)\b"
                ),
            ),
            (
                "prompt_wrapped_instruction",
                "medium",
                "Likely injected instruction inside external content.",
                re.compile(
                    r"(?i)\b(new instruction|system override|assistant must|model must|instruksi baru|abaikan semua)\b"
                ),
            ),
        ]

    def scan_text(self, text: str, *, context: str = "chat") -> InjectionScanResult:
        findings: list[InjectionFinding] = []
        for kind, severity, reason, pattern in self.patterns:
            if pattern.search(text):
                findings.append(
                    InjectionFinding(
                        kind=kind,
                        severity=severity,
                        reason=reason,
                        pattern=pattern.pattern,
                    )
                )

        has_high = any(item.severity == "high" for item in findings)
        if self.block_high and has_high and context == "chat":
            return InjectionScanResult(False, "block", findings)
        if findings:
            return InjectionScanResult(True, "mark_untrusted", findings)
        return InjectionScanResult(True, "allow", findings)

    def safe_response(self) -> str:
        return (
            "Maaf, aku tidak bisa mengikuti instruksi yang mencoba mengubah aturan sistem, "
            "membocorkan prompt internal, atau mengambil memory pribadi. Aku tetap bisa bantu "
            "dengan pertanyaan normal atau analisis keamanan pada konteks yang memang kamu berikan."
        )
