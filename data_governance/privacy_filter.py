from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field


@dataclass
class SensitiveFinding:
    kind: str
    start: int
    end: int
    severity: str
    replacement: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PrivacyScanResult:
    original_chars: int
    redacted_text: str
    findings: list[SensitiveFinding] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def has_critical(self) -> bool:
        return any(item.severity == "critical" for item in self.findings)

    @property
    def has_high(self) -> bool:
        return any(item.severity in {"critical", "high"} for item in self.findings)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        severities: dict[str, int] = {}
        for item in self.findings:
            counts[item.kind] = counts.get(item.kind, 0) + 1
            severities[item.severity] = severities.get(item.severity, 0) + 1
        return {
            "original_chars": self.original_chars,
            "redacted_chars": len(self.redacted_text),
            "finding_count": len(self.findings),
            "kinds": counts,
            "severities": severities,
            "has_critical": self.has_critical,
            "has_high": self.has_high,
        }


@dataclass(frozen=True)
class SensitivePattern:
    kind: str
    regex: re.Pattern
    severity: str
    reason: str
    group: int = 0


class PrivacyFilter:
    """Detect and redact credentials, secrets, and common personal data.

    This is a conservative pre-training gate, not a legal compliance guarantee.
    Human review is still required before user/app/web data is used for training.
    """

    def __init__(self) -> None:
        self.patterns = [
            SensitivePattern(
                "private_key",
                re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
                "critical",
                "Private keys must never enter training data.",
            ),
            SensitivePattern(
                "credential_assignment",
                re.compile(
                    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|authorization)\b\s*[:=]\s*[\"']?([^\s\"',;]{6,})",
                ),
                "critical",
                "Credential-like key/value pair.",
                group=2,
            ),
            SensitivePattern(
                "bearer_token",
                re.compile(r"(?i)\bBearer\s+([A-Za-z0-9._~+/=-]{16,})"),
                "critical",
                "Bearer token.",
                group=1,
            ),
            SensitivePattern(
                "jwt",
                re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
                "critical",
                "JWT-like token.",
            ),
            SensitivePattern(
                "aws_access_key",
                re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"),
                "critical",
                "AWS access key.",
            ),
            SensitivePattern(
                "database_uri",
                re.compile(r"(?i)\b(?:postgresql|postgres|mysql|mongodb|redis)://[^\s]+:[^\s@]+@[^\s]+"),
                "critical",
                "Database URI with embedded credentials.",
            ),
            SensitivePattern(
                "whatsapp_jid",
                re.compile(r"(?<!\d)(?:\+?62|0)8[1-9]\d{7,13}@(?:s\.whatsapp\.net|c\.us)\b"),
                "high",
                "WhatsApp JID can identify a person or customer.",
            ),
            SensitivePattern(
                "url_basic_auth",
                re.compile(r"(?i)\bhttps?://[^/\s:@]+:[^/\s:@]+@[^/\s]+"),
                "critical",
                "URL contains username and password.",
            ),
            SensitivePattern(
                "email",
                re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
                "medium",
                "Email address.",
            ),
            SensitivePattern(
                "indonesian_phone",
                re.compile(r"(?<!\d)(?:\+62|62|0)8[1-9][0-9][\d\s.-]{6,13}(?!\d)"),
                "medium",
                "Indonesian mobile phone number.",
            ),
            SensitivePattern(
                "indonesian_nik_or_16_digit_id",
                re.compile(r"(?<!\d)\d{16}(?!\d)"),
                "high",
                "Possible Indonesian NIK or 16-digit personal identifier.",
            ),
            SensitivePattern(
                "npwp_like",
                re.compile(r"(?<!\d)\d{2}[.\-]?\d{3}[.\-]?\d{3}[.\-]?\d[.\-]?\d{3}[.\-]?\d{3}(?!\d)"),
                "high",
                "Possible Indonesian tax ID.",
            ),
            SensitivePattern(
                "bank_account_context",
                re.compile(
                    r"(?i)\b(?:no\.?\s*rek|rekening|bank\s*account|account\s*number|nomor\s*rekening)\b\s*[:#-]?\s*(\d[\d\s.-]{7,24}\d)",
                ),
                "critical",
                "Bank account number in financial context.",
                group=1,
            ),
            SensitivePattern(
                "address_context",
                re.compile(
                    r"(?i)\b(?:alamat|address|domisili)\b\s*[:=-]\s*([^,\n]{8,120}(?:[,][^\n]{0,160})?)",
                ),
                "high",
                "Address-like personal location data.",
                group=1,
            ),
            SensitivePattern(
                "financial_amount_context",
                re.compile(
                    r"(?i)\b(?:gaji|salary|income|pendapatan|utang|hutang|debt|cicilan|tagihan|saldo|balance|tabungan)\b[^.\n]{0,80}\b(?:rp\.?\s*)?\d[\d.,]{3,}",
                ),
                "medium",
                "Financial amount in household or personal finance context.",
            ),
        ]

    def scan_text(self, text: str) -> PrivacyScanResult:
        findings: list[SensitiveFinding] = []
        for pattern in self.patterns:
            for match in pattern.regex.finditer(text):
                start, end = match.span(pattern.group)
                findings.append(
                    SensitiveFinding(
                        kind=pattern.kind,
                        start=start,
                        end=end,
                        severity=pattern.severity,
                        replacement=f"<redacted:{pattern.kind}>",
                        reason=pattern.reason,
                    )
                )

        findings.extend(self._find_credit_cards(text))
        findings = self._dedupe_findings(findings)
        redacted = self._redact(text, findings)
        return PrivacyScanResult(
            original_chars=len(text),
            redacted_text=redacted,
            findings=findings,
        )

    def scan_payload(self, payload: dict[str, str]) -> tuple[dict[str, str], dict[str, dict]]:
        sanitized: dict[str, str] = {}
        reports: dict[str, dict] = {}
        for key, value in payload.items():
            if not isinstance(value, str):
                continue
            result = self.scan_text(value)
            sanitized[key] = result.redacted_text
            reports[key] = result.summary() | {
                "findings": [item.to_dict() for item in result.findings],
            }
        return sanitized, reports

    def _find_credit_cards(self, text: str) -> list[SensitiveFinding]:
        findings: list[SensitiveFinding] = []
        for match in re.finditer(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)", text):
            digits = re.sub(r"\D", "", match.group(0))
            if 13 <= len(digits) <= 19 and _luhn_valid(digits):
                findings.append(
                    SensitiveFinding(
                        kind="payment_card",
                        start=match.start(),
                        end=match.end(),
                        severity="critical",
                        replacement="<redacted:payment_card>",
                        reason="Payment card number.",
                    )
                )
        return findings

    def _dedupe_findings(self, findings: list[SensitiveFinding]) -> list[SensitiveFinding]:
        priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        findings = sorted(findings, key=lambda x: (x.start, -(x.end - x.start), -priority.get(x.severity, 0)))
        kept: list[SensitiveFinding] = []
        for item in findings:
            overlaps = any(not (item.end <= other.start or item.start >= other.end) for other in kept)
            if not overlaps:
                kept.append(item)
        return sorted(kept, key=lambda x: x.start)

    def _redact(self, text: str, findings: list[SensitiveFinding]) -> str:
        if not findings:
            return text
        parts: list[str] = []
        cursor = 0
        for item in findings:
            parts.append(text[cursor:item.start])
            parts.append(item.replacement)
            cursor = item.end
        parts.append(text[cursor:])
        return "".join(parts)


def _luhn_valid(digits: str) -> bool:
    total = 0
    reverse = digits[::-1]
    for index, char in enumerate(reverse):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
