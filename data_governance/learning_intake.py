from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .domain_policy import DomainPolicyEngine
from .privacy_filter import PrivacyFilter


@dataclass
class LearningIntakeConfig:
    base_dir: str = "data/intake"
    candidates_file: str = "candidates_sanitized.jsonl"
    quarantine_file: str = "quarantine_redacted.jsonl"
    approved_file: str = "approved_training.jsonl"
    approval_log_file: str = "approval_log.jsonl"
    reject_without_consent: bool = True
    require_human_approval: bool = True


@dataclass
class LearningIntakeRecord:
    source_type: str
    text: str = ""
    instruction: str = ""
    input: str = ""
    output: str = ""
    source_url: str = ""
    app_id: str = ""
    session_id: str = ""
    user_id: str = ""
    language: str = ""
    domain: str = "general"
    purpose: str = "model_improvement"
    learning_mode: str = "training_candidate"
    consent: bool = False
    allow_training: bool = False
    approved_for_training: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, str]:
        return {
            "text": self.text,
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }


class LearningDataIntake:
    """Privacy-first intake for web/app data that may later become training data."""

    def __init__(
        self,
        config: LearningIntakeConfig | None = None,
        privacy_filter: PrivacyFilter | None = None,
        policy_engine: DomainPolicyEngine | None = None,
    ) -> None:
        self.config = config or LearningIntakeConfig()
        self.privacy_filter = privacy_filter or PrivacyFilter()
        self.policy_engine = policy_engine or DomainPolicyEngine()
        self.base_dir = Path(self.config.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def candidates_path(self) -> Path:
        return self.base_dir / self.config.candidates_file

    @property
    def quarantine_path(self) -> Path:
        return self.base_dir / self.config.quarantine_file

    @property
    def approved_path(self) -> Path:
        return self.base_dir / self.config.approved_file

    @property
    def approval_log_path(self) -> Path:
        return self.base_dir / self.config.approval_log_file

    def submit(self, record: LearningIntakeRecord) -> dict:
        payload = record.payload()
        sanitized_payload, privacy_reports = self.privacy_filter.scan_payload(payload)
        privacy_summary = self._merge_privacy_reports(privacy_reports)
        policy = self.policy_engine.decide(
            source_type=record.source_type,
            domain=record.domain,
            metadata=record.metadata,
            privacy=privacy_summary,
            learning_mode=record.learning_mode,
        )

        status = self._decide_status(record, privacy_summary, policy.to_dict())
        intake_id = str(uuid4())
        row = {
            "intake_id": intake_id,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "source_type": record.source_type,
            "source_url": record.source_url,
            "app_id": record.app_id,
            "session_id": record.session_id,
            "user_id_hash_hint": _short_hash_hint(record.user_id),
            "language": record.language,
            "domain": record.domain,
            "purpose": record.purpose,
            "learning_mode": record.learning_mode,
            "consent": record.consent,
            "allow_training": record.allow_training,
            "approved_for_training": False,
            "payload": sanitized_payload,
            "privacy": privacy_summary,
            "policy": policy.to_dict(),
            "field_privacy": privacy_reports,
            "metadata": _sanitize_metadata(record.metadata),
        }

        if status in {"quarantined_sensitive", "rejected_no_consent"}:
            self._append_jsonl(self.quarantine_path, row)
            bucket = "quarantine"
        else:
            self._append_jsonl(self.candidates_path, row)
            bucket = "candidates"

        if (
            record.approved_for_training
            and status == "accepted_candidate"
            and not self.config.require_human_approval
        ):
            self.approve(intake_id, reviewer="system_auto", decision="approve")

        return {
            "status": status,
            "intake_id": intake_id,
            "bucket": bucket,
            "eligible_for_training_review": (
                status in {"accepted_candidate", "needs_review"}
                and policy.training_allowed
                and not privacy_summary.get("has_high")
                and not privacy_summary.get("has_critical")
            ),
            "privacy": privacy_summary,
            "policy": policy.to_dict(),
        }

    def approve(
        self,
        intake_id: str,
        reviewer: str,
        decision: str,
        note: str = "",
    ) -> dict:
        decision = decision.lower().strip()
        if decision not in {"approve", "reject"}:
            raise ValueError("decision must be 'approve' or 'reject'")

        row = self._find_row(self.candidates_path, intake_id)
        if row is None:
            return {
                "status": "not_found_or_not_approvable",
                "intake_id": intake_id,
                "approved": False,
            }

        if row.get("privacy", {}).get("has_high") or row.get("privacy", {}).get("has_critical"):
            decision = "reject"
            note = (note + " rejected_by_policy_sensitive_data").strip()
        if not row.get("policy", {}).get("training_allowed", True):
            decision = "reject"
            note = (note + " rejected_by_domain_policy").strip()

        log_row = {
            "intake_id": intake_id,
            "timestamp": datetime.now().isoformat(),
            "reviewer": reviewer,
            "decision": decision,
            "note": note,
        }
        self._append_jsonl(self.approval_log_path, log_row)

        if decision == "reject":
            return {"status": "rejected", "intake_id": intake_id, "approved": False}

        approved = {
            **row,
            "approved_for_training": True,
            "approved_at": datetime.now().isoformat(),
            "approved_by": reviewer,
        }
        self._append_jsonl(self.approved_path, approved)
        return {
            "status": "approved",
            "intake_id": intake_id,
            "approved": True,
            "training_path": str(self.approved_path),
        }

    def stats(self) -> dict:
        return {
            "base_dir": str(self.base_dir),
            "candidates": self._count_jsonl(self.candidates_path),
            "quarantine": self._count_jsonl(self.quarantine_path),
            "approved": self._count_jsonl(self.approved_path),
            "approval_logs": self._count_jsonl(self.approval_log_path),
        }

    def _decide_status(self, record: LearningIntakeRecord, privacy: dict, policy: dict) -> str:
        if self.config.reject_without_consent and (not record.consent or not record.allow_training):
            return "rejected_no_consent"
        if privacy.get("has_critical"):
            return "quarantined_sensitive"
        if not policy.get("training_allowed", True):
            return "needs_review"
        if policy.get("requires_review"):
            return "needs_review"
        if privacy.get("has_high") or privacy.get("finding_count", 0) > 0:
            return "needs_review"
        return "accepted_candidate"

    def _merge_privacy_reports(self, reports: dict[str, dict]) -> dict:
        kinds: dict[str, int] = {}
        severities: dict[str, int] = {}
        finding_count = 0
        original_chars = 0
        redacted_chars = 0
        has_critical = False
        has_high = False
        for report in reports.values():
            finding_count += int(report.get("finding_count", 0))
            original_chars += int(report.get("original_chars", 0))
            redacted_chars += int(report.get("redacted_chars", 0))
            has_critical = has_critical or bool(report.get("has_critical"))
            has_high = has_high or bool(report.get("has_high"))
            for kind, count in report.get("kinds", {}).items():
                kinds[kind] = kinds.get(kind, 0) + int(count)
            for severity, count in report.get("severities", {}).items():
                severities[severity] = severities.get(severity, 0) + int(count)
        return {
            "original_chars": original_chars,
            "redacted_chars": redacted_chars,
            "finding_count": finding_count,
            "kinds": kinds,
            "severities": severities,
            "has_critical": has_critical,
            "has_high": has_high,
        }

    def _append_jsonl(self, path: Path, row: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _find_row(self, path: Path, intake_id: str) -> dict | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("intake_id") == intake_id:
                    return row
        return None

    def _count_jsonl(self, path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())


def _short_hash_hint(value: str) -> str:
    if not value:
        return ""
    return f"len:{len(value)}"


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in metadata.items():
        key_str = str(key)[:80]
        if key_str.lower() in {"password", "token", "secret", "authorization", "cookie"}:
            safe[key_str] = "<redacted:metadata_secret>"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[key_str] = value if not isinstance(value, str) else value[:500]
        else:
            safe[key_str] = str(value)[:500]
    return safe
