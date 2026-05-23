from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DomainPolicyDecision:
    policy_name: str
    risk_level: str
    requires_review: bool
    training_allowed: bool
    recommended_use: str
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "risk_level": self.risk_level,
            "requires_review": self.requires_review,
            "training_allowed": self.training_allowed,
            "recommended_use": self.recommended_use,
            "reasons": list(self.reasons),
        }


class DomainPolicyEngine:
    """Business-context policy for app/web data intake.

    Regex privacy scans catch explicit secrets. Domain policy catches classes of
    data that are sensitive by context, such as CRM conversations or household
    finance records.
    """

    CRM_CHAT_MARKERS = {
        "crm_chat",
        "crm_conversation",
        "chatwoot",
        "whatsapp",
        "evolution_api",
        "evolution-api",
        "customer_support",
    }
    CRM_BEHAVIOR_MARKERS = {
        "crm_behavior",
        "crm_event",
        "user_behavior",
        "behavior_event",
        "analytics",
        "usage_event",
    }
    HOUSEHOLD_FINANCE_MARKERS = {
        "household_finance",
        "finance_tracking",
        "family_finance",
        "budgeting",
        "expense",
        "income",
    }

    def decide(
        self,
        *,
        source_type: str,
        domain: str,
        metadata: dict[str, Any],
        privacy: dict,
        learning_mode: str = "training_candidate",
    ) -> DomainPolicyDecision:
        markers = self._markers(source_type, domain, metadata)
        learning_mode = (learning_mode or "training_candidate").lower()

        if markers & self.HOUSEHOLD_FINANCE_MARKERS:
            aggregate_only = bool(metadata.get("aggregate_only")) or learning_mode == "aggregate"
            return DomainPolicyDecision(
                policy_name="household_finance",
                risk_level="high",
                requires_review=True,
                training_allowed=aggregate_only and not privacy.get("has_high") and not privacy.get("has_critical"),
                recommended_use="aggregate_patterns_only",
                reasons=[
                    "Household finance data can reveal income, debt, spending behavior, and family routines.",
                    "Use aggregates/categories/templates, not raw transactions or personal ledgers.",
                ],
            )

        if markers & self.CRM_CHAT_MARKERS:
            anonymized = bool(metadata.get("anonymized")) or learning_mode in {"anonymized", "template"}
            return DomainPolicyDecision(
                policy_name="crm_chat",
                risk_level="high",
                requires_review=True,
                training_allowed=anonymized and not privacy.get("has_high") and not privacy.get("has_critical"),
                recommended_use="anonymized_support_patterns_or_rag",
                reasons=[
                    "CRM conversations can contain customer PII, order details, complaints, and consent-sensitive content.",
                    "Prefer tenant-local RAG for customer-specific facts; train only on anonymized support patterns.",
                ],
            )

        if markers & self.CRM_BEHAVIOR_MARKERS:
            return DomainPolicyDecision(
                policy_name="crm_behavior",
                risk_level="medium",
                requires_review=bool(privacy.get("finding_count", 0)),
                training_allowed=not privacy.get("has_high") and not privacy.get("has_critical"),
                recommended_use="workflow_patterns_and_ui_assistance",
                reasons=[
                    "Behavior events are useful for CRM workflow learning, but user identifiers and tenant IDs must stay out.",
                ],
            )

        return DomainPolicyDecision(
            policy_name="default",
            risk_level="low" if not privacy.get("finding_count", 0) else "medium",
            requires_review=bool(privacy.get("finding_count", 0)),
            training_allowed=not privacy.get("has_high") and not privacy.get("has_critical"),
            recommended_use="general_training_candidate",
            reasons=[],
        )

    def _markers(self, source_type: str, domain: str, metadata: dict[str, Any]) -> set[str]:
        values = {
            str(source_type or "").lower(),
            str(domain or "").lower(),
            str(metadata.get("source") or "").lower(),
            str(metadata.get("provider") or "").lower(),
            str(metadata.get("event_type") or "").lower(),
            str(metadata.get("app_section") or "").lower(),
        }
        return {value.replace(" ", "_") for value in values if value}
