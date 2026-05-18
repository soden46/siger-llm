from __future__ import annotations

import re


SIGER_BASE_NAME = "SIGER"


def canonical_model_name(alias: str | None = None) -> str:
    """Return the public model name while keeping SIGER as immutable base."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", str(alias or "").strip()).strip("-")
    if not cleaned:
        return SIGER_BASE_NAME

    upper = cleaned.upper()
    if upper == SIGER_BASE_NAME:
        return SIGER_BASE_NAME
    if upper.startswith(f"{SIGER_BASE_NAME}-"):
        suffix = upper[len(SIGER_BASE_NAME) + 1 :].strip("-")
        return f"{SIGER_BASE_NAME}-{suffix}" if suffix else SIGER_BASE_NAME
    return f"{SIGER_BASE_NAME}-{upper}"
