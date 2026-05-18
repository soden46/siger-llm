from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path
from typing import Iterable


DEFAULT_EXCLUDE_PATTERNS = [
    "checkpoints/*",
    "**/checkpoints/*",
    "**/.git/*",
    "**/__pycache__/*",
    "**/*.log",
]


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    normalized = value.replace(";", ",")
    return [item.strip() for item in normalized.split(",") if item.strip()]


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    normalized = path.as_posix()
    return any(fnmatch(normalized, pattern) for pattern in patterns)


def _expand_source(source: str | Path, pattern: str) -> list[Path]:
    path = Path(source)
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(path.rglob(pattern))
    return sorted(Path().glob(str(path)))


def resolve_text_sources(config: dict) -> list[Path]:
    """Resolve base-training text files from config and environment.

    Environment overrides:
    - SIGER_TEXT_SOURCES: comma/semicolon-separated files, dirs, or globs
    - SIGER_TEXT_INCLUDE: glob pattern inside source dirs, default *.txt
    - SIGER_TEXT_EXCLUDE: comma/semicolon-separated exclude globs
    - SIGER_MAX_TEXT_FILES: optional cap after sorting/dedup
    """

    sources = _split_env_list(os.environ.get("SIGER_TEXT_SOURCES"))
    if not sources:
        configured = config.get("text_sources")
        if isinstance(configured, str):
            sources = [configured]
        elif configured:
            sources = [str(item) for item in configured]
    if not sources:
        sources = ["data"]

    include_pattern = os.environ.get("SIGER_TEXT_INCLUDE") or str(config.get("text_include", "*.txt"))
    exclude_patterns = [
        *DEFAULT_EXCLUDE_PATTERNS,
        *[str(item) for item in config.get("text_exclude", [])],
        *_split_env_list(os.environ.get("SIGER_TEXT_EXCLUDE")),
    ]

    resolved: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        for path in _expand_source(source, include_pattern):
            if not path.exists() or not path.is_file():
                continue
            try:
                absolute = path.resolve()
            except OSError:
                continue
            if absolute in seen:
                continue
            if _matches_any(path, exclude_patterns) or _matches_any(absolute, exclude_patterns):
                continue
            seen.add(absolute)
            resolved.append(path)

    resolved = sorted(resolved, key=lambda item: item.as_posix())
    max_files_value = os.environ.get("SIGER_MAX_TEXT_FILES") or config.get("max_text_files")
    if max_files_value:
        resolved = resolved[: int(max_files_value)]
    return resolved
