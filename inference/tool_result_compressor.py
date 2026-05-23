from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
GREP_RE = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):(?P<text>.*)$")
TREE_PREFIX_RE = re.compile(r"^[\s|`+\\-]*")


@dataclass
class CompressionConfig:
    enabled: bool = True
    min_chars: int = 1200
    max_lines: int = 120
    max_tail_lines: int = 24
    max_items: int = 80
    max_items_per_file: int = 8
    include_header: bool = True
    fallback_to_original_if_larger: bool = True


@dataclass
class CompressionResult:
    text: str
    original_chars: int
    compressed_chars: int
    mode: str
    compressed: bool
    lossless: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def saved_ratio(self) -> float:
        if self.original_chars <= 0:
            return 0.0
        return max(0.0, 1.0 - (self.compressed_chars / self.original_chars))

    def metadata(self) -> dict:
        return {
            "type": "tool_result",
            "compression_mode": self.mode,
            "compressed": self.compressed,
            "lossless": self.lossless,
            "original_chars": self.original_chars,
            "compressed_chars": self.compressed_chars,
            "saved_ratio": round(self.saved_ratio, 4),
            "warnings": list(self.warnings),
        }


class ToolResultCompressor:
    """
    Conservative RTK/9Router-style compressor for verbose tool output.

    This layer compresses context before it reaches the model. It does not alter
    tokenizer training, model weights, or the base SigerLM architecture.
    """

    def __init__(self, config: CompressionConfig | None = None):
        self.config = config or CompressionConfig()

    def compress(self, output: str, command: str = "", source: str = "tool") -> CompressionResult:
        original = _clean_output(output)
        original_chars = len(original)
        if not self.config.enabled or original_chars == 0:
            return self._unchanged(original, "disabled" if not self.config.enabled else "empty")

        mode = self.detect_mode(command, original)
        if original_chars < self.config.min_chars and mode == "generic":
            return self._unchanged(original, "short")

        if mode == "git_status":
            body = self._compress_git_status(original)
        elif mode == "git_diff":
            body = self._compress_git_diff(original)
        elif mode == "grep":
            body = self._compress_grep(original)
        elif mode == "listing":
            body = self._compress_listing(original)
        elif mode == "test_log":
            body = self._compress_test_log(original)
        elif mode == "install_log":
            body = self._compress_install_log(original)
        else:
            body = self._compress_generic(original)

        warnings = ["lossy_summary"]
        compressed_text = body.strip()
        if self.config.include_header and compressed_text != original:
            saved = 1.0 - (len(compressed_text) / max(1, original_chars))
            header = (
                f"[compressed tool_result source={source} mode={mode} "
                f"chars={original_chars}->{len(compressed_text)} saved={saved:.1%} lossless=false]"
            )
            compressed_text = f"{header}\n{compressed_text}"

        if (
            self.config.fallback_to_original_if_larger
            and len(compressed_text) >= original_chars
        ):
            return self._unchanged(original, f"{mode}_no_gain")

        return CompressionResult(
            text=compressed_text,
            original_chars=original_chars,
            compressed_chars=len(compressed_text),
            mode=mode,
            compressed=compressed_text != original,
            lossless=False,
            warnings=warnings,
        )

    def detect_mode(self, command: str, output: str) -> str:
        cmd = command.lower()
        sample = output[:4000].lower()
        if "git status" in cmd or "on branch " in sample or "changes not staged" in sample:
            return "git_status"
        if "git diff" in cmd or "\ndiff --git " in f"\n{output[:4000]}":
            return "git_diff"
        if any(name in cmd for name in ("rg ", "grep ", "ripgrep")) or _has_many_grep_lines(output):
            return "grep"
        if any(name in cmd for name in ("ls", "dir", "tree", "find")) and not _looks_like_log(output):
            return "listing"
        if any(name in cmd for name in ("npm ", "pnpm ", "yarn ", "pip ", "composer ")):
            return "install_log"
        if _looks_like_log(output):
            return "test_log"
        return "generic"

    def _unchanged(self, text: str, mode: str) -> CompressionResult:
        return CompressionResult(
            text=text,
            original_chars=len(text),
            compressed_chars=len(text),
            mode=mode,
            compressed=False,
            lossless=True,
        )

    def _compress_git_status(self, text: str) -> str:
        branch: list[str] = []
        changed: list[str] = []
        untracked: list[str] = []
        porcelain: list[str] = []
        capture_untracked = False

        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(("On branch", "Your branch", "nothing to commit")):
                branch.append(line)
                continue
            if re.match(r"^(M|A|D|R|C|U|\?\?)\s+", line):
                porcelain.append(line)
                continue
            if "Untracked files:" in line:
                capture_untracked = True
                continue
            if line.startswith(("Changes to be committed", "Changes not staged")):
                capture_untracked = False
                continue
            if re.match(r"^(modified|new file|deleted|renamed|both modified):", line):
                changed.append(line)
                continue
            if capture_untracked and not line.startswith("(") and not line.endswith(":"):
                untracked.append(line)

        parts = []
        if branch:
            parts.append("status:\n" + "\n".join(f"- {x}" for x in _dedupe(branch)))
        if porcelain:
            parts.append("porcelain_changes:\n" + "\n".join(f"- {x}" for x in porcelain[: self.config.max_items]))
        if changed:
            parts.append("changed:\n" + "\n".join(f"- {x}" for x in changed[: self.config.max_items]))
        if untracked:
            parts.append("untracked:\n" + "\n".join(f"- {x}" for x in untracked[: self.config.max_items]))
        return "\n\n".join(parts) if parts else self._compress_generic(text)

    def _compress_git_diff(self, text: str) -> str:
        files: list[dict] = []
        current: dict | None = None

        for raw in text.splitlines():
            if raw.startswith("diff --git "):
                if current:
                    files.append(current)
                parts = raw.split()
                path = parts[-1][2:] if len(parts) >= 4 and parts[-1].startswith("b/") else raw
                current = {"path": path, "plus": 0, "minus": 0, "hunks": [], "samples": []}
                continue
            if current is None:
                continue
            if raw.startswith("@@"):
                current["hunks"].append(raw.strip())
                continue
            if raw.startswith("+++") or raw.startswith("---"):
                continue
            if raw.startswith("+"):
                current["plus"] += 1
                _append_sample(current["samples"], raw, self.config.max_items_per_file)
            elif raw.startswith("-"):
                current["minus"] += 1
                _append_sample(current["samples"], raw, self.config.max_items_per_file)

        if current:
            files.append(current)
        if not files:
            return self._compress_generic(text)

        lines = ["changed_files:"]
        for item in files[: self.config.max_items]:
            lines.append(f"- {item['path']}: +{item['plus']} -{item['minus']}")
            for hunk in item["hunks"][:3]:
                lines.append(f"  hunk: {hunk}")
            for sample in item["samples"][: self.config.max_items_per_file]:
                lines.append(f"  {sample[:220]}")
        if len(files) > self.config.max_items:
            lines.append(f"- ... {len(files) - self.config.max_items} more files omitted")
        return "\n".join(lines)

    def _compress_grep(self, text: str) -> str:
        by_file: dict[str, list[str]] = defaultdict(list)
        other: list[str] = []

        for raw in text.splitlines():
            match = GREP_RE.match(raw)
            if not match:
                if len(other) < self.config.max_tail_lines:
                    other.append(raw.strip())
                continue
            path = match.group("path")
            line_no = match.group("line")
            body = match.group("text").strip()
            if len(by_file[path]) < self.config.max_items_per_file:
                by_file[path].append(f"{line_no}: {body[:220]}")

        if not by_file:
            return self._compress_generic(text)

        lines = ["matches:"]
        for path in sorted(by_file)[: self.config.max_items]:
            lines.append(f"- {path}")
            for item in by_file[path]:
                lines.append(f"  {item}")
        if other:
            lines.append("other:")
            lines.extend(f"- {x[:220]}" for x in other)
        return "\n".join(lines)

    def _compress_listing(self, text: str) -> str:
        dirs: list[str] = []
        files: list[str] = []

        for raw in text.splitlines():
            name = _extract_listing_name(raw)
            if not name:
                continue
            if name.endswith("/") or raw.lstrip().startswith("d"):
                dirs.append(name.rstrip("/"))
            else:
                files.append(name)

        dirs = _dedupe(dirs)[: self.config.max_items]
        files = _dedupe(files)[: self.config.max_items]
        if not dirs and not files:
            return self._compress_generic(text)

        parts = []
        if dirs:
            parts.append("dirs: " + ", ".join(dirs))
        if files:
            parts.append("files: " + ", ".join(files))
        return "\n".join(parts)

    def _compress_test_log(self, text: str) -> str:
        lines = text.splitlines()
        important = _important_log_lines(lines, limit=self.config.max_lines)
        tail = _tail(lines, self.config.max_tail_lines)
        parts = []
        if important:
            parts.append("important_log_lines:\n" + "\n".join(important))
        if tail:
            parts.append("tail:\n" + "\n".join(tail))
        return "\n\n".join(parts) if parts else self._compress_generic(text)

    def _compress_install_log(self, text: str) -> str:
        lines = text.splitlines()
        important = _important_log_lines(
            lines,
            keywords=("error", "err!", "warn", "warning", "failed", "missing", "deprecated", "conflict"),
            limit=self.config.max_lines,
        )
        tail = _tail(lines, self.config.max_tail_lines)
        parts = []
        if important:
            parts.append("package_log_signals:\n" + "\n".join(important))
        if tail:
            parts.append("tail:\n" + "\n".join(tail))
        return "\n\n".join(parts) if parts else self._compress_generic(text)

    def _compress_generic(self, text: str) -> str:
        lines = text.splitlines()
        if len(lines) <= self.config.max_lines:
            return text

        important = _important_log_lines(lines, limit=max(20, self.config.max_lines // 2))
        head_count = max(12, self.config.max_lines // 4)
        tail_count = max(12, self.config.max_tail_lines)
        parts = ["head:", *_tail(lines[:head_count], head_count)]
        if important:
            parts.extend(["", "signals:", *important])
        parts.extend(["", "tail:", *_tail(lines, tail_count)])
        return "\n".join(parts)


def compress_tool_result(
    output: str,
    command: str = "",
    config: CompressionConfig | None = None,
    source: str = "tool",
) -> CompressionResult:
    return ToolResultCompressor(config).compress(output, command=command, source=source)


def _clean_output(text: str) -> str:
    return ANSI_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _dedupe(items: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _append_sample(samples: list[str], line: str, limit: int) -> None:
    stripped = line.rstrip()
    if len(samples) < limit and _is_signal_line(stripped):
        samples.append(stripped)


def _has_many_grep_lines(text: str) -> bool:
    count = 0
    for line in text.splitlines()[:80]:
        if GREP_RE.match(line):
            count += 1
    return count >= 3


def _looks_like_log(text: str) -> bool:
    sample = text[:6000].lower()
    return any(
        marker in sample
        for marker in (
            "traceback",
            "exception",
            "failed",
            "error:",
            " assertion",
            "pytest",
            "test failed",
            "npm err!",
            "composer detected",
        )
    )


def _extract_listing_name(raw: str) -> str:
    line = raw.strip()
    if not line or line.startswith(("total ", "Directory:", "Mode ", "----")):
        return ""

    parts = line.split()
    if len(parts) >= 9 and re.match(r"^[\-dl]", parts[0]):
        return parts[-1]
    if len(parts) >= 4 and re.match(r"^\d{2,4}[-/]\d{1,2}[-/]\d{1,2}", parts[0]):
        return parts[-1]

    if line.startswith(("|--", "`--", "+--", "\\--")) or line.startswith("|   "):
        tree_name = TREE_PREFIX_RE.sub("", line).strip()
        if tree_name and tree_name not in (".", ".."):
            line = tree_name
    return line.rstrip("/")


def _important_log_lines(
    lines: list[str],
    keywords: tuple[str, ...] = (
        "error",
        "failed",
        "failure",
        "exception",
        "traceback",
        "assert",
        "panic",
        "fatal",
        "warning",
    ),
    limit: int = 120,
) -> list[str]:
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        lower = line.lower()
        if _is_signal_line(line, keywords):
            out.append(line[:260])
            if len(out) >= limit:
                break
    return _dedupe(out)


def _is_signal_line(
    line: str,
    keywords: tuple[str, ...] = (
        "error",
        "failed",
        "failure",
        "exception",
        "traceback",
        "assert",
        "panic",
        "fatal",
        "warning",
        "todo",
        "fixme",
        "raise ",
        "return ",
    ),
) -> bool:
    lower = line.lower()
    return any(word in lower for word in keywords)


def _tail(lines: list[str], count: int) -> list[str]:
    return [line.rstrip()[:260] for line in lines[-count:] if line.strip()]
