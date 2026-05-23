from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.tool_result_compressor import CompressionConfig, ToolResultCompressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compress verbose tool output before storing it as SigerLM context.",
    )
    parser.add_argument("--command", default="", help="Original command, e.g. 'git diff' or 'rg TODO'.")
    parser.add_argument("--input", help="Read tool output from a file. Defaults to stdin.")
    parser.add_argument("--output", help="Optional path to write compressed output.")
    parser.add_argument("--min-chars", type=int, default=1200)
    parser.add_argument("--max-lines", type=int, default=120)
    parser.add_argument("--no-header", action="store_true")
    parser.add_argument("--stats", action="store_true", help="Print compression stats to stderr.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8", errors="replace")
    else:
        text = sys.stdin.read()

    compressor = ToolResultCompressor(
        CompressionConfig(
            min_chars=args.min_chars,
            max_lines=args.max_lines,
            include_header=not args.no_header,
        )
    )
    result = compressor.compress(text, command=args.command, source="cli")

    if args.output:
        Path(args.output).write_text(result.text, encoding="utf-8")
    else:
        sys.stdout.write(result.text)
        if result.text and not result.text.endswith("\n"):
            sys.stdout.write("\n")

    if args.stats:
        print(
            (
                f"mode={result.mode} compressed={result.compressed} "
                f"chars={result.original_chars}->{result.compressed_chars} "
                f"saved={result.saved_ratio:.1%}"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
