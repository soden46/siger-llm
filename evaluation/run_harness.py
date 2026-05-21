from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.harness import run_harness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SigerLM engineering evaluation harness.")
    parser.add_argument("--config", default="configs/evaluation/harness_smoke.json")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path.")
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help="Skip/fail model-backed suites in the report instead of raising during setup.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of suite names to run.",
    )
    parser.add_argument("--no-fail", action="store_true", help="Always exit with status 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_harness(
        args.config,
        checkpoint=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        allow_missing_model=args.allow_missing_model,
        only=set(args.only) if args.only else None,
    )
    if report.get("status") != "passed" and not args.no_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
