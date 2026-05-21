from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from chat_cli import load_model
from inference.chat import ChatSession
from inference.generator import Generator
from inference.lampung_pipeline import LampungPipeline
from inference.router import SigerRouter

from .checks import (
    contains_all,
    contains_none,
    fingerprint,
    iter_jsonl,
    metadata_license,
    normalize_for_match,
    normalize_text,
    repetition_ratio,
    rough_token_count,
    row_text,
)


MODEL_SUITE_KINDS = {"generation", "router", "lampung_lookup"}


class HarnessRunner:
    """Config-driven evaluation harness for SigerLM checkpoints and data."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        checkpoint: str | None = None,
        device: str | None = None,
        output_dir: str | None = None,
        allow_missing_model: bool = False,
    ) -> None:
        self.config = config
        self.name = normalize_text(config.get("name") or "siger_harness")
        self.checkpoint = checkpoint or config.get("checkpoint")
        self.device = device or config.get("device") or "auto"
        self.allow_missing_model = allow_missing_model
        self.output_root = Path(output_dir or config.get("output_dir") or "logs/eval/harness")
        self.run_dir = self._make_run_dir()
        self.model = None
        self.tokenizer = None
        self.generator = None
        self.chat = None
        self.lampung = None
        self.router = None
        self.model_error = ""

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        checkpoint: str | None = None,
        device: str | None = None,
        output_dir: str | None = None,
        allow_missing_model: bool = False,
    ) -> "HarnessRunner":
        with Path(path).open("r", encoding="utf-8") as f:
            config = json.load(f)
        return cls(
            config,
            checkpoint=checkpoint,
            device=device,
            output_dir=output_dir,
            allow_missing_model=allow_missing_model,
        )

    def run(self, only: set[str] | None = None) -> dict[str, Any]:
        start = time.time()
        suites = list(self.config.get("suites", []))
        if only:
            suites = [suite for suite in suites if suite.get("name") in only]

        if self._needs_model(suites):
            self._setup_model()

        suite_results = []
        for suite in suites:
            suite_results.append(self._run_suite(suite))

        elapsed = round(time.time() - start, 3)
        passed = sum(1 for result in suite_results if result["status"] == "passed")
        failed = sum(1 for result in suite_results if result["status"] == "failed")
        skipped = sum(1 for result in suite_results if result["status"] == "skipped")
        status = "failed" if failed else "passed"
        report = {
            "name": self.name,
            "status": status,
            "checkpoint": str(self.checkpoint or ""),
            "device": self.device,
            "run_dir": str(self.run_dir),
            "elapsed_sec": elapsed,
            "summary": {
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "total": len(suite_results),
            },
            "suites": suite_results,
        }
        self._write_report(report)
        return report

    def _make_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re_safe_name(self.name)
        run_dir = self.output_root / f"{timestamp}_{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _needs_model(self, suites: list[dict[str, Any]]) -> bool:
        return any(str(suite.get("kind")) in MODEL_SUITE_KINDS for suite in suites)

    def _setup_model(self) -> None:
        try:
            model, tokenizer, resolved = load_model(self.checkpoint)
            resolved_device = "cuda" if self.device == "auto" and torch.cuda.is_available() else self.device
            if resolved_device == "auto":
                resolved_device = "cpu"
            self.device = resolved_device
            self.checkpoint = str(resolved)
            self.model = model
            self.tokenizer = tokenizer
            self.generator = Generator(model, tokenizer, device=resolved_device)
            self.chat = ChatSession(self.generator, max_context_tokens=1024)
            self.lampung = LampungPipeline(self.generator, tokenizer)
            self.router = SigerRouter(self.chat, self.lampung)
        except Exception as exc:
            self.model_error = str(exc)
            if not self.allow_missing_model:
                raise

    def _run_suite(self, suite: dict[str, Any]) -> dict[str, Any]:
        kind = str(suite.get("kind") or "").strip()
        name = str(suite.get("name") or kind or "unnamed").strip()
        required = bool(suite.get("required", True))
        started = time.time()

        if kind in MODEL_SUITE_KINDS and self.generator is None:
            return {
                "name": name,
                "kind": kind,
                "status": "failed" if required else "skipped",
                "elapsed_sec": round(time.time() - started, 3),
                "error": self.model_error or "model is not available",
                "required": required,
            }

        try:
            if kind == "dataset_audit":
                result = self._run_dataset_audit(suite)
            elif kind == "generation":
                result = self._run_generation_suite(suite)
            elif kind == "router":
                result = self._run_router_suite(suite)
            elif kind == "lampung_lookup":
                result = self._run_lampung_lookup_suite(suite)
            else:
                result = {
                    "status": "failed" if required else "skipped",
                    "error": f"unsupported suite kind: {kind}",
                }
        except Exception as exc:
            result = {"status": "failed" if required else "skipped", "error": str(exc)}

        result.update(
            {
                "name": name,
                "kind": kind,
                "elapsed_sec": round(time.time() - started, 3),
                "required": required,
            }
        )
        return result

    def _load_cases(self, suite: dict[str, Any]) -> list[dict[str, Any]]:
        cases = list(suite.get("cases", []))
        path = suite.get("path")
        if path:
            for _, row, error in iter_jsonl(path):
                if error or row is None:
                    continue
                cases.append(row)
        max_cases = suite.get("max_cases")
        if max_cases is not None:
            cases = cases[: int(max_cases)]
        return cases

    def _run_generation_suite(self, suite: dict[str, Any]) -> dict[str, Any]:
        assert self.generator is not None
        cases = self._load_cases(suite)
        generation_args = dict(self.config.get("generation", {}))
        generation_args.update(suite.get("generation", {}))

        results = []
        for idx, case in enumerate(cases, start=1):
            prompt = str(case.get("prompt") or case.get("instruction") or "").strip()
            output = self.generator.generate(prompt, **generation_args) if prompt else ""
            checks = self._check_text_output(output, case)
            results.append(
                {
                    "id": case.get("id", idx),
                    "prompt": prompt,
                    "output": output,
                    "passed": checks["passed"],
                    "checks": checks,
                }
            )
        return self._case_summary(results)

    def _run_router_suite(self, suite: dict[str, Any]) -> dict[str, Any]:
        assert self.router is not None
        cases = self._load_cases(suite)
        max_new_tokens = int(suite.get("max_new_tokens", self.config.get("max_new_tokens", 80)))

        results = []
        for idx, case in enumerate(cases, start=1):
            prompt = str(case.get("prompt") or case.get("instruction") or "").strip()
            response = self.router.route(prompt, max_new_tokens=max_new_tokens) if prompt else None
            output = response.text if response else ""
            checks = self._check_text_output(output, case)
            expected_route = normalize_text(case.get("expected_route"))
            if expected_route:
                checks["expected_route"] = response is not None and response.route == expected_route
            expected_source_contains = normalize_text(case.get("expected_source_contains"))
            if expected_source_contains:
                source = response.source if response else ""
                checks["expected_source_contains"] = expected_source_contains.lower() in source.lower()
            checks["passed"] = all(value for key, value in checks.items() if key != "passed")
            results.append(
                {
                    "id": case.get("id", idx),
                    "prompt": prompt,
                    "output": output,
                    "route": response.route if response else "",
                    "source": response.source if response else "",
                    "passed": checks["passed"],
                    "checks": checks,
                }
            )
        return self._case_summary(results)

    def _run_lampung_lookup_suite(self, suite: dict[str, Any]) -> dict[str, Any]:
        assert self.lampung is not None
        cases = self._load_cases(suite)
        results = []

        for idx, case in enumerate(cases, start=1):
            source_lang = str(case.get("source_lang") or "Lampung O")
            target_lang = str(case.get("target_lang") or "Bahasa Indonesia")
            text = str(case.get("input") or case.get("prompt") or "").strip()
            response = self.lampung.translate(source_lang, target_lang, text)
            expected = normalize_for_match(case.get("expected") or case.get("output"))
            exact = normalize_for_match(response.text) == expected if expected else bool(response.text.strip())
            results.append(
                {
                    "id": case.get("id", idx),
                    "input": text,
                    "expected": case.get("expected") or case.get("output", ""),
                    "output": response.text,
                    "source": response.source,
                    "passed": exact,
                    "checks": {"exact_match": exact},
                }
            )
        return self._case_summary(results)

    def _run_dataset_audit(self, suite: dict[str, Any]) -> dict[str, Any]:
        path = Path(str(suite.get("path") or ""))
        if not path.exists():
            return {"status": "failed", "error": f"dataset not found: {path}", "cases": []}

        required_fields = list(suite.get("required_fields", ["instruction", "output"]))
        max_row_tokens = int(suite.get("max_row_tokens", 2048))
        allowed_licenses = {normalize_for_match(item) for item in suite.get("allowed_licenses", [])}
        metadata_path = suite.get("metadata_path")
        if allowed_licenses and metadata_path:
            license_value = normalize_for_match(metadata_license(metadata_path))
            license_ok = bool(license_value and license_value in allowed_licenses)
        else:
            license_value = ""
            license_ok = True

        seen = set()
        invalid_json = 0
        missing_required = 0
        too_long = 0
        duplicates = 0
        total = 0
        examples = []

        max_rows = suite.get("max_rows")
        for line_number, row, error in iter_jsonl(path):
            if max_rows is not None and total >= int(max_rows):
                break
            if error or row is None:
                invalid_json += 1
                examples.append({"line": line_number, "error": error})
                continue

            total += 1
            missing = [field for field in required_fields if not normalize_text(row.get(field))]
            if missing:
                missing_required += 1
                if len(examples) < 10:
                    examples.append({"line": line_number, "missing": missing})

            tokens = rough_token_count(row_text(row))
            if max_row_tokens > 0 and tokens > max_row_tokens:
                too_long += 1
                if len(examples) < 10:
                    examples.append({"line": line_number, "tokens": tokens, "error": "too_long"})

            fp = fingerprint(row)
            if fp in seen:
                duplicates += 1
            else:
                seen.add(fp)

        failures = invalid_json + missing_required + too_long + (0 if license_ok else 1)
        return {
            "status": "passed" if failures == 0 else "failed",
            "path": str(path),
            "total_rows": total,
            "invalid_json": invalid_json,
            "missing_required": missing_required,
            "too_long": too_long,
            "duplicates": duplicates,
            "metadata_license": license_value,
            "license_ok": license_ok,
            "examples": examples,
        }

    def _check_text_output(self, output: str, case: dict[str, Any]) -> dict[str, Any]:
        min_chars = int(case.get("min_chars", 1))
        max_repetition = float(case.get("max_repetition_ratio", 0.85))
        expected_contains = _string_list(case.get("expected_contains", []))
        expected_not_contains = _string_list(case.get("expected_not_contains", []))

        checks = {
            "non_empty": bool(output.strip()),
            "min_chars": len(output.strip()) >= min_chars,
            "max_repetition_ratio": repetition_ratio(output) <= max_repetition,
            "expected_contains": contains_all(output, expected_contains),
            "expected_not_contains": contains_none(output, expected_not_contains),
        }
        checks["passed"] = all(checks.values())
        return checks

    def _case_summary(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(results)
        passed = sum(1 for item in results if item.get("passed"))
        return {
            "status": "passed" if total > 0 and passed == total else "failed",
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "cases": results,
        }

    def _write_report(self, report: dict[str, Any]) -> None:
        json_path = self.run_dir / "report.json"
        json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        lines = [
            f"# SigerLM Harness Report: {report['name']}",
            "",
            f"- Status: {report['status']}",
            f"- Checkpoint: {report.get('checkpoint') or 'not loaded'}",
            f"- Device: {report.get('device')}",
            f"- Elapsed: {report.get('elapsed_sec')}s",
            "",
            "| Suite | Kind | Status | Passed | Failed |",
            "|---|---|---|---:|---:|",
        ]
        for suite in report["suites"]:
            lines.append(
                "| {name} | {kind} | {status} | {passed} | {failed} |".format(
                    name=suite.get("name", ""),
                    kind=suite.get("kind", ""),
                    status=suite.get("status", ""),
                    passed=suite.get("passed", ""),
                    failed=suite.get("failed", ""),
                )
            )
            if suite.get("error"):
                lines.append(f"\nError in `{suite.get('name')}`: {suite['error']}\n")

        md_path = self.run_dir / "summary.md"
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Harness report: {json_path}")
        print(f"Harness summary: {md_path}")


def re_safe_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return safe.strip("_") or "siger_harness"


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def run_harness(
    config_path: str,
    *,
    checkpoint: str | None = None,
    device: str | None = None,
    output_dir: str | None = None,
    allow_missing_model: bool = False,
    only: set[str] | None = None,
) -> dict[str, Any]:
    runner = HarnessRunner.from_file(
        config_path,
        checkpoint=checkpoint,
        device=device,
        output_dir=output_dir,
        allow_missing_model=allow_missing_model,
    )
    return runner.run(only=only)
