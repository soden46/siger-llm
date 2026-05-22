from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_DIR = Path.home() / ".siger"
CONFIG_PATH = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "checkpoint": None,
    "device": "auto",
    "mode": "dynamic",
    "max_context_tokens": 1024,
    "retrieval_top_k": 5,
    "retrieval_token_budget": None,
    "recent_turn_token_budget": None,
    "long_input_threshold_chars": 1200,
    "max_new_tokens": 120,
}

CONFIG_ALIASES = {
    "ckpt": "checkpoint",
    "context": "max_context_tokens",
    "ctx": "max_context_tokens",
}


@dataclass(frozen=True)
class RouteCommand:
    mode: str
    text: str
    label: str
    forced_domains: tuple[str, ...] = ()


ROUTE_COMMANDS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "code": (
        "expertise",
        "code",
        ("programming_basic", "programming_intermediate", "programming_expert"),
    ),
    "basic": ("expertise", "programming_basic", ("programming_basic",)),
    "debug": ("expertise", "programming_intermediate", ("programming_intermediate",)),
    "expert": ("expertise", "programming_expert", ("programming_expert",)),
    "reasoning": ("expertise", "reasoning", ("reasoning",)),
    "reason": ("expertise", "reasoning", ("reasoning",)),
    "general": ("expertise", "general_knowledge", ("general_knowledge",)),
    "lampung": ("expertise", "lampung", ("lampung",)),
    "chat": ("chat", "chat", ()),
    "auto": ("auto", "auto", ()),
}
ONE_SHOT_ROUTE_COMMANDS = {
    key: value for key, value in ROUTE_COMMANDS.items() if key != "chat"
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "config":
        return handle_config(args)

    config = merged_config(args)
    if args.command in ONE_SHOT_ROUTE_COMMANDS:
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit(f"Prompt kosong. Contoh: siger {args.command} \"buat REST API\"")
        return run_route_command(config, str(args.command), prompt)
    if args.command == "info":
        return run_info(config)
    if args.command == "ask":
        prompt = " ".join(args.prompt).strip()
        if not prompt:
            raise SystemExit("Prompt kosong. Contoh: siger ask \"Apa itu REST API?\"")
        return run_ask(config, prompt)

    return run_chat(config)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    common.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"], help="Inference device.")
    common.add_argument("--mode", default=None, help="dynamic/auto/chat/lo-id/id-lo/lo-en/reason/reorder/expertise.")
    common.add_argument("--max-new-tokens", type=int, default=None)
    common.add_argument("--max-context-tokens", type=int, default=None)
    common.add_argument("--retrieval-top-k", type=int, default=None)
    common.add_argument("--retrieval-token-budget", type=int, default=None)
    common.add_argument("--recent-turn-token-budget", type=int, default=None)
    common.add_argument("--long-input-threshold-chars", type=int, default=None)
    common.add_argument("--context-file", action="append", default=[], help="Load file into long-context memory.")

    parser = argparse.ArgumentParser(
        prog="siger",
        description="SigerLM command-line assistant.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("chat", parents=[common], help="Start interactive chat.")

    ask = sub.add_parser("ask", parents=[common], help="Ask once and exit.")
    ask.add_argument("prompt", nargs="*", help="Question or instruction.")

    for command, (_, label, _) in ONE_SHOT_ROUTE_COMMANDS.items():
        route = sub.add_parser(command, parents=[common], help=f"Ask through {label} route.")
        route.add_argument("prompt", nargs="*", help="Question or instruction.")

    sub.add_parser("info", parents=[common], help="Print model and runtime info.")

    config = sub.add_parser("config", help="Show or edit ~/.siger/config.json.")
    config_sub = config.add_subparsers(dest="config_command")
    config_sub.add_parser("path", help="Print config file path.")
    config_sub.add_parser("show", help="Show current config.")
    set_cmd = config_sub.add_parser("set", help="Set a config value.")
    set_cmd.add_argument("key")
    set_cmd.add_argument("value")
    unset_cmd = config_sub.add_parser("unset", help="Remove a config value.")
    unset_cmd.add_argument("key")

    parser.set_defaults(command="chat")
    return parser


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Config rusak: {CONFIG_PATH} ({exc})") from exc


def save_config(config: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def normalize_config_key(key: str) -> str:
    normalized = key.strip().replace("-", "_")
    return CONFIG_ALIASES.get(normalized, normalized)


def parse_config_value(key: str, value: str) -> Any:
    if value.lower() in {"none", "null"}:
        return None
    if key == "checkpoint":
        path = Path(value).expanduser()
        return str(path.resolve()) if path.exists() else value
    if key in {
        "max_context_tokens",
        "retrieval_top_k",
        "retrieval_token_budget",
        "recent_turn_token_budget",
        "long_input_threshold_chars",
        "max_new_tokens",
    }:
        return int(value)
    return value


def handle_config(args: argparse.Namespace) -> int:
    command = args.config_command or "show"
    current = load_config()

    if command == "path":
        print(CONFIG_PATH)
        return 0

    if command == "show":
        print(json.dumps({**DEFAULT_CONFIG, **current}, indent=2))
        return 0

    if command == "set":
        key = normalize_config_key(args.key)
        if key not in DEFAULT_CONFIG:
            raise SystemExit(f"Config key tidak dikenal: {args.key}")
        current[key] = parse_config_value(key, args.value)
        save_config(current)
        print(f"Set {key} = {current[key]!r}")
        return 0

    if command == "unset":
        key = normalize_config_key(args.key)
        current.pop(key, None)
        save_config(current)
        print(f"Unset {key}")
        return 0

    raise SystemExit(f"Config command tidak dikenal: {command}")


def merged_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {**DEFAULT_CONFIG, **load_config()}
    for key in DEFAULT_CONFIG:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    config["context_file"] = list(getattr(args, "context_file", []) or [])
    return config


class Runtime:
    def __init__(self, config: dict[str, Any]) -> None:
        from chat_cli import load_model
        from inference.chat import ChatSession
        from inference.expertise_router import ExpertiseOrchestrator
        from inference.generator import Generator
        from inference.lampung_pipeline import LampungPipeline
        from inference.router import SigerRouter

        self.config = config
        self.model, self.tokenizer, self.checkpoint = load_model(config.get("checkpoint"))
        self.device = resolve_device(str(config.get("device") or "auto"))
        self.generator = Generator(self.model, self.tokenizer, device=self.device)
        self.chat = ChatSession(
            self.generator,
            max_context_tokens=int(config["max_context_tokens"]),
            retrieval_top_k=int(config["retrieval_top_k"]),
            retrieval_token_budget=config.get("retrieval_token_budget"),
            recent_turn_token_budget=config.get("recent_turn_token_budget"),
            long_input_threshold_chars=int(config["long_input_threshold_chars"]),
        )
        self.lampung = LampungPipeline(self.generator, self.tokenizer)
        self.router = SigerRouter(self.chat, self.lampung)
        self.expertise = ExpertiseOrchestrator(
            self.generator,
            self.lampung,
            memory=self.chat.memory,
            retrieval_top_k=int(config["retrieval_top_k"]),
            retrieval_token_budget=config.get("retrieval_token_budget")
            or max(96, int(int(config["max_context_tokens"]) * 0.35)),
            long_input_threshold_chars=int(config["long_input_threshold_chars"]),
        )
        load_context_files(self.chat, config.get("context_file", []))


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_context_files(chat, paths: list[str]) -> None:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        chat.add_document(text, metadata={"source": str(path), "title": path.name})
        print(f"Loaded context: {path} ({len(text):,} chars)")


def run_info(config: dict[str, Any]) -> int:
    runtime = Runtime(config)
    print(f"Device: {runtime.device}")
    print(f"Checkpoint: {runtime.checkpoint}")
    print(f"Mode: {config['mode']}")
    print(f"Memory: {runtime.chat.memory_stats()}")
    return 0


def run_ask(config: dict[str, Any], prompt: str) -> int:
    runtime = Runtime(config)
    answer(runtime, normalize_mode(str(config["mode"])), prompt, int(config["max_new_tokens"]))
    return 0


def run_route_command(config: dict[str, Any], command: str, prompt: str) -> int:
    runtime = Runtime(config)
    mode, label, domains = ROUTE_COMMANDS[command]
    print(f"[route: {label}]")
    answer_forced(
        runtime,
        normalize_mode(mode),
        prompt,
        int(config["max_new_tokens"]),
        domains,
    )
    return 0


def run_chat(config: dict[str, Any]) -> int:
    runtime = Runtime(config)
    mode = normalize_mode(str(config["mode"]))
    print("SigerLM interactive CLI")
    print(f"Checkpoint: {runtime.checkpoint}")
    print(f"Device: {runtime.device}")
    print(f"Mode: {mode_label(mode)}")
    print("Ketik /help untuk command, /exit untuk keluar.")

    while True:
        try:
            text = input("\nsiger> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not text:
            continue

        command = text.lower()
        if command in {"exit", "quit", "/exit", "/quit"}:
            return 0
        if command in {"/help", "help", "?"}:
            print_interactive_help()
            continue
        if command == "/memory":
            print(f"Memory: {runtime.chat.memory_stats()}")
            continue
        if command.startswith("/mode "):
            mode = normalize_mode(command.split(maxsplit=1)[1])
            print(f"Mode: {mode_label(mode)}")
            continue
        if command.startswith("/doc "):
            path = Path(text[5:].strip())
            if not path.exists():
                print(f"Document not found: {path}")
                continue
            doc_text = path.read_text(encoding="utf-8", errors="ignore")
            runtime.chat.add_document(doc_text, metadata={"source": str(path), "title": path.name})
            print(f"Loaded document: {path} ({len(doc_text):,} chars)")
            print(f"Memory: {runtime.chat.memory_stats()}")
            continue
        if command.startswith("/checkpoint "):
            new_checkpoint = text.split(maxsplit=1)[1].strip()
            config["checkpoint"] = new_checkpoint
            runtime = Runtime(config)
            print(f"Checkpoint: {runtime.checkpoint}")
            print(f"Device: {runtime.device}")
            continue

        route = parse_route_command(text)
        if route is not None:
            payload = route.text or input("Input: ").strip()
            if payload:
                print(f"[route: {route.label}]")
                answer_forced(
                    runtime,
                    route.mode,
                    payload,
                    int(config["max_new_tokens"]),
                    route.forced_domains,
                )
            continue

        answer(runtime, mode, text, int(config["max_new_tokens"]))


def answer(runtime: Runtime, mode: str, text: str, max_new_tokens: int) -> None:
    route = parse_route_command(text)
    if route is not None and route.text:
        print(f"[route: {route.label}]")
        answer_forced(runtime, route.mode, route.text, max_new_tokens, route.forced_domains)
        return

    if mode == "dynamic":
        answer_dynamic(runtime, text, max_new_tokens)
        return

    answer_forced(runtime, mode, text, max_new_tokens)


def answer_dynamic(runtime: Runtime, text: str, max_new_tokens: int) -> None:
    selected_mode, reason = choose_dynamic_mode(runtime, text)
    print(f"[dynamic route: {mode_label(selected_mode)} | reason: {reason}]")
    answer_forced(runtime, selected_mode, text, max_new_tokens)


def answer_forced(
    runtime: Runtime,
    mode: str,
    text: str,
    max_new_tokens: int,
    forced_domains: tuple[str, ...] = (),
) -> None:
    if mode == "5":
        reply = runtime.chat.chat(
            text,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
        )
        print(reply.strip())
        return

    if mode == "7":
        response = runtime.expertise.route(
            text,
            max_new_tokens=max_new_tokens,
            forced_domains=list(forced_domains) or None,
        )
        print(response.text.strip())
        print(f"[domains: {', '.join(response.domains)} | source: {response.source}]")
        return

    if mode in {"1", "2", "3", "4", "6"}:
        response = run_domain_mode(runtime, mode, text, max_new_tokens)
        print(response.text.strip())
        print(f"[source: {response.source}]")
        return

    response = runtime.router.route(text, max_new_tokens=max_new_tokens)
    print(response.text.strip())
    print(f"[route: {response.route} | source: {response.source}]")


def choose_dynamic_mode(runtime: Runtime, text: str) -> tuple[str, str]:
    intent = runtime.router.detect_intent(text)
    specs = runtime.expertise.detect_expertise(text, max_domains=4)
    domains = [spec.name for spec in specs]

    if intent != "general_chat":
        if "lampung" in domains and len(domains) > 1:
            return "7", "lampung + multi-domain expertise"
        return "0", intent

    expertise_domains = {
        "programming_basic",
        "programming_intermediate",
        "programming_expert",
        "reasoning",
    }
    if any(domain in expertise_domains for domain in domains):
        return "7", ",".join(domains)
    if len(domains) > 1:
        return "7", ",".join(domains)
    return "0", "general chat"


def parse_route_command(text: str) -> RouteCommand | None:
    match = re.match(
        r"^\s*/([a-z][a-z0-9_-]{1,16})(?:\s+(.+))?\s*$",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None

    key = match.group(1).lower()
    payload = (match.group(2) or "").strip()
    if key not in ROUTE_COMMANDS:
        return None

    mode, label, domains = ROUTE_COMMANDS[key]
    return RouteCommand(
        mode=normalize_mode(mode),
        text=payload,
        label=label,
        forced_domains=domains,
    )


def run_domain_mode(runtime: Runtime, mode: str, text: str, max_new_tokens: int):
    if mode == "1":
        return runtime.lampung.translate("Lampung O", "Bahasa Indonesia", text)
    if mode == "2":
        return runtime.lampung.translate("Bahasa Indonesia", "Lampung O", text)
    if mode == "3":
        return runtime.lampung.translate("Lampung O", "English", text)
    if mode == "4":
        return runtime.lampung.reason_lo_to_id(text, max_new_tokens=max_new_tokens)
    if mode == "6":
        return runtime.lampung.reorder("Lampung O", text)
    raise ValueError(mode)


def mode_label(mode: str) -> str:
    return {
        "0": "auto",
        "1": "lo-id",
        "2": "id-lo",
        "3": "lo-en",
        "4": "reason",
        "5": "chat",
        "6": "reorder",
        "7": "expertise",
    }.get(mode, mode)


def normalize_mode(mode: str) -> str:
    return {
        "dynamic": "dynamic",
        "auto": "0",
        "lo-id": "1",
        "id-lo": "2",
        "lo-en": "3",
        "reason": "4",
        "chat": "5",
        "reorder": "6",
        "expertise": "7",
    }.get(mode.lower(), mode)


def print_interactive_help() -> None:
    print("Commands:")
    print("  /help                 tampilkan bantuan")
    print("  /exit                 keluar")
    print("  /memory               lihat memory")
    print("  /doc PATH             muat dokumen ke long-context memory")
    print("  /mode dynamic|auto|chat|expertise|lo-id|id-lo|lo-en|reason|reorder")
    print("  /checkpoint PATH      reload checkpoint")
    print()
    print("Route commands:")
    print("  /code TASK       coding/developer expertise")
    print("  /basic TASK      programming basic")
    print("  /debug TASK      algorithm/debug/intermediate")
    print("  /expert TASK     software engineering expert")
    print("  /reasoning TASK  reasoning")
    print("  /lampung TASK    Lampung expertise")
    print("  /general TASK    general knowledge")
    print("  /chat TASK       force direct chat")
    print("  /auto TASK       force auto router")
    print()
    print("Examples:")
    print("  siger ask \"Jelaskan REST API\"")
    print("  siger chat --context-file docs\\ARCHITECTURE.md")
    print("  siger config set checkpoint checkpoints\\lora\\model_cpu_repair_general_merged.pt")


if __name__ == "__main__":
    raise SystemExit(main())
