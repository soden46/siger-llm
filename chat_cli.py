import argparse
import os
from pathlib import Path
import torch
import re

from config.model_config import SigerConfig
from config.model_identity import canonical_model_name
from inference.chat import ChatSession
from inference.expertise_router import ExpertiseOrchestrator
from inference.generator import Generator
from inference.lampung_pipeline import LampungPipeline, LampungResponse
from inference.router import SigerRouter
from model.siger_model import SigerLM
from tokenizer.hybrid_tokenizer import build_tokenizer


DEFAULT_CHECKPOINT_CANDIDATES = [
    "checkpoints/lora/model_reasoning_merged.pt",
    "checkpoints/lora/model_indonesian_hf_mix_plus_kaggle_reasoning_merged.pt",
    "checkpoints/lora/model_indonesian_hf_mix_plus_kaggle_merged.pt",
    "checkpoints/lora/model_software_engineering_merged.pt",
    "checkpoints/lora/model_indonesian_hf_mix_merged.pt",
    "checkpoints/lora/model_general_merged.pt",
    "checkpoints/lora/model_lampung_merged.pt",
    "checkpoints/best_model.pt",
]


def infer_config_from_state_dict(state_dict: dict) -> SigerConfig:
    vocab_size, d_model = state_dict["embedding.weight"].shape
    layer_indices = set()

    for key in state_dict:
        match = re.match(r"layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))

    d_state = 16
    expand = 2
    d_conv = 4

    if "layers.0.ssm.A_log" in state_dict:
        d_inner, d_state = state_dict["layers.0.ssm.A_log"].shape
        expand = max(1, d_inner // d_model)

    if "layers.0.conv1d.weight" in state_dict:
        d_conv = state_dict["layers.0.conv1d.weight"].shape[-1]
    norm_type = "layernorm" if "norm_f.bias" in state_dict else "rmsnorm"
    moe_layers = set()
    moe_num_experts = 0
    moe_expert_hidden_mult = 2
    for key, value in state_dict.items():
        match = re.match(r"layers\.(\d+)\.moe\.experts\.(\d+)\.0\.weight", key)
        if match:
            moe_layers.add(int(match.group(1)))
            moe_num_experts = max(moe_num_experts, int(match.group(2)) + 1)
            if hasattr(value, "shape") and len(value.shape) == 2:
                moe_expert_hidden_mult = max(1, int(value.shape[0]) // d_model)

    use_moe = bool(moe_layers)
    moe_layers_every = 1
    if len(moe_layers) >= 2:
        sorted_layers = sorted(moe_layers)
        moe_layers_every = max(1, sorted_layers[1] - sorted_layers[0])
    elif len(moe_layers) == 1:
        moe_layers_every = max(1, next(iter(moe_layers)) + 1)

    return SigerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=max(layer_indices) + 1,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        max_seq_len=512,
        norm_type=norm_type,
        norm_bias=("norm_f.bias" in state_dict),
        use_moe=use_moe,
        moe_num_experts=moe_num_experts or 8,
        moe_top_k=2,
        moe_expert_hidden_mult=moe_expert_hidden_mult,
        moe_layers_every=moe_layers_every,
    )


def resolve_checkpoint(path: str | None = None) -> Path:
    env_path = os.environ.get("SIGER_CHECKPOINT")
    if path:
        candidate = Path(path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"SIGER_CHECKPOINT not found: {candidate}")

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        path_obj = Path(candidate)
        if path_obj.exists():
            return path_obj

    step_candidates = sorted(Path("checkpoints").glob("step_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if step_candidates:
        return step_candidates[0]

    lora_candidates = sorted(Path("checkpoints/lora").glob("*merged.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if lora_candidates:
        return lora_candidates[0]

    raise FileNotFoundError(
        "No checkpoint found. Use --checkpoint PATH or set SIGER_CHECKPOINT."
    )


def load_checkpoint_state(checkpoint_path: Path) -> tuple[dict, str]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_name = "SIGER"
    if isinstance(ckpt, dict):
        model_name = str(ckpt.get("model_name") or model_name)
        if "model_state" in ckpt:
            return ckpt["model_state"], model_name
    return ckpt, model_name


def load_model(checkpoint_path: str | None = None) -> tuple[SigerLM, object, Path]:
    resolved = resolve_checkpoint(checkpoint_path)
    print(f"Checkpoint: {resolved}")
    tok = build_tokenizer("auto")
    print(f"Tokenizer backend: {tok.backend} | vocab_size={tok.vocab_size}")

    ckpt, model_name = load_checkpoint_state(resolved)

    if "embedding.weight" in ckpt and ckpt["embedding.weight"].shape[0] != tok.vocab_size:
        raise RuntimeError(
            "Tokenizer tidak cocok dengan checkpoint. "
            f"tokenizer={tok.vocab_size}, checkpoint={ckpt['embedding.weight'].shape[0]}. "
            "Gunakan backend tokenizer yang sama dengan saat training."
        )

    config = infer_config_from_state_dict(ckpt)
    config.model_alias = canonical_model_name(model_name)
    model = SigerLM(config)

    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint | missing={len(missing)} unexpected={len(unexpected)}")
    print(f"Model name: {model.model_name}")
    return model, tok, resolved


def print_response(label: str, response: LampungResponse) -> None:
    print(f"{label}: {response.text!r}")
    print(f"Source: {response.source}")


def print_help() -> None:
    print("Ketik pertanyaan langsung untuk general assistant / auto router.")
    print("Command opsional:")
    print("  /help      tampilkan bantuan")
    print("  /exit      keluar")
    print("  /memory    lihat memory chat")
    print("  /lo-id     Lampung O -> Indonesia")
    print("  /id-lo     Indonesia -> Lampung O")
    print("  /lo-en     Lampung O -> English")
    print("  /reason    reasoning Lampung O -> Indonesia")
    print("  /reorder   susun kata Lampung O")
    print("  /expertise general expertise orchestrator")
    print("Mode angka lama juga masih bisa: 0 auto, 1 LO->ID, 2 ID->LO, 3 LO->EN, 4 reasoning, 5 chat, 6 susun kata, 7 expertise.")


def read_followup_input() -> str:
    return input("Input: ").strip()


def handle_manual_mode(
    mode: str,
    text: str,
    *,
    chat: ChatSession,
    lampung: LampungPipeline,
    router: SigerRouter,
    expertise: ExpertiseOrchestrator,
) -> None:
    if mode in {"0", "auto"}:
        response = router.route(text)
        print(f"Assistant: {response.text!r}")
        print(f"Route: {response.route}")
        print(f"Source: {response.source}")
    elif mode == "1":
        print_response("Indonesia", lampung.translate("Lampung O", "Bahasa Indonesia", text))
    elif mode == "2":
        print_response("Lampung O", lampung.translate("Bahasa Indonesia", "Lampung O", text))
    elif mode == "3":
        print_response("English", lampung.translate("Lampung O", "English", text))
    elif mode == "4":
        print_response("Reasoning", lampung.reason_lo_to_id(text))
    elif mode == "5":
        reply = chat.chat(
            text,
            max_new_tokens=80,
            temperature=0.3,
            top_k=20,
            top_p=0.8,
        )
        print(f"Chat: {reply!r}")
        print(f"Memory: {chat.memory_stats()}")
    elif mode == "6":
        print_response("Susunan", lampung.reorder("Lampung O", text))
    elif mode == "7":
        response = expertise.route(text, max_new_tokens=120)
        print(f"Expertise: {response.text!r}")
        print(f"Domains: {', '.join(response.domains)}")
        print(f"Summary: {response.task_summary}")
        print(f"Source: {response.source}")
    else:
        print("Mode tidak dikenal.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SIGER chat CLI and Kaggle smoke tester.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to latest known SIGER checkpoint.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--prompt", default=None, help="Run one prompt and exit.")
    parser.add_argument("--mode", default="auto", help="auto/chat/lo-id/id-lo/lo-en/reason/reorder or legacy 0-6.")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--info", action="store_true", help="Print model info and exit.")
    return parser.parse_args()


def normalize_mode(mode: str) -> str:
    return {
        "auto": "0",
        "lo-id": "1",
        "id-lo": "2",
        "lo-en": "3",
        "reason": "4",
        "chat": "5",
        "reorder": "6",
        "expertise": "7",
    }.get(mode.lower(), mode)


def main() -> None:
    args = parse_args()
    model, tok, checkpoint = load_model(args.checkpoint)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    gen = Generator(model, tok, device=device)
    chat = ChatSession(gen, max_context_tokens=1024)
    lampung = LampungPipeline(gen, tok)
    router = SigerRouter(chat, lampung)
    expertise = ExpertiseOrchestrator(gen, lampung)

    if args.info:
        print(f"Device: {device}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Memory: {chat.memory_stats()}")
        return

    if args.prompt:
        handle_manual_mode(
            normalize_mode(args.mode),
            args.prompt,
            chat=chat,
            lampung=lampung,
            router=router,
            expertise=expertise,
        )
        return

    print("SIGER_LLM CLI")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {device}")
    print("Langsung ketik pertanyaan. Router otomatis memilih general chat atau tool Lampung.")
    print("Ketik /help untuk command opsional, /exit untuk keluar.")

    while True:
        text = input("\nYou: ").strip()
        command = text.lower()

        if command in {"exit", "quit", "/exit", "/quit"}:
            break

        if not text:
            continue

        if command in {"/help", "help", "?"}:
            print_help()
            continue

        if command == "/memory":
            print(f"Memory: {chat.memory_stats()}")
            continue

        slash_modes = {
            "/auto": "0",
            "/lo-id": "1",
            "/id-lo": "2",
            "/lo-en": "3",
            "/reason": "4",
            "/chat": "5",
            "/reorder": "6",
            "/expertise": "7",
        }

        if command in slash_modes:
            followup = read_followup_input()
            if followup:
                handle_manual_mode(
                    slash_modes[command],
                    followup,
                    chat=chat,
                    lampung=lampung,
                    router=router,
                    expertise=expertise,
                )
            continue

        if command in {"0", "auto", "1", "2", "3", "4", "5", "6", "7", "expertise"}:
            followup = read_followup_input()
            if followup:
                handle_manual_mode(
                    normalize_mode(command),
                    followup,
                    chat=chat,
                    lampung=lampung,
                    router=router,
                    expertise=expertise,
                )
            continue

        response = router.route(text)
        print(f"Assistant: {response.text!r}")
        print(f"Route: {response.route}")
        print(f"Source: {response.source}")

        if response.route == "general_chat":
            print(f"Memory: {chat.memory_stats()}")


if __name__ == "__main__":
    main()
