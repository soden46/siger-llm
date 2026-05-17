import torch
import re

from config.model_config import SigerConfig
from inference.chat import ChatSession
from inference.generator import Generator
from inference.lampung_pipeline import LampungPipeline, LampungResponse
from inference.router import SigerRouter
from model.siger_model import SigerLM
from tokenizer.hybrid_tokenizer import build_tokenizer


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

    return SigerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=max(layer_indices) + 1,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        max_seq_len=512,
    )


def load_model(checkpoint_path: str) -> tuple[SigerLM, object]:
    tok = build_tokenizer("auto")
    print(f"Tokenizer backend: {tok.backend} | vocab_size={tok.vocab_size}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if "model_state" in ckpt:
        ckpt = ckpt["model_state"]

    if "embedding.weight" in ckpt and ckpt["embedding.weight"].shape[0] != tok.vocab_size:
        raise RuntimeError(
            "Tokenizer tidak cocok dengan checkpoint. "
            f"tokenizer={tok.vocab_size}, checkpoint={ckpt['embedding.weight'].shape[0]}. "
            "Gunakan backend tokenizer yang sama dengan saat training."
        )

    config = infer_config_from_state_dict(ckpt)
    model = SigerLM(config)

    missing, unexpected = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded checkpoint | missing={len(missing)} unexpected={len(unexpected)}")
    return model, tok


def print_response(label: str, response: LampungResponse) -> None:
    print(f"{label}: {response.text!r}")
    print(f"Source: {response.source}")


def main() -> None:
    model, tok = load_model("checkpoints/lora/model_lampung_merged.pt")
    gen = Generator(model, tok, device="cpu")
    chat = ChatSession(gen, max_context_tokens=1024)
    lampung = LampungPipeline(gen, tok)
    router = SigerRouter(chat, lampung)

    print("SIGER_LLM CLI")
    print(
        "Mode: 0=auto/general, 1=LO->ID, 2=ID->LO, 3=LO->EN, "
        "4=reasoning, 5=chat, 6=susun kata, exit=keluar"
    )

    while True:
        mode = input("\nMode: ").strip().lower()
        if mode in {"exit", "quit"}:
            break

        text = input("Input: ").strip()
        if not text:
            continue

        if mode in {"0", "auto"}:
            response = router.route(text)
            print(f"Assistant: {response.text!r}")
            print(f"Route: {response.route}")
            print(f"Source: {response.source}")
            continue
        elif mode == "1":
            print_response("Indonesia", lampung.translate("Lampung O", "Bahasa Indonesia", text))
            continue
        elif mode == "2":
            print_response("Lampung O", lampung.translate("Bahasa Indonesia", "Lampung O", text))
            continue
        elif mode == "3":
            print_response("English", lampung.translate("Lampung O", "English", text))
            continue
        elif mode == "4":
            print_response("Reasoning", lampung.reason_lo_to_id(text))
            continue
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
            continue
        elif mode == "6":
            print_response("Susunan", lampung.reorder("Lampung O", text))
            continue
        else:
            print("Mode tidak dikenal.")
            continue


if __name__ == "__main__":
    main()
