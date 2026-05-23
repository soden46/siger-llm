from __future__ import annotations

import argparse

import torch
import uvicorn

from chat_cli import load_model
from inference.api import app, init_api
from inference.generator import Generator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SigerLM FastAPI server.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to SIGER_CHECKPOINT or latest known checkpoint.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload for local API development.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, tokenizer, checkpoint = load_model(args.checkpoint)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    generator = Generator(model, tokenizer, device=device)
    init_api(generator, checkpoint_path=str(checkpoint))
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
