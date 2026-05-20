from pathlib import Path

from optimization.cpu.threading import configure_cpu
from optimization.cpu.memory    import load_model_efficient
from config.model_config        import SigerConfig
from model.siger_model          import SigerLM
from tokenizer.hybrid_tokenizer import build_tokenizer
from inference.generator        import Generator
from evaluation.runner          import EvaluationRunner


def main():
    configure_cpu(n_cores=2)

    checkpoint_dir = Path("./checkpoints")
    config_path = checkpoint_dir / "model_config.json"

    if config_path.exists():
        print(f"Loading model config from {config_path}")
        config = SigerConfig.from_json(str(config_path))
    else:
        print("model_config.json not found; using SigerConfig.base() fallback.")
        config = SigerConfig.base()

    tok = build_tokenizer("auto")
    config.vocab_size = tok.vocab_size
    print(f"Model size approx: {config.model_size_approx} | vocab_size={config.vocab_size}")

    model_path = checkpoint_dir / "best_model.pt"
    if not model_path.exists():
        latest_checkpoints = sorted(checkpoint_dir.glob("step_*.pt"))
        if latest_checkpoints:
            model_path = latest_checkpoints[-1]
            print(f"best_model.pt not found; using latest checkpoint: {model_path}")
        else:
            raise FileNotFoundError(f"No model checkpoint (.pt) found in {checkpoint_dir}")

    model = load_model_efficient(SigerLM, config, str(model_path))
    gen = Generator(model, tok)

    # Run evaluation
    runner = EvaluationRunner(
        model     = model,
        tokenizer = tok,
        generator = gen,
        device    = "cpu",
    )

    print("Running evaluation for base model...")
    runner.run(
        n_samples = 100,   # kecil dulu di VPS
        tag       = "base_model",
    )

    # Kalau udah ada LoRA:
    # lora_model = LoRAModel(model, lora_config)
    # lora_model.load_lora("./checkpoints/lora/lora_step_005000.pt")
    # runner.model = lora_model
    # runner.run(tag="lora_finetuned")


if __name__ == "__main__":
    main()
