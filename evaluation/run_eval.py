# run_eval.py
from optimization.cpu.threading import configure_cpu
from optimization.cpu.memory    import load_model_efficient
from config.model_config        import MambaConfig
from model.mamba_model          import MambaLM
from tokenizer.tokenizer        import MultilingualTokenizer
from inference.generator        import Generator
from evaluation.runner          import EvaluationRunner


def main():
    configure_cpu(n_cores=2)

    # Load model
    config = MambaConfig(vocab_size=100277, d_model=512, n_layers=12)
    model  = load_model_efficient(MambaLM, config, "./checkpoints/best_model.pt")
    tok    = MultilingualTokenizer()
    gen    = Generator(model, tok)

    # Run evaluation
    runner = EvaluationRunner(
        model     = model,
        tokenizer = tok,
        generator = gen,
        device    = "cpu",
    )

    # Eval base model
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