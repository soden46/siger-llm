# run_lora.py
from optimization.cpu.threading  import configure_cpu
from optimization.cpu.memory     import load_model_efficient
from config.model_config         import MambaConfig
from model.mamba_model           import MambaLM
from tokenizer.tokenizer         import MultilingualTokenizer
from lora.config                 import LoRAConfig
from lora.model                  import LoRAModel
from lora.dataset                import InstructionDataset
from lora.trainer                import LoRATrainer


def main():
    configure_cpu(n_cores=2)

    # ── 1. Load base model ────────────────────────────────
    print("📦 Loading base model...")
    model_config = MambaConfig(vocab_size=100277, d_model=512, n_layers=12)
    base_model   = load_model_efficient(
        MambaLM, model_config, "./checkpoints/best_model.pt"
    )

    # ── 2. Setup LoRA ─────────────────────────────────────
    lora_config = LoRAConfig(
        rank            = 8,
        alpha           = 16.0,
        target_modules  = ["in_proj", "out_proj", "x_proj", "dt_proj"],
        learning_rate   = 2e-4,
        max_steps       = 5_000,
        batch_size      = 4,
        grad_accum      = 8,
        max_seq_len     = 512,

        # Dataset — pilih salah satu dari RECOMMENDED_DATASETS
        dataset_name    = "HuggingFaceH4/ultrachat_200k",
        dataset_split   = "train_sft",
        max_samples     = 20_000,   # mulai kecil dulu di VPS

        save_dir        = "./checkpoints/lora",
        save_every      = 500,
    )

    lora_model = LoRAModel(base_model, lora_config)

    # ── 3. Load dataset ───────────────────────────────────
    tok     = MultilingualTokenizer()
    dataset = InstructionDataset(
        dataset_name = lora_config.dataset_name,
        tokenizer    = tok,
        split        = lora_config.dataset_split,
        max_seq_len  = lora_config.max_seq_len,
        max_samples  = lora_config.max_samples,
    )

    # ── 4. Train ──────────────────────────────────────────
    trainer = LoRATrainer(lora_model, lora_config, tok)
    trainer.train(dataset)

    # ── 5. Merge & export ─────────────────────────────────
    print("\n🔀 Merging LoRA into base model...")
    lora_model.merge_and_export("./checkpoints/lora/model_merged.pt")
    print("🎉 Done! Model siap di-deploy.")


if __name__ == "__main__":
    main()