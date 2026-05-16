# main.py
import torch
from config.model_config   import MambaConfig
from model.mamba_model     import MambaLM
from tokenizer.tokenizer   import MultilingualTokenizer
from training.dataset      import TextDataset
from training.trainer      import Trainer

# ── Config ──────────────────────────────────────────────────
TRAIN_CONFIG = {
    # Model
    "vocab_size":    100277,
    "d_model":       512,
    "n_layers":      12,

    # Training
    "max_steps":     100_000,
    "batch_size":    8,
    "max_seq_len":   1024,
    "grad_accum_steps": 4,      # effective batch = 8 * 4 = 32

    # Optimizer
    "max_lr":        3e-4,
    "min_lr":        3e-5,
    "warmup_steps":  2_000,
    "weight_decay":  0.1,
    "grad_clip":     1.0,

    # Logging & saving
    "log_interval":  10,
    "save_every":    500,
    "checkpoint_dir": "./checkpoints",
    "num_workers":   2,
}

def main():
    # 1. Tokenizer
    tok = MultilingualTokenizer()

    # 2. Dataset — ganti dengan corpus lo
    sample_texts = [
        "Ini adalah contoh teks bahasa Indonesia untuk training.",
        "This is a sample English text for training the model.",
        "def hello():\n    print('Hello, World!')",
        # ... load dari file/dataset lo
    ]
    dataset = TextDataset(
        texts=sample_texts,
        tokenizer=tok,
        max_seq_len=TRAIN_CONFIG["max_seq_len"]
    )

    # 3. Model
    model_config = MambaConfig(
        vocab_size=TRAIN_CONFIG["vocab_size"],
        d_model=TRAIN_CONFIG["d_model"],
        n_layers=TRAIN_CONFIG["n_layers"],
        max_seq_len=TRAIN_CONFIG["max_seq_len"],
    )
    model = MambaLM(model_config)

    # 4. Trainer
    trainer = Trainer(model, TRAIN_CONFIG)
    trainer.train(dataset, resume=True)

if __name__ == "__main__":
    main()