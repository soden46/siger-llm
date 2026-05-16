from pathlib import Path
from datasets import load_dataset
import pandas as pd

OUT_DIR = Path("data/lampung/processed/nusax_preview")
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading NusaX dataset...")

dataset = load_dataset("indonlp/NusaX-MT")

print(dataset)

for split in dataset.keys():
    df = dataset[split].to_pandas()

    print(f"\n=== SPLIT: {split} ===")
    print(df.head())

    print("\nColumns:")
    print(df.columns.tolist())

    output_file = OUT_DIR / f"{split}_preview.csv"

    df.head(100).to_csv(output_file, index=False)

    print(f"Saved preview to {output_file}")

print("\nDone.")