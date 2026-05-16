from pathlib import Path
import json

FINAL = Path("data/lampung/final")

def build(split: str):
    input_path = FINAL / f"{split}.jsonl"
    output_path = FINAL / f"{split}_instruction.jsonl"

    if not input_path.exists():
        print(f"Skip {input_path}")
        return

    tasks = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            dialect = row.get("dialect", "o")
            lampung = row["lampung"]
            indo = row["indonesian"]
            eng = row.get("english", "")

            dialect_name = "Lampung O" if dialect == "o" else "Lampung Nyo"

            tasks.append({
                "instruction": f"Terjemahkan {dialect_name} ke Bahasa Indonesia",
                "input": lampung,
                "output": indo,
            })

            tasks.append({
                "instruction": f"Terjemahkan Bahasa Indonesia ke {dialect_name}",
                "input": indo,
                "output": lampung,
            })

            if eng:
                tasks.append({
                    "instruction": f"Translate {dialect_name} to English",
                    "input": lampung,
                    "output": eng,
                })

                tasks.append({
                    "instruction": f"Translate English to {dialect_name}",
                    "input": eng,
                    "output": lampung,
                })

    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"{split}: {len(tasks)} rows saved to {output_path}")

for split in ["train", "valid", "test"]:
    build(split)