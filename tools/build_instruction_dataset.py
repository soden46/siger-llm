from pathlib import Path
import json
import random

FINAL = Path("data/lampung/final")


def make_reasoning(source_lang: str, target_lang: str, source: str, target: str) -> str:
    return (
        f"Teks {source_lang} \"{source}\" diterjemahkan ke {target_lang} "
        f"sebagai \"{target}\". Pertahankan nama diri dan istilah budaya "
        "bila tidak punya padanan langsung."
    )


def shuffled_words(text: str) -> str:
    words = text.split()
    if len(words) < 3:
        return ""

    rng = random.Random(text)
    shuffled = words[:]

    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != words:
            break

    if shuffled == words:
        return ""

    return " ".join(shuffled)


def add_word_order_task(tasks: list[dict], lang: str, text: str) -> None:
    words = text.split()
    if len(words) < 3 or len(words) > 32:
        return

    shuffled = shuffled_words(text)
    if not shuffled:
        return

    tasks.append({
        "instruction": f"Susun kata {lang} berikut menjadi kalimat yang benar",
        "input": shuffled,
        "output": text,
    })


def build(split: str):
    input_path = FINAL / f"{split}.jsonl"
    output_path = FINAL / f"{split}_instruction.jsonl"
    reasoning_output_path = FINAL / f"{split}_reasoning_instruction.jsonl"
    word_order_output_path = FINAL / f"{split}_word_order_instruction.jsonl"
    augmented_output_path = FINAL / f"{split}_augmented_instruction.jsonl"

    if not input_path.exists():
        print(f"Skip {input_path}")
        return

    tasks = []
    reasoning_tasks = []
    word_order_tasks = []

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            dialect = row.get("dialect", "o")
            lampung = row.get("lampung", "").strip()
            indo = row.get("indonesian", "").strip()
            eng = row.get("english", "").strip()

            if not lampung or not indo:
                continue

            dialect_name = "Lampung O" if dialect == "o" else "Lampung Nyo"
            row_type = row.get("type", "")

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

            if row_type != "dictionary":
                add_word_order_task(word_order_tasks, dialect_name, lampung)
                add_word_order_task(word_order_tasks, "Bahasa Indonesia", indo)

            reasoning_tasks.append({
                "instruction": f"Terjemahkan {dialect_name} ke Bahasa Indonesia dan jelaskan kata per kata",
                "input": lampung,
                "reasoning": make_reasoning(dialect_name, "Bahasa Indonesia", lampung, indo),
                "output": indo,
            })

            reasoning_tasks.append({
                "instruction": f"Terjemahkan Bahasa Indonesia ke {dialect_name} dan jelaskan kata per kata",
                "input": indo,
                "reasoning": make_reasoning("Bahasa Indonesia", dialect_name, indo, lampung),
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

                reasoning_tasks.append({
                    "instruction": f"Translate {dialect_name} to English and explain word by word",
                    "input": lampung,
                    "reasoning": make_reasoning(dialect_name, "English", lampung, eng),
                    "output": eng,
                })

                reasoning_tasks.append({
                    "instruction": f"Translate English to {dialect_name} and explain word by word",
                    "input": eng,
                    "reasoning": make_reasoning("English", dialect_name, eng, lampung),
                    "output": lampung,
                })

                if row_type != "dictionary":
                    add_word_order_task(word_order_tasks, "English", eng)

    with output_path.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    with reasoning_output_path.open("w", encoding="utf-8") as f:
        for task in reasoning_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    with word_order_output_path.open("w", encoding="utf-8") as f:
        for task in word_order_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    augmented_tasks = tasks + reasoning_tasks + word_order_tasks
    random.Random(split).shuffle(augmented_tasks)

    with augmented_output_path.open("w", encoding="utf-8") as f:
        for task in augmented_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"{split}: {len(tasks)} rows saved to {output_path}")
    print(f"{split}: {len(reasoning_tasks)} rows saved to {reasoning_output_path}")
    print(f"{split}: {len(word_order_tasks)} rows saved to {word_order_output_path}")
    print(f"{split}: {len(augmented_tasks)} rows saved to {augmented_output_path}")

for split in ["train", "valid", "test"]:
    build(split)
