# lora/dataset.py
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset

from tokenizer.tokenizer import MultilingualTokenizer


RECOMMENDED_DATASETS = {
    "ultrachat": "HuggingFaceH4/ultrachat_200k",
    "alpaca_id": "indonlp/indonesian-alpaca",
    "oasst": "OpenAssistant/oasst2",
    "dolly": "databricks/databricks-dolly-15k",
    "flan": "Muennighoff/flan",
}


SYSTEM_PROMPT = (
    "Kamu adalah SigerLM, asisten AI umum yang cerdas, ringkas, dan akurat. "
    "Ikuti instruksi user dan jawab sesuai konteks."
)


def format_local_instruction(example: Dict) -> str:
    """
    Format dataset JSONL lokal Lampung:
    {
      "instruction": "...",
      "input": "...",
      "output": "..."
    }

    Menjadi format chat model:
    <|system|>...<|end_turn|>
    <|user|>instruction + input<|end_turn|>
    <|assistant|>output<|end_turn|>
    """
    system_prompt = str(example.get("system", SYSTEM_PROMPT)).strip() or SYSTEM_PROMPT
    instruction = str(example.get("instruction", "")).strip()
    input_text = str(example.get("input", "")).strip()
    reasoning = str(example.get("reasoning", "")).strip()
    output = str(example.get("output", "")).strip()

    if not instruction or not output:
        return ""

    if input_text:
        user_message = f"{instruction}\n\n{input_text}"
    else:
        user_message = instruction

    if reasoning:
        assistant_message = (
            f"Penjelasan:\n{reasoning}\n\n"
            f"Jawaban:\n{output}"
        )
    else:
        assistant_message = output

    return (
        f"<|system|>{system_prompt}<|end_turn|>\n"
        f"<|user|>{user_message}<|end_turn|>\n"
        f"<|assistant|>{assistant_message}<|end_turn|>"
    )


def format_hf_instruction(example: Dict, dataset_name: str) -> str:
    """
    Fallback untuk dataset HuggingFace lama:
    ultrachat / alpaca / dolly / oasst.
    """
    if "ultrachat" in dataset_name:
        messages = example.get("messages", [])
        parts = [f"<|system|>{SYSTEM_PROMPT}<|end_turn|>"]

        for msg in messages:
            role = msg.get("role", "")
            content = str(msg.get("content", "")).strip()

            if role in ("user", "assistant") and content:
                parts.append(f"<|{role}|>{content}<|end_turn|>")

        return "\n".join(parts)

    if "alpaca" in dataset_name or "dolly" in dataset_name:
        instruction = str(example.get("instruction", "")).strip()
        inp = str(example.get("input", "")).strip()
        output = str(example.get("output", "")).strip()

        if not instruction or not output:
            return ""

        user_msg = f"{instruction}\n{inp}".strip() if inp else instruction

        return (
            f"<|system|>{SYSTEM_PROMPT}<|end_turn|>\n"
            f"<|user|>{user_msg}<|end_turn|>\n"
            f"<|assistant|>{output}<|end_turn|>"
        )

    if "oasst" in dataset_name:
        role = example.get("role", "user")
        text = str(example.get("text", "")).strip()

        if not text:
            return ""

        return f"<|{role}|>{text}<|end_turn|>"

    # Fallback
    text = str(example.get("text", "")).strip()
    return text


class InstructionDataset(Dataset):
    """
    Dataset instruction tuning.

    Fitur:
    - Bisa membaca JSONL lokal Lampung
    - Bisa fallback ke HuggingFace dataset
    - Loss masking:
        system/user tokens = -100
        assistant tokens   = real token ids
    """

    IGNORE_INDEX = -100

    def __init__(
        self,
        tokenizer: MultilingualTokenizer,
        max_seq_len: int = 512,
        max_samples: Optional[int] = None,
        dataset_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        split: str = "train",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if dataset_path:
            raw_examples = self._load_local_jsonl(dataset_path)
            source_label = dataset_path
            formatter = self._format_local
        elif dataset_name:
            raw_examples = self._load_hf_dataset(dataset_name, split)
            source_label = f"{dataset_name}:{split}"
            formatter = lambda ex: format_hf_instruction(ex, dataset_name)
        else:
            raise ValueError(
                "InstructionDataset butuh dataset_path atau dataset_name."
            )

        if max_samples is not None:
            raw_examples = raw_examples[:max_samples]

        print(f"Loading instruction dataset: {source_label}")
        print(f"Raw examples: {len(raw_examples):,}")

        self.examples = self._process(raw_examples, formatter)

        print(f"Dataset ready: {len(self.examples):,} valid examples")

        if len(self.examples) == 0:
            raise RuntimeError(
                "InstructionDataset kosong setelah diproses. "
                "Cek format JSONL atau max_seq_len."
            )

    def _load_local_jsonl(self, path: str) -> List[Dict]:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Dataset lokal tidak ditemukan: {file_path}")

        rows: List[Dict] = []

        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Skip JSON invalid line {line_number}: {e}")
                    continue

                rows.append(row)

        return rows

    def _load_hf_dataset(self, dataset_name: str, split: str) -> List[Dict]:
        dataset = load_dataset(dataset_name, split=split)
        return [dict(row) for row in dataset]

    def _format_local(self, example: Dict) -> str:
        return format_local_instruction(example)

    def _process(self, raw_examples: List[Dict], formatter) -> List[Dict]:
        examples: List[Dict] = []

        for example in raw_examples:
            text = formatter(example)

            if not text or len(text) < 10:
                continue

            input_ids = self.tokenizer.encode(
                text,
                add_bos=True,
                add_eos=True,
            )

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]

            if len(input_ids) < 5:
                continue

            labels = self._build_labels(input_ids)

            # Pastikan memang ada assistant token yang dihitung loss-nya
            if all(label == self.IGNORE_INDEX for label in labels):
                continue

            examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

        return examples

    def _build_labels(self, input_ids: List[int]) -> List[int]:
        """
        Label masking:
        - Sebelum <|assistant|> => -100
        - Token assistant output => real token ids
        - Setelah <|end_turn|> assistant => -100

        Catatan alignment:
        labels menyimpan token target pada posisi aslinya. Trainer LoRA
        melakukan causal shift logits[:, :-1] vs labels[:, 1:], sehingga
        token <|assistant|> memprediksi token jawaban pertama.
        """
        labels = [self.IGNORE_INDEX] * len(input_ids)

        assistant_id = self.tokenizer.special_tokens.get("<|assistant|>")
        end_turn_id = self.tokenizer.special_tokens.get("<|end_turn|>")

        if assistant_id is None or end_turn_id is None:
            raise RuntimeError(
                "Tokenizer tidak punya <|assistant|> atau <|end_turn|>."
            )

        in_assistant_answer = False

        for i, token_id in enumerate(input_ids):
            if token_id == assistant_id:
                in_assistant_answer = True
                labels[i] = self.IGNORE_INDEX
                continue

            if token_id == end_turn_id and in_assistant_answer:
                labels[i] = token_id
                in_assistant_answer = False
                continue

            if in_assistant_answer:
                labels[i] = token_id
            else:
                labels[i] = self.IGNORE_INDEX

        return labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        return {
            "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
            "labels": torch.tensor(example["labels"], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Pad batch ke panjang sequence terbesar di batch.
    """
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        ids = item["input_ids"]
        lbl = item["labels"]

        seq_len = ids.size(0)
        pad_len = max_len - seq_len

        padded_ids = torch.cat([
            ids,
            torch.full((pad_len,), pad_id, dtype=torch.long),
        ])

        padded_labels = torch.cat([
            lbl,
            torch.full((pad_len,), -100, dtype=torch.long),
        ])

        attention_mask = torch.cat([
            torch.ones(seq_len, dtype=torch.float32),
            torch.zeros(pad_len, dtype=torch.float32),
        ])

        input_ids.append(padded_ids)
        labels.append(padded_labels)
        attention_masks.append(attention_mask)

    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_masks),
    }
