# lora/dataset.py
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tokenizer.tokenizer import MultilingualTokenizer
from typing import List, Dict


# ── Dataset yang recommended buat instruction following ───
RECOMMENDED_DATASETS = {
    "ultrachat":    "HuggingFaceH4/ultrachat_200k",       # 200k multi-turn chat
    "alpaca_id":    "indonlp/indonesian-alpaca",            # Indo instruction
    "oasst":        "OpenAssistant/oasst2",                 # multilingual RLHF
    "dolly":        "databricks/databricks-dolly-15k",      # 15k diverse tasks
    "flan":         "Muennighoff/flan",                     # 1.8M instruction
}


def format_instruction(example: Dict, dataset_name: str) -> str:
    """
    Convert berbagai format dataset → format chat model lo.

    Format output:
    <|system|>...<|end_turn|>
    <|user|>...<|end_turn|>
    <|assistant|>...<|end_turn|>
    """
    if "ultrachat" in dataset_name:
        # Format: {"messages": [{"role": ..., "content": ...}]}
        messages = example.get("messages", [])
        parts    = ["<|system|>Kamu adalah asisten yang helpful.<|end_turn|>"]
        for msg in messages:
            role    = msg["role"]
            content = msg["content"].strip()
            if role in ("user", "assistant"):
                parts.append(f"<|{role}|>{content}<|end_turn|>")
        return "\n".join(parts)

    elif "alpaca" in dataset_name or "dolly" in dataset_name:
        # Format: {"instruction": ..., "input": ..., "output": ...}
        instruction = example.get("instruction", "").strip()
        inp         = example.get("input", "").strip()
        output      = example.get("output", "").strip()
        user_msg    = f"{instruction}\n{inp}".strip() if inp else instruction
        return (
            f"<|system|>Kamu adalah asisten yang helpful.<|end_turn|>\n"
            f"<|user|>{user_msg}<|end_turn|>\n"
            f"<|assistant|>{output}<|end_turn|>"
        )

    elif "oasst" in dataset_name:
        # Format: {"role": ..., "text": ...}
        role = example.get("role", "user")
        text = example.get("text", "").strip()
        return f"<|{role}|>{text}<|end_turn|>"

    else:
        # Fallback: treat sebagai plain text
        return example.get("text", str(example))


class InstructionDataset(Dataset):
    """
    Dataset untuk instruction fine-tuning.
    
    Loss mask: hanya hitung loss di bagian ASSISTANT,
    bukan di system/user prompt — ini kunci supaya model
    belajar nge-generate response, bukan ngulang prompt.
    """

    IGNORE_INDEX = -100   # index ini di-ignore sama CrossEntropyLoss

    def __init__(
        self,
        dataset_name: str,
        tokenizer: MultilingualTokenizer,
        split: str      = "train",
        max_seq_len: int = 512,
        max_samples: int = 50_000,
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len

        print(f"📥 Loading dataset: {dataset_name}")
        raw = load_dataset(dataset_name, split=split, streaming=False)

        if max_samples and len(raw) > max_samples:
            raw = raw.select(range(max_samples))

        print(f"📊 Processing {len(raw):,} examples...")
        self.examples = self._process(raw, dataset_name)
        print(f"✅ Dataset ready: {len(self.examples):,} valid examples")

    def _process(self, raw_dataset, dataset_name: str) -> List[Dict]:
        examples = []

        for example in raw_dataset:
            text = format_instruction(example, dataset_name)
            if not text or len(text) < 20:
                continue

            # Tokenize full text
            input_ids = self.tokenizer.encode(
                text, add_bos=True, add_eos=True
            )

            if len(input_ids) > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]

            if len(input_ids) < 10:
                continue

            # Build loss mask
            labels = self._build_labels(input_ids, text)

            examples.append({
                "input_ids": input_ids,
                "labels":    labels,
            })

        return examples

    def _build_labels(self, input_ids: List[int], text: str) -> List[int]:
        """
        Buat labels dengan masking:
        - system / user tokens → IGNORE_INDEX (-100)
        - assistant tokens     → actual token id (dihitung loss-nya)

        Ini yang bikin model fokus belajar ngejawab,
        bukan belajar nge-repeat pertanyaan.
        """
        labels = list(input_ids)   # copy

        tok = self.tokenizer
        assistant_id = tok.special_tokens.get("<|assistant|>")
        end_turn_id  = tok.special_tokens.get("<|end_turn|>")

        in_assistant = False
        for i, token_id in enumerate(input_ids):
            if token_id == assistant_id:
                in_assistant = True
                labels[i] = self.IGNORE_INDEX   # mask token <|assistant|> itu sendiri
                continue
            if token_id == end_turn_id and in_assistant:
                in_assistant = False

            if not in_assistant:
                labels[i] = self.IGNORE_INDEX   # mask system & user

        return labels

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "labels":    torch.tensor(ex["labels"],    dtype=torch.long),
        }


def collate_fn(batch: List[Dict], pad_id: int) -> Dict[str, torch.Tensor]:
    """Pad batch ke length yang sama."""
    max_len    = max(b["input_ids"].size(0) for b in batch)
    input_ids  = []
    labels     = []
    attn_masks = []

    for b in batch:
        seq_len = b["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([b["input_ids"],
                       torch.full((pad_len,), pad_id, dtype=torch.long)])
        )
        labels.append(
            torch.cat([b["labels"],
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )
        attn_masks.append(
            torch.cat([torch.ones(seq_len), torch.zeros(pad_len)])
        )

    return {
        "input_ids":      torch.stack(input_ids),
        "labels":         torch.stack(labels),
        "attention_mask": torch.stack(attn_masks),
    }