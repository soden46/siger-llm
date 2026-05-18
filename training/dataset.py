# training/dataset.py
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tokenizer.tokenizer import MultilingualTokenizer

class TextDataset(Dataset):
    """
    Sliding window dataset untuk language modeling.
    Mirip kayak chunking teks di Laravel pagination — potong-potong.
    """
    def __init__(self, texts: list[str], tokenizer: MultilingualTokenizer,
                 max_seq_len: int = 2048, stride: int = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.stride = stride or max_seq_len  # non-overlapping default

        # Tokenize semua teks, gabung jadi satu stream
        all_ids = []
        for text in texts:
            ids = tokenizer.encode(text, add_eos=True)
            all_ids.extend(ids)

        # Sliding window: potong jadi chunk max_seq_len + 1
        # +1 karena target = input shifted by 1
        self.chunks = []
        for i in range(0, len(all_ids) - max_seq_len, self.stride):
            chunk = all_ids[i : i + max_seq_len + 1]
            self.chunks.append(chunk)

        print(f"📚 Dataset: {len(texts)} docs → {len(self.chunks)} chunks")

    @classmethod
    def from_text_files(
        cls,
        paths: list[str | Path],
        tokenizer: MultilingualTokenizer,
        max_seq_len: int = 2048,
        stride: int = None,
        max_chars_per_file: int | None = 8_000_000,
        max_chunks: int | None = None,
    ) -> "TextDataset":
        dataset = cls.__new__(cls)
        dataset.tokenizer = tokenizer
        dataset.max_seq_len = max_seq_len
        dataset.stride = stride or max_seq_len
        dataset.chunks = []

        file_count = 0
        token_buffer: list[int] = []
        max_window = max_seq_len + 1

        def flush_chunks(final: bool = False) -> None:
            nonlocal token_buffer
            limit = len(token_buffer) - max_seq_len
            i = 0
            while i < limit:
                dataset.chunks.append(token_buffer[i : i + max_window])
                if max_chunks is not None and len(dataset.chunks) >= max_chunks:
                    token_buffer = []
                    return
                i += dataset.stride
            token_buffer = [] if final else token_buffer[max(0, i):]

        for raw_path in paths:
            if max_chunks is not None and len(dataset.chunks) >= max_chunks:
                break

            path = Path(raw_path)
            if not path.exists() or not path.is_file():
                continue

            file_count += 1
            chars_read = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if max_chars_per_file is not None and chars_read >= max_chars_per_file:
                        break
                    line = line.strip()
                    if not line:
                        continue

                    if max_chars_per_file is not None:
                        remaining = max_chars_per_file - chars_read
                        line = line[:remaining]
                    chars_read += len(line)

                    token_buffer.extend(tokenizer.encode(line, add_eos=True))
                    if len(token_buffer) >= max_window + dataset.stride * 128:
                        flush_chunks()
                    if max_chunks is not None and len(dataset.chunks) >= max_chunks:
                        break

            flush_chunks()

        flush_chunks(final=True)
        print(f"Dataset: {file_count} files -> {len(dataset.chunks)} chunks")
        return dataset

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Input: token[0..n-1], Target: token[1..n] (next token prediction)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y
