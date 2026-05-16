# training/dataset.py
import torch
from torch.utils.data import Dataset
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

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Input: token[0..n-1], Target: token[1..n] (next token prediction)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:],  dtype=torch.long)
        return x, y