# optimization/quantization/calibrate.py
"""
Calibration dataset untuk INT8 Static Quantization.

Static quantization butuh "calibration" — jalanin beberapa batch
representatif supaya observer bisa catat range activation tiap layer.
Range ini dipakai untuk menentukan scale factor quantization.

Makin representatif calibration data = makin akurat quantization.
Biasanya 100-200 batch sudah cukup.
"""
import torch
from torch.utils.data import DataLoader, Dataset
from typing import List, Optional
from pathlib import Path


class CalibrationDataset(Dataset):
    """
    Dataset kecil khusus untuk calibration quantization.
    Berisi teks representatif dari domain yang akan diinference.
    """

    # Teks default — campuran Indo, EN, dan code
    DEFAULT_TEXTS = [
        # Bahasa Indonesia
        "Pemerintah Indonesia mengumumkan kebijakan baru terkait pendidikan.",
        "Teknologi kecerdasan buatan berkembang pesat dalam beberapa tahun terakhir.",
        "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa bagian barat.",
        "Sistem kesehatan nasional perlu diperkuat untuk menghadapi tantangan masa depan.",
        "Ekonomi digital Indonesia terus tumbuh dengan pesat seiring adopsi teknologi.",
        "Para ilmuwan berhasil mengembangkan vaksin baru yang efektif melawan penyakit.",
        "Infrastruktur transportasi publik di kota besar perlu segera ditingkatkan.",
        "Pendidikan berkualitas adalah kunci untuk menciptakan generasi muda yang kompetitif.",

        # English
        "The development of large language models has revolutionized natural language processing.",
        "Climate change remains one of the most pressing challenges facing humanity today.",
        "Advances in quantum computing may soon break current encryption standards.",
        "The global economy is undergoing significant transformation due to digitalization.",
        "Machine learning algorithms can now detect diseases from medical imaging with high accuracy.",
        "Renewable energy sources are becoming increasingly cost-competitive with fossil fuels.",
        "Space exploration has entered a new era with private companies leading missions.",
        "The human brain contains approximately 86 billion neurons forming complex networks.",

        # Code
        "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
        "class DataProcessor:\n    def __init__(self, data):\n        self.data = data\n    def process(self):\n        return [x * 2 for x in self.data]",
        "SELECT users.name, orders.total FROM users JOIN orders ON users.id = orders.user_id WHERE orders.status = 'completed';",
        "import pandas as pd\ndf = pd.read_csv('data.csv')\ndf_clean = df.dropna().reset_index(drop=True)\nprint(df_clean.describe())",

        # Mixed / conversational
        "Apa perbedaan antara machine learning dan deep learning dalam konteks AI modern?",
        "What are the best practices for securing a REST API in a production environment?",
        "Bagaimana cara mengoptimalkan query database untuk meningkatkan performa aplikasi?",
        "Explain the concept of gradient descent in simple terms for a beginner programmer.",
    ]

    def __init__(
        self,
        tokenizer,
        texts: Optional[List[str]] = None,
        max_seq_len: int = 256,
        repeat: int = 4,   # repeat dataset supaya dapat lebih banyak batch
    ):
        self.tokenizer   = tokenizer
        self.max_seq_len = max_seq_len
        texts            = texts or self.DEFAULT_TEXTS

        # Tokenize semua teks
        self.samples = []
        for text in texts * repeat:
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            if len(ids) < 4:
                continue

            # Truncate ke max_seq_len
            ids = ids[:max_seq_len]

            # Pad ke max_seq_len
            pad_len = max_seq_len - len(ids)
            ids     = ids + [tokenizer.pad_id] * pad_len

            self.samples.append(torch.tensor(ids, dtype=torch.long))

        print(f"📊 Calibration dataset: {len(self.samples)} samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def build_calibration_loader(
    tokenizer,
    texts: Optional[List[str]] = None,
    max_seq_len: int = 256,
    batch_size: int  = 4,
    n_batches: int   = 100,
) -> DataLoader:
    """
    Build DataLoader untuk calibration.

    Args:
        tokenizer   : MultilingualTokenizer
        texts       : teks representatif (None = pakai default)
        max_seq_len : panjang sequence per sample
        batch_size  : batch size
        n_batches   : jumlah batch yang diproses (100 sudah cukup)

    Returns:
        DataLoader siap dipakai di quantize_int8_static()
    """
    dataset = CalibrationDataset(
        tokenizer   = tokenizer,
        texts       = texts,
        max_seq_len = max_seq_len,
        repeat      = max(1, (n_batches * batch_size) // max(len(texts or CalibrationDataset.DEFAULT_TEXTS), 1) + 1),
    )

    loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,   # 0 untuk CPU quantization
        drop_last   = True,
    )

    return loader


def load_calibration_from_file(
    path: str,
    tokenizer,
    max_seq_len: int = 256,
    batch_size: int  = 4,
    max_lines: int   = 500,
) -> DataLoader:
    """
    Load calibration data dari file teks.
    Setiap baris = satu sample.

    Args:
        path        : path ke file .txt
        tokenizer   : MultilingualTokenizer
        max_seq_len : panjang sequence per sample
        batch_size  : batch size
        max_lines   : maksimum baris yang diload

    Returns:
        DataLoader siap dipakai
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file tidak ditemukan: {path}")

    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if len(line) > 20:
                texts.append(line)

    print(f"📂 Loaded {len(texts)} calibration samples dari {path}")

    return build_calibration_loader(
        tokenizer   = tokenizer,
        texts       = texts,
        max_seq_len = max_seq_len,
        batch_size  = batch_size,
    )


def run_calibration(model, loader: DataLoader, n_batches: int = 100):
    """
    Jalankan calibration loop pada model yang sudah di-prepare.
    Model harus sudah melewati torch.quantization.prepare() sebelumnya.

    Args:
        model     : model yang sudah di-prepare untuk static quantization
        loader    : DataLoader dari build_calibration_loader()
        n_batches : jumlah batch untuk calibration
    """
    model.eval()
    print(f"🔍 Running calibration ({n_batches} batches)...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break

            # batch shape: (B, seq_len)
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            model(x)

            if (i + 1) % 10 == 0:
                print(f"  Calibration batch {i+1}/{n_batches}")

    print(f"✅ Calibration selesai ({min(i+1, n_batches)} batches diproses)")