# tokenizer/tests/sample_texts.py
"""
Kumpulan sample teks untuk testing tokenizer.
Mencakup: Bahasa Indonesia, English, Bahasa Lampung Dialek O,
          kode Python/PHP, dan teks campuran.
"""

# ── Bahasa Indonesia ──────────────────────────────────────────
INDONESIAN_TEXTS = [
    "Halo, apa kabar? Saya sedang belajar membuat LLM dari nol.",
    "Pemerintah Indonesia mengumumkan kebijakan baru terkait pendidikan nasional.",
    "Teknologi kecerdasan buatan berkembang pesat dalam beberapa tahun terakhir.",
    "Jakarta adalah ibu kota Indonesia yang terletak di Pulau Jawa bagian barat.",
    "Sistem kesehatan nasional perlu diperkuat untuk menghadapi tantangan masa depan.",
    "Ekonomi digital Indonesia terus tumbuh dengan pesat seiring adopsi teknologi.",
    "Para ilmuwan berhasil mengembangkan vaksin baru yang efektif melawan penyakit menular.",
    "Infrastruktur transportasi publik di kota-kota besar perlu segera ditingkatkan kualitasnya.",
    "Pendidikan berkualitas adalah kunci untuk menciptakan generasi muda yang kompetitif di era global.",
    "Bahasa Indonesia adalah bahasa persatuan yang menghubungkan lebih dari 270 juta penduduk.",
]

# ── English ───────────────────────────────────────────────────
ENGLISH_TEXTS = [
    "Hello! I'm building a language model from scratch using PyTorch.",
    "The development of large language models has revolutionized natural language processing.",
    "Climate change remains one of the most pressing challenges facing humanity today.",
    "Advances in quantum computing may soon break current encryption standards.",
    "Machine learning algorithms can now detect diseases from medical imaging with high accuracy.",
    "Renewable energy sources are becoming increasingly cost-competitive with fossil fuels.",
    "Space exploration has entered a new era with private companies leading ambitious missions.",
    "The quick brown fox jumps over the lazy dog near the river bank at sunset.",
    "Deep neural networks learn hierarchical representations of data through multiple layers.",
    "Transformer architecture revolutionized NLP by enabling parallel sequence processing.",
]

# ── Bahasa Lampung Dialek O ───────────────────────────────────
LAMPUNG_O_TEXTS = [
    "Nyak haga mengan di pasar.",
    "Api kabar niku sekeluarga?",
    "Jak lapah temon ke pekon.",
    "Niku munih haga minom kopi?",
    "Nyak kona sai haga maju.",
    "Api gawoh niku pok?",
    "Lapah majeu, munih lapah megou.",
    "Temon niku sai jak tunggu.",
    "Nyak haga lapah mit Jakarta.",
    "Sanak-sanak jak lapah sekulah.",
]

# ── Pasangan Terjemahan Lampung O ↔ Indonesia ─────────────────
LAMPUNG_TRANSLATION_PAIRS = [
    ("nyak haga mengan", "saya mau makan"),
    ("api kabar niku", "apa kabar kamu"),
    ("jak lapah temon", "kita pergi juga"),
    ("niku haga lapah pok", "kamu mau pergi ke mana"),
    ("nyak kona sai", "saya yang punya"),
    ("temon niku", "kamu juga"),
    ("lapah majeu", "pergi ke sana"),
    ("mit Jakarta", "ke Jakarta"),
    ("sanak-sanak", "anak-anak"),
    ("mengan munih minom", "makan juga minum"),
]

# ── Kode Python ───────────────────────────────────────────────
PYTHON_CODE_TEXTS = [
    "def calculate_fibonacci(n):\n    if n <= 1:\n        return n\n    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)",
    "import torch\nimport torch.nn as nn\n\nclass SimpleModel(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(128, 64)",
    "for i in range(10):\n    print(f'Step {i}: loss = {loss:.4f}')",
    "df = pd.read_csv('data.csv')\ndf_clean = df.dropna().reset_index(drop=True)\nprint(df_clean.describe())",
    "with torch.no_grad():\n    logits = model(input_ids)\n    probs = F.softmax(logits, dim=-1)",
]

# ── Kode PHP/Laravel ──────────────────────────────────────────
PHP_CODE_TEXTS = [
    "<?php\nnamespace App\\Http\\Controllers;\n\nclass UserController extends Controller\n{\n    public function index()\n    {\n        return User::all();\n    }\n}",
    "Route::get('/users', [UserController::class, 'index'])->middleware('auth');",
    "$users = User::where('active', true)->orderBy('created_at', 'desc')->paginate(15);",
    "return response()->json(['status' => 'success', 'data' => $result], 200);",
    "Schema::create('users', function (Blueprint $table) {\n    $table->id();\n    $table->string('name');\n    $table->string('email')->unique();\n    $table->timestamps();\n});",
]

# ── Teks Campuran (Code-Switching) ────────────────────────────
MIXED_TEXTS = [
    "Gua lagi debug kode Laravel tapi ada error di migration.",
    "The model perlu di-fine-tune buat bahasa Lampung dulu.",
    "Training loss udah turun ke 1.5, lumayan bagus buat model sekecil ini.",
    "Deploy ke VPS DewaCloud yang 2 core 4GB RAM, semoga cukup.",
    "Nyak haga belajar machine learning supaya bisa bikin AI untuk bahasa Lampung.",
]

# ── Instruksi Format (untuk LoRA testing) ─────────────────────
INSTRUCTION_SAMPLES = [
    {
        "system": "Kamu adalah asisten penerjemah Bahasa Lampung yang akurat.",
        "user": "Terjemahkan ke Bahasa Indonesia: nyak haga mengan",
        "assistant": "saya mau makan",
    },
    {
        "system": "Kamu adalah asisten penerjemah Bahasa Lampung yang akurat.",
        "user": "Terjemahkan ke Bahasa Indonesia: api kabar niku?",
        "assistant": "apa kabar kamu?",
    },
    {
        "system": "Kamu adalah asisten AI yang helpful.",
        "user": "Apa ibu kota Indonesia?",
        "assistant": "Ibu kota Indonesia adalah Jakarta.",
    },
    {
        "system": "Kamu adalah asisten coding yang ahli Python dan Laravel.",
        "user": "Bagaimana cara membuat route di Laravel?",
        "assistant": "Di Laravel, route didefinisikan di file routes/web.php menggunakan facade Route. Contoh: Route::get('/users', [UserController::class, 'index']);",
    },
]

# ── Semua teks digabung untuk testing menyeluruh ──────────────
ALL_TEXTS = (
    INDONESIAN_TEXTS +
    ENGLISH_TEXTS +
    LAMPUNG_O_TEXTS +
    PYTHON_CODE_TEXTS +
    PHP_CODE_TEXTS +
    MIXED_TEXTS
)

# ── Helper functions ──────────────────────────────────────────

def get_texts_by_lang(lang: str) -> list:
    """
    Ambil sample texts berdasarkan bahasa.

    Args:
        lang: "id", "en", "lampung", "code", "mixed", "all"

    Returns:
        List of strings
    """
    mapping = {
        "id":      INDONESIAN_TEXTS,
        "en":      ENGLISH_TEXTS,
        "lampung": LAMPUNG_O_TEXTS,
        "code":    PYTHON_CODE_TEXTS + PHP_CODE_TEXTS,
        "mixed":   MIXED_TEXTS,
        "all":     ALL_TEXTS,
    }
    return mapping.get(lang, ALL_TEXTS)


def format_instruction_sample(sample: dict, tok) -> str:
    """
    Format satu instruction sample ke format chat model.

    Args:
        sample : dict dengan key system/user/assistant
        tok    : MultilingualTokenizer instance

    Returns:
        Formatted string dengan special tokens
    """
    system    = sample.get("system", "")
    user      = sample.get("user", "")
    assistant = sample.get("assistant", "")

    return (
        f"<|system|>{system}<|end_turn|>\n"
        f"<|user|>{user}<|end_turn|>\n"
        f"<|assistant|>{assistant}<|end_turn|>"
    )