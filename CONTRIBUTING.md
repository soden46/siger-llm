# Contributing to SIGER LLM

Terima kasih sudah tertarik berkontribusi ke **SIGER LLM**.

SIGER adalah eksperimen general-purpose language model yang dibangun dari scratch menggunakan arsitektur State Space Model/SSM, lengkap dengan pipeline training, LoRA fine-tuning, evaluasi, optimasi, dan eksperimen low-resource language.

Saat ini, Bahasa Lampung Dialek O digunakan sebagai salah satu objek eksperimen awal untuk pengembangan dataset translasi.

---

## Cara Berkontribusi

Kontribusi yang diterima antara lain:

- Perbaikan bug
- Peningkatan dokumentasi
- Penambahan evaluasi atau benchmark
- Optimasi inference/training
- Perbaikan arsitektur model
- Penambahan dataset yang legal dan relevan
- Validasi dataset Bahasa Lampung Dialek O
- Penambahan contoh percakapan atau parallel sentence yang sudah ditinjau

---

## Sebelum Membuat Pull Request

Sebelum mengirim PR:

1. Pastikan branch sudah up to date dengan `main`
2. Jalankan test atau smoke test yang relevan
3. Pastikan tidak ada file besar, checkpoint, dataset mentah, atau secret yang ikut ter-commit
4. Jelaskan perubahan secara ringkas dan jelas
5. Sertakan contoh output jika perubahan memengaruhi training, scraping, dataset, atau inference

---

## Setup Lokal

```bash
git clone https://github.com/soden46/siger-llm.git
cd siger-llm

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

Menjalankan Smoke Test
python main.py

Untuk LoRA Lampung:

python -m lora.run_lora
Kontribusi Dataset Lampung Dialek O

Dataset yang sangat dibutuhkan:

Percakapan sehari-hari
Sapaan dan perkenalan
Aktivitas rumah dan sekolah
Dialog pasar
Cerita pendek
Ungkapan budaya
Parallel sentence Lampung O ↔ Indonesia
Terjemahan English opsional

Format JSONL yang disarankan:

{
  "dialect": "o",
  "lampung": "api kabar niku?",
  "indonesian": "apa kabar kamu?",
  "english": "how are you?",
  "source": "manual_native_review",
  "type": "daily_conversation"
}

Mohon pastikan:

Data tidak diambil dari sumber yang melarang penggunaan ulang
Sumber dicantumkan bila berasal dari publikasi/artikel
Data sebisa mungkin divalidasi penutur atau reviewer yang memahami Dialek O
Gaya Commit

Gunakan commit message yang jelas. Contoh:

feat: add Lampung conversation dataset parser
fix: handle empty instruction dataset after tokenization
docs: update SIGER architecture overview
refactor: rename Mamba references to SIGER naming
Membuat Issue

Gunakan issue template yang tersedia:

Bug Report
Feature Request

Jelaskan konteks, langkah reproduksi, dan ekspektasi hasil dengan lengkap.

Code Review

Maintainer berhak:

meminta revisi
menolak perubahan yang tidak sesuai scope proyek
meminta bukti sumber dataset
meminta benchmark tambahan untuk perubahan performa

Terima kasih sudah membantu mengembangkan SIGER LLM.