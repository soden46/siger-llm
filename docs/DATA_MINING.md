# DATA_MINING.md - Mining Dataset Q&A dan Laravel untuk SigerLM

Dokumen ini menjelaskan cara mengubah dataset Q&A Bahasa Indonesia, instruction dataset, dokumentasi Laravel, dan tutorial Laravel menjadi format yang bisa dibaca pipeline SigerLM.

Output akhirnya adalah JSONL instruction row:

```json
{"instruction":"...","input":"...","output":"...","system":"...","source":"...","type":"..."}
```

Format ini bisa langsung diproses oleh:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json
```

## 1. Sumber Dataset

Q&A Bahasa Indonesia:

- `SEACrowd/indoqa`
- `SEACrowd/tydiqa_id`
- `Wikidepia/indonesia_dataset`
- `indonlp/indonlu`
- Kaggle `firqaaa/indonewsqa`

General instruction:

- `togethercomputer/MoAA-SFT`
- `FreedomIntelligence/alpaca-gpt4-indonesian`
- `FreedomIntelligence/evol-instruct-indonesian`
- `Iftitahu/indonesian_instruct_stories`

Laravel:

- Laravel official docs `9.x`, `10.x`, `11.x`, `12.x`, `13.x`
- SantriKoding tag Laravel, difilter hanya artikel yang menyebut Laravel 9 sampai 13

Catatan: link `fchis/Laravel-13x-Qwen2.5-Coder-3B-Instruct-LoRA` dan `yannelli/Laravel-11-Llama-3.2-1B-Instruct-GGUF` adalah model/checkpoint, bukan dataset mentah. Jangan dilatih ulang dari bobot model itu. Untuk data Laravel, pakai official docs, tutorial, dan dataset instruction seperti MoAA-SFT yang punya contoh Laravel.

Indonesian HF mix untuk eksperimen Kaggle:

- `indonesian-nlp/wikipedia-id` sebagai raw text / text completion
- `Lyon28/Corpus-Indonesia` sebagai raw text / text completion
- `Hemgg/indonesian2english-dataset` sebagai translation Indonesia-Inggris
- `hndrbrm/indonesia_vocabulary` sebagai vocabulary
- `abid/indonesia-medical-qna` sebagai medical Q&A
- `morissu/indonesian_corpus` sebagai raw text / text completion
- `IndonesiaAI/translated-samples` sebagai translation
- `kaitchup/opus-Indonesian-to-English` sebagai translation
- `Ichsan2895/alpaca-gpt4-indonesian` sebagai instruction
- `AnnasBlackHat/alpaca-indonesia-llama` sebagai instruction
- `ermandmand/indonesian-simple-instruction-dataset` sebagai instruction sederhana
- `IndonesiaAI/sft-dataset` sebagai SFT/instruction
- `audichandra/bitext_customer_support_llm_dataset_indonesian` sebagai customer support instruction
- `LorthGyu/indonesian-qa` sebagai Q&A Bahasa Indonesia
- `theonlydo/indonesia-slang` sebagai slang/vocabulary
- `nahiar/indonesia-slang` sebagai slang/vocabulary
- `arihadilahhasan/quran_terjemah_indonesia` sebagai raw text / text completion

Catatan: `arihadilahhasan/quran_terjemah_indonesia` pernah muncul dobel di daftar eksperimen, tetapi cukup dimining satu kali.

## 2. Install Dependency Tambahan

Dependency utama sudah ada di `requirements.txt`: `datasets`, `requests`, dan `beautifulsoup4`.

Untuk dataset SEACrowd tertentu, HuggingFace membutuhkan loader tambahan:

```powershell
pip install seacrowd nusacrowd
```

Untuk Kaggle, download dataset dari UI Kaggle atau Kaggle CLI, lalu konversi file JSON/JSONL/CSV lokal dengan `--local-qa-file`.

## 3. Mining Q&A Indonesia

```powershell
python tools\mine_general_assistant_data.py --preset qa
```

Output:

```txt
data/mined/instruction/indonesian_qa_instruction.jsonl
```

Format prompt yang dibuat:

```txt
instruction: Jawab pertanyaan berdasarkan konteks berikut.
input      : Konteks: ...
             Pertanyaan: ...
output     : jawaban
```

Untuk Kaggle IndoNewsQA:

```powershell
python tools\mine_general_assistant_data.py --preset qa --local-qa-file data\external\indonewsqa\train.jsonl
```

File lokal bisa `.jsonl`, `.json`, atau `.csv`, selama punya kolom seperti `question`, `context`, dan `answer`.

## 4. Mining General Instruction

```powershell
python tools\mine_general_assistant_data.py --preset instruction
```

Output:

```txt
data/mined/instruction/indonesian_general_instruction.jsonl
```

Tool akan membaca field umum seperti:

- `instruction`
- `prompt`
- `question`
- `output`
- `response`
- `answer`
- `messages`
- `conversations`

## 5. Mining Laravel Docs dan SantriKoding

```powershell
python tools\mine_general_assistant_data.py --preset laravel
```

Default versi:

```txt
9 10 11 12 13
```

Output:

```txt
data/mined/instruction/laravel_instruction.jsonl
```

Untuk membatasi request saat uji coba:

```powershell
python tools\mine_general_assistant_data.py --preset laravel --max-laravel-pages 5 --max-santrikoding-articles 10 --delay 1
```

Untuk versi tertentu:

```powershell
python tools\mine_general_assistant_data.py --preset laravel --laravel-versions 9 10 11 12 13
```

SantriKoding difilter dengan marker versi seperti `Laravel 9`, `Laravel 10`, `Laravel 11`, `Laravel 12`, atau `Laravel 13`. Artikel yang tidak jelas versinya tidak dimasukkan.

## 6. Mining Semua Sumber

Untuk eksperimen penuh:

```powershell
python tools\mine_general_assistant_data.py --preset all
```

Untuk smoke test kecil dulu:

```powershell
python tools\mine_general_assistant_data.py --preset all --max-items 200 --max-laravel-pages 5 --max-santrikoding-articles 10
```

Report:

```txt
data/mined/instruction/mining_report.json
```

## 6.1 Mining Indonesian HF Mix untuk Kaggle

Gunakan tool khusus ini untuk mengambil dataset HF Indonesia campuran secara streaming dan mengubahnya ke format instruction SigerLM:

```powershell
python tools\mine_indonesian_hf_mix.py --max-items-per-source 50000
```

Output utama:

```txt
data/mined/hf_indonesia/indonesian_hf_mix_instruction.jsonl
data/indonesian_hf_mix.txt
data/mined/hf_indonesia/hf_mix_report.json
```

`indonesian_hf_mix_instruction.jsonl` dipakai untuk LoRA/instruction tuning. `data/indonesian_hf_mix.txt` bisa ikut terbaca oleh `main.py` untuk base pretraining karena `main.py` membaca file `.txt` di folder `data/`.

Untuk smoke test kecil:

```powershell
python tools\mine_indonesian_hf_mix.py --max-items-per-source 200
```

Untuk menambahkan source custom:

```powershell
python tools\mine_indonesian_hf_mix.py --source nama/dataset:instruction:train
```

Format `--source`:

```txt
dataset:kind[:split[:config[:max_items]]]
```

Kind yang didukung:

```txt
text
instruction
qa
translation
vocab
```

Kalau salah satu dataset gagal load karena schema, akses, atau dependency HF, tool akan mencatat error di report dan lanjut ke dataset berikutnya.

## 7. Build Corpus agar Bisa Dibaca SigerLM

Setelah file mining tersedia:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json
```

Output corpus:

```txt
data/corpus/general_assistant_mining_train.jsonl
```

Untuk LoRA general assistant, arahkan config training ke corpus ini atau tambahkan sources mining ke `configs/datasets/general_instruction.json` setelah file mining stabil.

Untuk Indonesian HF mix:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix.json
```

Output corpus:

```txt
data/corpus/indonesian_hf_mix_train.jsonl
```

## 8. Rekomendasi Mixing

Untuk tahap awal general assistant:

```txt
Q&A Indonesia          weight 2
Laravel docs/tutorial  weight 2
General instruction    weight 1
Lampung domain         weight 1
```

Alasannya:

- Q&A melatih model menjawab berdasarkan konteks.
- Laravel docs/tutorial memberi kemampuan domain coding yang konkret.
- General instruction menjaga gaya assistant umum.
- Lampung tetap ada sebagai kemampuan lokal, tetapi tidak mendominasi general assistant.

## 9. Quality Gate

Sebelum training serius:

1. Cek sampel JSONL manual.
2. Hapus data yang terlalu panjang, kosong, atau noisy.
3. Pastikan jawaban Q&A tidak kehilangan konteks.
4. Pisahkan eval set kecil untuk Q&A dan Laravel.
5. Jangan mencampur data berlisensi ketat ke release publik tanpa review lisensi.

Contoh inspeksi cepat:

```powershell
Get-Content data\mined\instruction\indonesian_qa_instruction.jsonl -TotalCount 3
Get-Content data\mined\instruction\laravel_instruction.jsonl -TotalCount 3
```

Compile check:

```powershell
python -m py_compile tools\mine_general_assistant_data.py training\dataset_registry.py tools\build_instruction_corpus.py
python -m py_compile tools\mine_indonesian_hf_mix.py
```
