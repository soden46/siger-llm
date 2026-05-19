# DATA_MINING.md - Mining Dataset Q&A, Code, dan Laravel untuk SigerLM

Dokumen ini menjelaskan cara mengubah dataset Q&A Bahasa Indonesia, instruction dataset, code evaluation, dokumentasi Laravel, dan tutorial Laravel menjadi format yang bisa dibaca pipeline SigerLM.

Output akhirnya adalah JSONL instruction row:

```json
{"instruction":"...","input":"...","output":"...","system":"...","source":"...","type":"..."}
```

Semua tool mining harus menormalisasi output ke schema ini. Field tambahan dari dataset asal tidak dibawa ke corpus final kecuali sengaja ditambahkan ke metadata di masa depan.

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
- `Iftitahu/indonesian_instruct_stories` jika akses gated disetujui
- `QuixiAI/dolphin` (`flan1m-alpaca-uncensored`, Apache 2.0)

Code / HumanEval:

- `openai/openai_humaneval`
- `loubnabnl/humaneval_infilling`

Semua source code umum digabung ke:

```txt
data/mined/instruction/code_instruction.jsonl
```

Commercial-safe reasoning / synthetic education:

- `microsoft/orca-math-word-problems-200k` (MIT)
- `openbmb/UltraInteract_pair` (MIT)
- `HuggingFaceTB/cosmopedia` (`stories`, Apache 2.0)
- `xTayyub/High-Quality-Synthetic-Python-Dataset-with-Reasoning-Traces-Chain-of-Thought-for-LLM-Fine-Tuning` (Apache 2.0)

Semua source reasoning komersial-safe digabung ke:

```txt
data/mined/instruction/commercial_safe_reasoning_instruction.jsonl
```

Tidak dimasukkan karena provenance/distillation masih abu-abu untuk target profit:

- `AlicanKiraz0/Agentic-Chain-of-Thought-Coding-SFT-Dataset-v1.1`
- `ssbuild/alaca_chain-of-thought`

Laravel:

- Laravel official docs `9.x`, `10.x`, `11.x`, `12.x`, `13.x`
- SantriKoding tag Laravel, difilter hanya artikel yang menyebut Laravel 9 sampai 13
- `nqhung97/docs-laravel-v13`
- `fchis/laravel-buildspec-training`
- `fchis/Laravel-13x-Planner-Instructions`
- `fchis/Laravel-13x-Code-Instructions`
- `patelakshay3943/laravel12-dataset-cp`
- `patelakshay3943/laravel12-dataset`
- `brijmansuriya/web-beast-laravel`
- `codeXpedite/Laravel`
- `relai-ai/laravel-reasoning`

Semua source Laravel digabung ke satu file agar registry tetap sederhana:

```txt
data/mined/instruction/laravel_instruction.jsonl
```

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
- `akahana/english-indonesia-wikimatrix` sebagai translation
- `akahana/english-indonesia` sebagai translation/chat translation
- `ermandmand/indonesian-simple-instruction-dataset` sebagai instruction sederhana
- `IndonesiaAI/sft-dataset` sebagai SFT/instruction
- `audichandra/bitext_customer_support_llm_dataset_indonesian` sebagai customer support instruction
- `LorthGyu/indonesian-qa` sebagai Q&A Bahasa Indonesia
- `theonlydo/indonesia-slang` sebagai slang/vocabulary
- `nahiar/indonesia-slang` sebagai slang/vocabulary

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

Preset `laravel` juga mencoba dataset HuggingFace Laravel yang tercatat di bagian sumber. Semua hasilnya tetap di-merge ke `laravel_instruction.jsonl`.

## 6. Mining Code / HumanEval

```powershell
python tools\mine_general_assistant_data.py --preset code
```

Output:

```txt
data/mined/instruction/code_instruction.jsonl
```

Tool membaca schema umum seperti `prompt`, `canonical_solution`, `test`, `entry_point`, `completion`, `solution`, `code`, `messages`, dan `conversations`, lalu menormalisasi semuanya ke instruction row SigerLM.

## 7. Mining Semua Sumber

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

## 7.1 Mining Indonesian HF Mix untuk Kaggle

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
Report HF mix juga mencatat `sample_keys` untuk membantu memperbaiki source yang menghasilkan `0 rows`.

## 7.2 Ingest Kaggle Add Input Lokal

Dataset yang ditambahkan dari panel **Add Input** Kaggle tersedia di `/kaggle/input`, tetapi tidak otomatis ikut training. Gunakan tool ini untuk scan file `.txt`, `.csv`, `.json`, dan `.jsonl` lalu mengubahnya menjadi data lokal SigerLM:

```powershell
python tools\ingest_kaggle_inputs.py
```

Output:

```txt
data/kaggle/kaggle_extra_text.txt
data/kaggle/kaggle_extra_instruction.jsonl
configs/datasets/kaggle_local_inputs.json
data/kaggle/kaggle_ingest_report.json
```

`kaggle_extra_text.txt` otomatis ikut base training karena `main.py` membaca file `.txt` di folder `data/`. `kaggle_extra_instruction.jsonl` bisa dipakai LoRA/instruction tuning melalui registry.

Untuk membatasi jumlah file saat smoke test:

```powershell
python tools\ingest_kaggle_inputs.py --max-files 20
```

Untuk build corpus dari input Kaggle saja:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\kaggle_local_inputs.json
```

Untuk build corpus gabungan HF mix + Kaggle input + Lampung:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle.json
```

## 8. Build Corpus agar Bisa Dibaca SigerLM

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
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix.json --max-row-tokens 2048
```

Output corpus:

```txt
data/corpus/indonesian_hf_mix_train.jsonl
```

Untuk Indonesian HF mix + Kaggle Add Input:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle.json --max-row-tokens 2048
```

Output corpus:

```txt
data/corpus/indonesian_hf_mix_plus_kaggle_train.jsonl
```

## 9. Rekomendasi Mixing

Untuk tahap awal general assistant:

```txt
Q&A Indonesia          weight 2
Laravel docs/tutorial  weight 2
Code/HumanEval         weight 1
Commercial-safe reasoning weight 1
General instruction    weight 1
Lampung domain         weight 1
```

Alasannya:

- Q&A melatih model menjawab berdasarkan konteks.
- Laravel docs/tutorial memberi kemampuan domain coding yang konkret.
- Code/HumanEval memberi latihan problem solving dan penyelesaian fungsi teruji.
- Commercial-safe reasoning memberi latihan matematika, coding, dan penalaran sintetis yang lisensinya lebih aman untuk target profit.
- General instruction menjaga gaya assistant umum.
- Lampung tetap ada sebagai kemampuan lokal, tetapi tidak mendominasi general assistant.

## 10. Quality Gate

Sebelum training serius:

1. Cek sampel JSONL manual.
2. Hapus data yang terlalu panjang, kosong, atau noisy.
3. Pastikan jawaban Q&A tidak kehilangan konteks.
4. Pisahkan eval set kecil untuk Q&A dan Laravel.
5. Jangan mencampur data berlisensi ketat ke release publik tanpa review lisensi.

`tools/build_instruction_corpus.py` sudah punya quality gate default:

- deduplikasi lintas source setelah semua data digabung,
- filter row terlalu panjang dengan `--max-row-tokens` default `2048`,
- sanity check ringan untuk row Laravel/PHP, termasuk code fence yang belum tertutup dan snippet `?>` tanpa `<?php`,
- report JSON otomatis di samping output corpus, misalnya `data/corpus/general_assistant_mining_train.report.json`.

Untuk model kecil 11.8M parameter, gunakan batas aman:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json --max-row-tokens 2048
```

Kalau ingin smoke test lebih ketat:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json --max-row-tokens 1024
```

Di Kaggle, pantau disk sebelum mining besar:

```bash
du -sh data data/mined data/corpus checkpoints 2>/dev/null || true
df -h /kaggle/working
```

Untuk `mine_indonesian_hf_mix.py`, hindari langsung `--max-items-per-source 50000` kalau disk mulai mepet. Naikkan bertahap dari `200`, `5000`, `10000`, lalu baru `50000`.

Untuk eksperimen Kaggle 2x T4 dengan source lebih besar, gunakan run awal yang konservatif:

```powershell
python tools\mine_indonesian_hf_mix.py --max-items-per-source 60000
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 512
python tools\inspect_lora_dataset.py data\corpus\indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl --limit 10 --stats-limit 500 --max-seq-len 512
```

Untuk model `small` 11.8M parameter, default base training lokal diset ke `max_steps=3000`, dan config LoRA reasoning mix diset ke 3000 optimizer updates agar run pertama tidak terlalu panjang. Jika loss sudah turun sehat dan output mulai koheren, lanjutkan eksperimen kedua dengan base/LoRA 5000 steps atau `--max-row-tokens 768`.

Contoh inspeksi cepat:

```powershell
Get-Content data\mined\instruction\indonesian_qa_instruction.jsonl -TotalCount 3
Get-Content data\mined\instruction\code_instruction.jsonl -TotalCount 3
Get-Content data\mined\instruction\laravel_instruction.jsonl -TotalCount 3
```

Compile check:

```powershell
python -m py_compile tools\mine_general_assistant_data.py training\dataset_registry.py tools\build_instruction_corpus.py
python -m py_compile tools\mine_indonesian_hf_mix.py
python -m py_compile tools\ingest_kaggle_inputs.py
```

## 11. Debug LoRA Instruction Mask

Sebelum training LoRA besar di Kaggle, cek dulu apakah row JSONL sudah dibungkus ke format chat dan loss hanya dihitung pada jawaban assistant:

```powershell
python tools\debug_lora_dataset.py data\corpus\indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl --limit 5 --max-seq-len 512
```

Catatan penting: file corpus SigerLM tidak perlu menyimpan token `<|assistant|>` secara mentah. Token itu ditambahkan oleh `lora/dataset.py` saat training. Karena itu, debug mask harus mengecek hasil formatter, bukan hanya `str(row)` dari JSONL.

Untuk general chat, prioritaskan corpus instruction yang bersih dan sempit dulu. Dataset campuran text completion, reasoning, software, dan Lampung bisa dipakai nanti, tetapi model kecil 11.8M parameter lebih mudah stabil jika LoRA pertamanya fokus ke instruction-following umum.

Training LoRA general assistant yang lebih fokus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_assistant_mining.json --max-row-tokens 2048
python lora\run_lora.py --config configs\training\general_assistant_lora.json
```
