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

### Phase-2 mining tanpa mengulang row lama

Untuk tahap awal, aman membatasi tiap source, misalnya `--max-items-per-source 60000`. Saat phase 2, jangan mulai lagi dari row awal. Gunakan report run sebelumnya sebagai offset, lalu append ke file output yang sama:

```powershell
python tools\mine_indonesian_hf_mix.py `
  --max-items-per-source 60000 `
  --resume-from-report data\mined\hf_indonesia\hf_mix_report.json `
  --append `
  --dedupe-existing
```

`--resume-from-report` membaca jumlah raw row yang sudah discan per source dari report sebelumnya. `--append` menambahkan hasil baru ke JSONL/TXT lama, sedangkan `--dedupe-existing` membaca JSONL lama dan melewati instruction/input/output yang sudah ada. Report baru akan menulis `start_offset`, `skipped`, `scanned`, dan `next_start_offset` untuk tiap source.

Jika ingin membuat output phase 2 terpisah dulu:

```powershell
python tools\mine_indonesian_hf_mix.py `
  --max-items-per-source 60000 `
  --resume-from-report data\mined\hf_indonesia\hf_mix_report.json `
  --instruction-output data\mined\hf_indonesia\indonesian_hf_mix_instruction_phase2.jsonl `
  --text-output data\indonesian_hf_mix_phase2.txt `
  --report-output data\mined\hf_indonesia\hf_mix_report_phase2.json
```

Untuk skip manual tanpa report:

```powershell
python tools\mine_indonesian_hf_mix.py --max-items-per-source 60000 --start-row-per-source 60000
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

### Normalisasi otomatis HF mix

`tools/mine_indonesian_hf_mix.py` sekarang dibuat lebih toleran terhadap schema dataset HuggingFace yang tidak seragam. Semua output tetap dinormalisasi ke schema SigerLM:

```json
{"instruction":"...","input":"...","output":"...","system":"...","source":"...","type":"..."}
```

Normalizer saat ini menangani pola umum berikut:

- translation berbentuk dict, misalnya `{"id":"...","en":"..."}`,
- translation berbentuk stringified dict di satu kolom, misalnya `"{'id': '...', 'en': '...'}"`,
- pasangan teks satu kolom dengan delimiter seperti `###>`, `##>`, `|||`, atau tab,
- dataset instruction/chat dengan field `messages`, `conversations`, `question`, `response_j`, `response`, `answer`, atau `output`,
- customer support CSV-like dengan kolom generik `column3` sebagai instruction dan `column4` sebagai response,
- slang/vocabulary dengan pasangan `slang` dan `formal`,
- vocabulary satu kolom `text` untuk contoh kosakata ringan.

Kalau sebuah source tetap menghasilkan `0 rows`, cek `sample_keys`, `sample_row`, dan `error` di:

```txt
data/mined/hf_indonesia/hf_mix_report.json
```

Tujuannya bukan membuat model menghafal semua format dataset, tetapi memastikan pipeline mining bisa membaca sebanyak mungkin source lalu mengubahnya ke format instruction yang konsisten.

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

## 7.3 Mining Percakapan Lampung dari Dataset Lokal

Jika sudah ada hasil mining lokal di `data/mined/`, `data/kaggle/`, atau `data/corpus/`, gunakan tool ini untuk mengambil kandidat percakapan lalu menerjemahkan setiap baris ke backend `translatelampung.com`:

```powershell
python tools\mine_lampung_conversations.py --target-count 1000
```

Default input yang dibaca:

```txt
data/mined/instruction/*.jsonl
data/mined/hf_indonesia/*.jsonl
data/kaggle/*.jsonl
data/corpus/*_train.jsonl
```

Tool akan mencari percakapan dari beberapa bentuk data:

- `messages` atau `conversations`,
- teks dengan marker pembicara seperti `A: ...` dan `B: ...`,
- row instruction yang terlihat seperti Q&A/chat/customer support dan bisa dijadikan dialog dua giliran (`A: instruction`, `B: output`).

Secara default, tool tidak mengubah semua instruction row menjadi percakapan agar data coding, reasoning, translation, dan vocabulary tidak ikut terseret sebagai dialog palsu. Kalau memang ingin mengambil semua pasangan `instruction`/`output`, aktifkan:

```powershell
python tools\mine_lampung_conversations.py --target-count 1000 --include-any-instruction-pairs
```

Output:

```txt
data/mined/lampung/lampung_conversations_translated.jsonl
data/mined/lampung/lampung_conversations_checkpoint.json
data/mined/lampung/lampung_conversations_report.json
```

Setiap kandidat percakapan menghasilkan row instruction untuk Dialek O (`abl`) dan Dialek A (`ljp`) jika kedua dialek diaktifkan. Gunakan `--dry-run` untuk menguji ekstraksi tanpa hit API:

```powershell
python tools\mine_lampung_conversations.py --target-count 10 --dry-run --fresh
```

Build corpus khusus hasil mining percakapan Lampung:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_conversations_mined.json --max-row-tokens 512
```

Catatan etis dan teknis: tool ini memakai jeda default per baris dan per percakapan agar tidak agresif ke server pihak ketiga. Jika mendapat HTTP 429, tool otomatis cooldown dan melanjutkan dari checkpoint. Pastikan penggunaan sesuai aturan situs/API yang dipakai.

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
python tools\build_software_engineering_seed.py
python tools\build_reasoning_seed.py
python tools\build_uncertainty_seed.py
python tools\build_instruction_corpus.py --registry configs\datasets\indonesian_hf_mix_plus_kaggle_reasoning.json --max-row-tokens 512
python tools\inspect_lora_dataset.py data\corpus\indonesian_hf_mix_plus_kaggle_reasoning_train.jsonl --limit 10 --stats-limit 500 --max-seq-len 512
```

`build_uncertainty_seed.py` bukan seed hard-refusal untuk semua hal yang tidak diketahui. Isinya pola "honest & helpful": model tetap mencoba membantu dengan asumsi eksplisit, koreksi miskonsepsi, checklist verifikasi, dan caveat sumber/versi. Hard refusal hanya dipakai untuk kasus yang memang berisiko seperti secret, kredensial, diagnosis pasti, atau klaim finansial pasti.

Contoh gaya output uncertainty yang diharapkan:

```txt
<thought>User bertanya tentang framework yang berbeda. Saya bisa memetakan konsep umum,
tetapi harus memberi caveat agar user memverifikasi dokumentasi sesuai versi.</thought>
Jawaban: Di Django, padanan umum untuk membatasi akses user login adalah
`@login_required`. Ini mirip tujuan auth middleware di Laravel, tetapi detailnya
tetap perlu dicek di dokumentasi Django sesuai versi yang dipakai.
```

Porsi uncertainty seed sebaiknya kecil, sekitar 2-3% dari corpus instruction. Tujuannya membentuk kebiasaan jujur tentang tingkat keyakinan, bukan membuat model terlalu sering menolak tugas.

Untuk model `small` 11.8M parameter, default base training lokal diset ke `max_steps=3000`, dan config LoRA reasoning mix diset ke 3000 optimizer updates agar run pertama tidak terlalu panjang. Jika loss sudah turun sehat dan output mulai koheren, lanjutkan eksperimen kedua dengan base/LoRA 5000 steps atau `--max-row-tokens 768`.

Jika ingin base training dengan konteks lebih panjang tanpa menaikkan parameter model, gunakan profile dense `small_context`:

```powershell
$env:SIGER_MODEL_PROFILE="small_context"
python main.py
```

Profile ini tetap `d_model=256` dan `n_layers=8`, tetapi menaikkan `max_seq_len` dari 128 ke 256. Pakai setelah run `small` stabil karena biaya trainingnya lebih besar.

Untuk mencoba Sparse Mamba MoE tanpa merusak jalur dense lama, aktifkan profile opt-in saat base training:

```powershell
$env:SIGER_MODEL_PROFILE="small_moe"
python main.py
```

Profile default `small` tetap dense dan kompatibel dengan checkpoint lama. `small_moe` menambahkan sparse feed-forward experts pada sebagian layer (`8 experts`, `top_k=2`) dengan auxiliary load-balance loss kecil. Untuk jalur Dense -> MoE otomatis, pretrain dense dengan `moe_dense_base` terlebih dahulu karena shape-nya cocok dengan `small_moe`:

```powershell
$env:SIGER_MODEL_PROFILE="moe_dense_base"
$env:SIGER_CHECKPOINT_DIR="checkpoints/auto/dense_moe_base"
python main.py
```

Jangan warm-start `small_moe` dari checkpoint `siger_medium` tanpa converter khusus, karena `siger_medium` memakai `d_model=512`, `n_layers=12`, sedangkan `small_moe` memakai `d_model=384`, `n_layers=10`.

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
