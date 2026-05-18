from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Kamu adalah SIGERLLM, asisten penalaran yang menjawab dengan runtut. "
    "Jika tugas membutuhkan analisis, tulis penalaran di dalam tag <thought>...</thought> "
    "lalu berikan jawaban akhir yang jelas."
)


def row(instruction: str, output: str, input_text: str = "", task_type: str = "reasoning_cot") -> dict[str, Any]:
    return {
        "instruction": " ".join(instruction.split()),
        "input": " ".join(input_text.split()),
        "output": " ".join(output.split()),
        "system": SYSTEM_PROMPT,
        "source": "siger_reasoning_seed",
        "type": task_type,
    }


def thought(reasoning: str, answer: str) -> str:
    return f"<thought> {' '.join(reasoning.split())} </thought> {' '.join(answer.split())}"


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.extend(
        [
            row(
                "Terjemahkan ke Dialek O: Kamu mau pergi ke mana?",
                thought(
                    "Pertanyaan meminta translasi ke Dialek O. Subjek 'kamu' dalam konteks umum dapat memakai 'niku' atau 'gham' tergantung ragam. Untuk bentuk tanya natural, pakai kata 'ago' untuk mau, 'mit/mik' untuk pergi/ke, dan 'jow' untuk ke mana. Susun sebagai pertanyaan singkat.",
                    "Niku ago mit jow?",
                ),
                task_type="lampung_translation_cot",
            ),
            row(
                "Terjemahkan ke Bahasa Indonesia dan jelaskan struktur: Nyak haga mengan manuk di warung paghek jalan.",
                thought(
                    "Kalimat Lampung ini memakai 'nyak' sebagai aku, 'haga' sebagai mau, 'mengan' sebagai makan, 'manuk' sebagai ayam, 'di warung' lokasi, dan 'paghek jalan' berarti dekat jalan. Struktur kalimatnya subjek + keinginan + verba + objek + keterangan tempat.",
                    "Artinya: aku mau makan ayam di warung dekat jalan. Strukturnya S-P-O-K: nyak sebagai subjek, haga mengan sebagai predikat verbal, manuk sebagai objek, dan di warung paghek jalan sebagai keterangan tempat.",
                ),
                task_type="lampung_grammar_cot",
            ),
            row(
                "Kapan sebaiknya memakai dialek O atau Nyo saat menjawab user?",
                thought(
                    "Pemilihan dialek harus mengikuti konteks user. Jika user menulis bentuk O seperti nyak, haga, paghek, maka balas dengan dialek O. Jika user menulis bentuk Nyo atau meminta Nyo secara eksplisit, gunakan Nyo. Jika konteks tidak jelas, tanyakan preferensi atau gunakan Indonesia agar tidak salah dialek.",
                    "Gunakan dialek yang sama dengan input user. Jika user eksplisit meminta Dialek O, pakai O. Jika meminta Nyo, pakai Nyo. Jika tidak jelas, tanyakan dulu atau jelaskan bahwa ada beberapa dialek.",
                ),
                task_type="dialect_reasoning_cot",
            ),
            row(
                "Jelaskan mengapa model tidak boleh langsung menerjemahkan kata-per-kata pada Bahasa Lampung.",
                thought(
                    "Terjemahan kata-per-kata sering gagal karena urutan kata, ragam dialek, tingkat kesopanan, dan idiom dapat berbeda. Model harus melihat konteks kalimat, tujuan ujaran, dan target bahasa sebelum memilih padanan.",
                    "Karena Bahasa Lampung punya variasi dialek dan struktur yang tidak selalu sejajar dengan Bahasa Indonesia. Terjemahan yang baik harus mempertahankan makna, konteks, dan kewajaran kalimat, bukan hanya mengganti kata satu per satu.",
                ),
                task_type="translation_reasoning_cot",
            ),
            row(
                "Jika user meminta jawaban singkat tanpa penjelasan, apa yang harus dilakukan?",
                thought(
                    "Instruksi user mengutamakan jawaban singkat. Penalaran internal tidak perlu ditampilkan jika tidak diminta. Jawaban harus langsung, padat, dan tetap benar.",
                    "Jawab langsung dan singkat. Jangan tampilkan penjelasan panjang kecuali user memintanya.",
                ),
                task_type="instruction_following_reasoning",
            ),
            row(
                "Pilih jawaban terbaik: Semua burung bisa terbang. Penguin adalah burung. Apakah penguin bisa terbang?",
                thought(
                    "Premis pertama menyatakan semua burung bisa terbang, tetapi di dunia nyata penguin adalah pengecualian. Jika ini soal logika formal, jawaban mengikuti premis: bisa. Jika ini soal fakta dunia nyata, penguin tidak bisa terbang. Karena pertanyaan tidak menyebut hanya mengikuti premis, jawab dengan membedakan keduanya.",
                    "Secara logika formal berdasarkan premis, penguin bisa terbang. Namun secara fakta dunia nyata, penguin tidak bisa terbang.",
                ),
                task_type="logic_cot",
            ),
            row(
                "Sebuah API menerima URL dari user lalu server mengambil kontennya. Apa risiko utamanya dan mitigasinya?",
                thought(
                    "Server-side fetch dari URL user rawan SSRF. User bisa mencoba localhost, IP privat, metadata cloud, atau redirect berbahaya. Mitigasi perlu validasi skema, allowlist domain, blok IP privat, timeout, limit redirect, dan ukuran response maksimum.",
                    "Risiko utamanya SSRF. Mitigasinya: allowlist domain, hanya izinkan http/https, blok localhost/IP privat/link-local, batasi redirect, timeout, limit ukuran response, dan audit log request.",
                ),
                task_type="security_reasoning_cot",
            ),
            row(
                "Buat rencana sederhana membangun fitur login yang aman.",
                thought(
                    "Login aman membutuhkan validasi input, hashing password, rate limit, session/JWT aman, audit log, dan test. Password tidak boleh disimpan plaintext atau dienkripsi reversible.",
                    "Rencananya: buat schema validasi email/password, hash password dengan Argon2id atau bcrypt, tambah rate limit, gunakan JWT/session dengan expiry, simpan secret di env, audit login gagal/berhasil, dan buat unit serta integration test.",
                ),
                task_type="software_reasoning_cot",
            ),
            row(
                "Selesaikan: Jika semua data penting harus punya audit log, dan perubahan role adalah data penting, apa konsekuensinya?",
                thought(
                    "Aturan umum menyatakan semua data penting harus diaudit. Perubahan role termasuk data penting. Maka perubahan role wajib dicatat di audit log.",
                    "Setiap perubahan role wajib masuk audit log, termasuk siapa yang mengubah, role lama, role baru, waktu, IP, dan correlation ID.",
                ),
                task_type="logic_cot",
            ),
            row(
                "Jelaskan bedanya jawaban akhir dan penalaran dalam format <thought>.",
                thought(
                    "Tag thought dipakai untuk melatih model menyusun alasan sebelum jawaban. Jawaban akhir tetap harus berada setelah tag penalaran agar mudah dipakai user atau sistem evaluator.",
                    "Penalaran diletakkan di dalam <thought>...</thought>. Jawaban final diletakkan setelahnya, ringkas dan langsung menjawab pertanyaan.",
                ),
                task_type="cot_format_instruction",
            ),
        ]
    )

    templates = [
        ("Analisis pilihan dialek untuk kalimat Lampung berikut.", "Pertama identifikasi penanda dialek, lalu pilih O/Nyo, lalu berikan jawaban."),
        ("Jelaskan struktur tata bahasa kalimat berikut.", "Pecah menjadi subjek, predikat, objek, dan keterangan jika ada."),
        ("Selesaikan soal logika berikut.", "Pisahkan premis, inferensi, kemungkinan pengecualian, lalu jawaban akhir."),
        ("Review rancangan API berikut dari sisi keamanan.", "Cari input berbahaya, auth, authorization, logging, dan mitigasi."),
        ("Buat aplikasi sederhana berikut dengan standar production.", "Tentukan stack, struktur folder, validasi, error handling, test, Docker, dan dokumentasi."),
    ]
    for instruction, guidance in templates:
        rows.append(
            row(
                instruction,
                thought(
                    f"Tugas ini membutuhkan proses runtut. {guidance} Pastikan jawaban akhir tidak hanya berupa klaim, tetapi menyebut keputusan dan alasan utamanya.",
                    "Ikuti langkah analisis tersebut, lalu berikan jawaban akhir yang jelas dan bisa dieksekusi.",
                ),
                task_type="reasoning_template",
            )
        )

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SigerLM reasoning/CoT seed dataset.")
    parser.add_argument("--output", default="data/capabilities/reasoning_cot_seed.jsonl")
    args = parser.parse_args()

    output = Path(args.output)
    rows = build_rows()
    write_jsonl(output, rows)
    print(f"Reasoning seed rows: {len(rows)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
