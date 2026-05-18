from __future__ import annotations

import hashlib
from typing import Any


THOUGHT_START = "<thought>"
THOUGHT_END = "</thought>"


def has_cot(output: str) -> bool:
    return THOUGHT_START in output and THOUGHT_END in output


def stable_fraction(*parts: str) -> float:
    payload = "\n".join(parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def should_apply_cot(row: dict[str, Any], ratio: float) -> bool:
    ratio = max(0.0, min(1.0, float(ratio)))
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    return stable_fraction(
        str(row.get("source", "")),
        str(row.get("type", "")),
        str(row.get("instruction", "")),
        str(row.get("input", "")),
    ) < ratio


def _task_type(row: dict[str, Any]) -> str:
    return str(row.get("type") or "").lower()


def build_reasoning(row: dict[str, Any], *, mode: str = "auto") -> str:
    instruction = str(row.get("instruction") or "")
    input_text = str(row.get("input") or "")
    task_type = _task_type(row)

    if mode == "minimal":
        return (
            "Identifikasi tugas user, periksa konteks input, pilih langkah yang relevan, "
            "lalu susun jawaban akhir secara ringkas dan akurat."
        )

    if "translation" in task_type or "terjemah" in instruction.lower():
        return (
            "Tugas ini adalah translasi. Tentukan bahasa sumber dan target, pertahankan makna, "
            "hindari terjemahan kata-per-kata jika membuat kalimat tidak natural, lalu berikan hasil akhir."
        )
    if "qa" in task_type or "pertanyaan" in instruction.lower() or "jawab" in instruction.lower():
        return (
            "Tugas ini adalah tanya jawab. Cari informasi yang relevan dari pertanyaan dan konteks, "
            "hindari mengarang fakta, lalu jawab dengan jelas."
        )
    if "vocab" in task_type or "kata" in instruction.lower() or "ungkapan" in instruction.lower():
        return (
            "Tugas ini membahas kata atau ungkapan. Identifikasi istilah utama, jelaskan makna, "
            "dan gunakan contoh atau konteks jika tersedia."
        )
    if "text_completion" in task_type or "lanjutkan" in instruction.lower():
        return (
            "Tugas ini adalah melanjutkan teks. Pahami topik, gaya bahasa, dan alur kalimat dari input, "
            "lalu lanjutkan secara natural tanpa mengubah arah pembahasan."
        )
    if "customer" in task_type or "support" in task_type:
        return (
            "Tugas ini adalah customer support. Identifikasi masalah pengguna, beri empati singkat, "
            "tawarkan langkah penyelesaian, dan jaga jawaban tetap jelas."
        )
    if "lampung" in task_type or "lampung" in instruction.lower() or "dialek" in instruction.lower():
        return (
            "Tugas ini berkaitan dengan Bahasa Lampung. Perhatikan dialek, konteks kalimat, "
            "struktur subjek-predikat-objek, dan pilih padanan yang natural."
        )
    if "code" in task_type or "program" in instruction.lower() or "aplikasi" in instruction.lower():
        return (
            "Tugas ini berkaitan dengan pengembangan software. Tentukan requirement, pisahkan layer, "
            "tambahkan validasi, error handling, test, logging, dan dokumentasi bila relevan."
        )

    if input_text:
        return (
            "Baca instruksi dan input dengan teliti. Tentukan inti permintaan, gunakan konteks yang tersedia, "
            "lalu susun jawaban akhir yang akurat."
        )
    return "Pahami instruksi user, pilih langkah paling relevan, lalu berikan jawaban akhir yang langsung menjawab."


def apply_cot(row: dict[str, Any], *, mode: str = "auto") -> dict[str, Any]:
    output = str(row.get("output") or "").strip()
    if not output or has_cot(output):
        return row

    converted = dict(row)
    converted["output"] = f"{THOUGHT_START} {build_reasoning(row, mode=mode)} {THOUGHT_END} {output}"
    converted["type"] = f"{row.get('type') or 'instruction'}_cot"
    converted["cot"] = True
    return converted


def maybe_apply_cot(row: dict[str, Any], *, ratio: float = 0.0, mode: str = "auto") -> dict[str, Any]:
    if should_apply_cot(row, ratio):
        return apply_cot(row, mode=mode)
    return row
