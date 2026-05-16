# Security Policy

## Supported Versions

SIGER LLM masih berada pada tahap eksperimen aktif. Untuk saat ini, hanya branch utama yang mendapat perhatian terhadap isu keamanan.

| Version | Supported |
|---|---|
| `main` | ✅ |
| older snapshots | ❌ |

---

## Reporting a Vulnerability

Jika menemukan isu keamanan, mohon **jangan langsung membuat public issue** yang berisi detail eksploitasi.

Laporkan secara privat melalui:

- Email: `syarifsoden30@gmail.com`

Sertakan:

- ringkasan masalah
- langkah reproduksi
- dampak potensial
- file atau modul terkait
- saran mitigasi jika ada

---

## Scope

Contoh isu yang layak dilaporkan:

- kebocoran secret/token
- path traversal pada API/inference server
- unsafe file loading
- command execution yang tidak semestinya
- dependency risk dengan dampak nyata
- scraping pipeline yang bisa mengekspos credential

---

## Response Expectations

Maintainer akan berusaha:

1. Mengonfirmasi laporan
2. Meninjau dampak
3. Menyusun perbaikan bila valid
4. Memberi kredit kepada pelapor jika disetujui

Karena proyek ini masih eksperimental dan dikelola secara personal, waktu respons dapat bervariasi.