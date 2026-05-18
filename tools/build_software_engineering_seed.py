from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "Kamu adalah SIGERLLM, asisten pengembang aplikasi yang mampu merancang, "
    "membangun, menguji, mengamankan, dan mendokumentasikan software lintas bahasa "
    "pemrograman dengan standar ISO/IEC 25010 dan ISO/IEC 27001."
)


def row(instruction: str, output: str, input_text: str = "", task_type: str = "software_engineering") -> dict[str, Any]:
    return {
        "instruction": " ".join(instruction.split()),
        "input": " ".join(input_text.split()),
        "output": " ".join(output.split()),
        "system": SYSTEM_PROMPT,
        "source": "siger_software_engineering_seed",
        "type": task_type,
    }


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.extend(
        [
            row(
                "Buat blueprint aplikasi web production-ready lintas bahasa.",
                "Mulai dari requirement, lalu pilih stack. Buat struktur folder berlapis: route/controller, service/use-case, repository/gateway, entity/model, validation/schema, middleware, config, tests, docs, dan deployment. Pisahkan business logic dari framework. Tambahkan Dockerfile, docker-compose, OpenAPI, structured logging, global error handler, input validation, RBAC, dan test unit/integration/e2e.",
            ),
            row(
                "Jelaskan cara menganalisis codebase bahasa apa pun secara language-agnostic.",
                "Scan tree folder, deteksi package manager, baca manifest seperti package.json, pyproject.toml, Cargo.toml, go.mod, composer.json, atau pom.xml. Parse file kode dengan Tree-sitter jika grammar tersedia, fallback ke token/regex aman hanya untuk metadata. Normalisasi hasil menjadi JSON berisi symbols, imports, exports, routes, models, functions, classes, dependencies, tests, dan risiko arsitektur.",
                task_type="parser_blueprint",
            ),
            row(
                "Buat format JSON standar untuk hasil AST/Tree-sitter.",
                json.dumps(
                    {
                        "language": "string",
                        "path": "string",
                        "symbols": [{"name": "string", "kind": "function|class|route|model", "line": 1}],
                        "imports": [{"module": "string", "symbols": ["string"]}],
                        "calls": [{"from": "string", "to": "string"}],
                        "routes": [{"method": "GET", "path": "/items", "handler": "ItemController.index"}],
                        "quality": {"cyclomatic_complexity": 1, "long_function": False},
                    },
                    ensure_ascii=False,
                ),
                task_type="parser_blueprint",
            ),
            row(
                "Buat aturan dynamic boilerplate engine.",
                "Gunakan intent user, bahasa, framework, tipe aplikasi, database, auth, dan deployment target sebagai input. Generate struktur folder yang mengikuti idiom stack tersebut, tetapi tetap menjaga lapisan domain, service, repository, validation, middleware, tests, docs, dan deployment. Jangan mencampur query database langsung di controller.",
                task_type="boilerplate_blueprint",
            ),
        ]
    )

    quality_controls = [
        (
            "Maintainability Guard",
            "Batasi fungsi agar fokus pada satu tanggung jawab. Jika cyclomatic complexity tinggi, pecah menjadi helper/service. Hindari file terlalu panjang, duplikasi, magic string, dan coupling langsung ke framework. Tambahkan type hints atau tipe eksplisit jika bahasa mendukung.",
        ),
        (
            "Portability & Compatibility Layer",
            "Sediakan Dockerfile, docker-compose untuk database/cache bila dibutuhkan, file env.example, healthcheck, dan dokumentasi cara menjalankan lokal. Jangan hardcode credential atau path lokal.",
        ),
        (
            "Reliability & Fault Tolerance Engine",
            "Tambahkan global error handler, timeout, retry terbatas dengan backoff untuk koneksi eksternal, idempotency untuk operasi penting, circuit breaker untuk dependency mikroservis, dan health/readiness endpoint.",
        ),
        (
            "Performance Efficiency Guard",
            "Gunakan pagination, index database yang relevan, query terukur, caching hanya untuk data aman, dan hindari N+1 query. Tambahkan benchmark ringan jika fitur kritis.",
        ),
    ]
    for name, guidance in quality_controls:
        rows.append(
            row(
                f"Terapkan ISO/IEC 25010 module: {name}.",
                guidance,
                task_type="iso25010_quality",
            )
        )

    security_controls = [
        (
            "Input Validation & Sanitize Pipeline",
            "Semua request harus melewati schema validation. Validasi tipe, panjang, format, enum, range, dan whitelist URL jika ada fetch eksternal. Gunakan parameterized query atau ORM aman. Escape output HTML untuk mencegah XSS. Blok SSRF dengan allowlist host dan larang IP privat.",
        ),
        (
            "Standard Cryptography & Hashing Implementation",
            "Password harus di-hash dengan Argon2id atau bcrypt, bukan dienkripsi. Data sensitif at-rest dapat dienkripsi dengan AES-256-GCM atau library resmi cloud KMS. Jangan membuat algoritma kripto sendiri. Simpan secret di environment atau secret manager.",
        ),
        (
            "Role-Based Access Control Generator",
            "Bangun autentikasi dan otorisasi berlapis. JWT/OAuth2 boleh dipakai jika sesuai kebutuhan. Setiap endpoint sensitif harus punya middleware auth, pemeriksaan role/permission, dan audit log untuk aksi penting.",
        ),
        (
            "Secure Defaults",
            "Aktifkan CORS sempit, rate limit, secure headers, CSRF untuk session-cookie app, dependency pinning, dan validasi konfigurasi startup. Jangan menampilkan stack trace ke user production.",
        ),
    ]
    for name, guidance in security_controls:
        rows.append(
            row(
                f"Terapkan ISO/IEC 27001 security control: {name}.",
                guidance,
                task_type="iso27001_security",
            )
        )

    rows.extend(
        [
            row(
                "Buat strategi test otomatis untuk aplikasi baru.",
                "Buat unit test untuk fungsi murni dan service, integration test untuk repository/API/database, serta e2e test untuk alur utama user. Target coverage minimal 80% untuk modul bisnis. Sertakan test error path, input invalid, authorization failure, dan boundary value.",
                task_type="testing_blueprint",
            ),
            row(
                "Tambahkan fuzzing dan edge-case tester.",
                "Buat property-based test atau fuzz test untuk parser, validator, dan endpoint yang menerima input kompleks. Uji string kosong, string sangat panjang, unicode, payload nested, angka ekstrem, file kosong, file besar, dan input yang menyerupai injection.",
                task_type="testing_blueprint",
            ),
            row(
                "Buat standar structured logging JSON.",
                "Gunakan log JSON berisi timestamp, level, service, environment, correlation_id, user_id jika aman, event, message, dan metadata non-sensitif. Jangan log password, token, secret, private key, atau PII mentah. Semua request harus punya correlation ID.",
                task_type="audit_logging",
            ),
            row(
                "Buat desain tamper-evident audit log.",
                "Catat aksi penting seperti login, perubahan role, perubahan data finansial, export data, dan penghapusan data. Simpan actor, action, target, timestamp, IP, user-agent, correlation_id, before/after hash, dan hash berantai antar event agar perubahan log terdeteksi.",
                task_type="audit_logging",
            ),
            row(
                "Buat dokumentasi API dan compliance mapping.",
                "Generate OpenAPI/Swagger dari route dan schema. Tambahkan README runbook, env.example, architecture decision record jika perlu, dan COMPLIANCE.md yang memetakan fitur ke ISO/IEC 25010 dan ISO/IEC 27001: validation, auth, logging, tests, portability, reliability, dan maintainability.",
                task_type="documentation_blueprint",
            ),
        ]
    )

    stacks = [
        (
            "Python FastAPI",
            "app/main.py, app/api/routes, app/controllers, app/services, app/repositories, app/models, app/schemas, app/core/config.py, app/core/security.py, app/middleware, tests/unit, tests/integration, Dockerfile, docker-compose.yml, openapi.json, COMPLIANCE.md",
        ),
        (
            "Node.js NestJS",
            "src/modules/*/*.controller.ts, *.service.ts, *.repository.ts, *.entity.ts, dto, guards, filters, interceptors, config, test/unit, test/e2e, Dockerfile, docker-compose.yml, swagger config, COMPLIANCE.md",
        ),
        (
            "Node.js Express",
            "src/routes, src/controllers, src/services, src/repositories, src/models, src/validators, src/middlewares, src/config, tests/unit, tests/integration, Dockerfile, docker-compose.yml, openapi.yaml, COMPLIANCE.md",
        ),
        (
            "Laravel",
            "app/Http/Controllers, app/Services, app/Repositories, app/Models, app/Http/Requests, app/Policies, app/Exceptions/Handler.php, tests/Feature, tests/Unit, Dockerfile, docker-compose.yml, OpenAPI docs, COMPLIANCE.md",
        ),
        (
            "Rust Axum",
            "src/main.rs, src/routes, src/handlers, src/services, src/repositories, src/models, src/dto, src/config, src/errors.rs, tests, Dockerfile, docker-compose.yml, openapi generation, COMPLIANCE.md",
        ),
        (
            "Go Gin",
            "cmd/api/main.go, internal/handler, internal/service, internal/repository, internal/model, internal/dto, internal/middleware, internal/config, tests, Dockerfile, docker-compose.yml, openapi.yaml, COMPLIANCE.md",
        ),
    ]
    for stack, structure in stacks:
        rows.append(
            row(
                f"Buat struktur aplikasi standar untuk {stack}.",
                f"Gunakan struktur berikut sebagai fondasi: {structure}. Tambahkan validasi input, global error handling, structured JSON logging, RBAC/auth middleware, unit/integration tests, Docker support, dan COMPLIANCE.md.",
                task_type="stack_boilerplate",
            )
        )

    app_examples = [
        (
            "Buat REST API manajemen tugas dengan FastAPI dan PostgreSQL.",
            "Hasilkan project FastAPI berlapis: routes/controller untuk HTTP, service untuk business logic, repository untuk query database, SQLAlchemy model/entity, Pydantic schema validation, global exception handler, JWT RBAC, Alembic migration, pytest unit/integration tests, Dockerfile, docker-compose PostgreSQL, OpenAPI docs, structured logging, dan COMPLIANCE.md.",
        ),
        (
            "Buat aplikasi customer support dengan NestJS.",
            "Hasilkan module tickets, users, auth, dan audit. Gunakan DTO validation, guards RBAC, exception filter global, service/repository separation, Prisma/TypeORM repository, JWT auth, structured logging dengan correlation ID, Jest unit/e2e tests, Docker Compose, Swagger, dan COMPLIANCE.md.",
        ),
        (
            "Buat API pembayaran sederhana dengan Go Gin.",
            "Prioritaskan keamanan dan reliability. Buat handler, service, repository, idempotency key, request validation, timeout context, retry terbatas untuk payment gateway, audit log tamper-evident, RBAC, test unit/integration, OpenAPI, Docker, dan COMPLIANCE.md.",
        ),
        (
            "Buat service URL fetcher yang aman.",
            "Tolak desain yang rawan SSRF. Gunakan allowlist domain, blok IP privat/link-local/localhost, validasi skema http/https, timeout, ukuran response maksimum, redirect limit, structured audit log, dan test fuzz untuk URL edge case.",
            "security_refusal_and_fix",
        ),
        (
            "User meminta menyimpan password dengan AES agar bisa dibuka lagi.",
            "Tolak bagian penyimpanan password reversible. Password harus di-hash dengan Argon2id atau bcrypt. Jika ada data rahasia lain yang memang perlu dibuka lagi, gunakan enkripsi authenticated seperti AES-256-GCM dengan key dari secret manager. Dokumentasikan alasan di COMPLIANCE.md.",
            "security_refusal_and_fix",
        ),
    ]
    for instruction, output, *maybe_type in app_examples:
        rows.append(row(instruction, output, task_type=maybe_type[0] if maybe_type else "app_generation_blueprint"))

    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SigerLM software engineering capability seed dataset.")
    parser.add_argument(
        "--output",
        default="data/capabilities/software_engineering_seed.jsonl",
        help="Output JSONL path.",
    )
    args = parser.parse_args()

    output = Path(args.output)
    rows = build_rows()
    write_jsonl(output, rows)
    print(f"Software engineering seed rows: {len(rows)}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
