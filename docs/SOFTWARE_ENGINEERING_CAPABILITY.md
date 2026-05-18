# Software Engineering Capability Pack

Dokumen ini menjelaskan fondasi kemampuan SIGERLLM untuk membantu membuat aplikasi/web lintas bahasa pemrograman tanpa mengubah core model.

Tujuannya bukan mengganti arsitektur SigerLM, tetapi menambahkan **capability dataset** yang bisa dipakai saat LoRA/instruction tuning agar model belajar pola kerja tim pengembang aplikasi yang lebih standar.

## Scope

Capability pack ini menanamkan pola:

- language-agnostic codebase analysis
- Tree-sitter/AST normalization blueprint
- dynamic boilerplate generation
- ISO/IEC 25010 software quality guidance
- ISO/IEC 27001 security controls
- automated testing strategy
- structured logging and audit trails
- OpenAPI and compliance mapping

## Files

```txt
tools/build_software_engineering_seed.py
data/capabilities/software_engineering_seed.jsonl
configs/datasets/software_engineering_instruction.json
configs/datasets/indonesian_hf_mix_plus_kaggle_software.json
configs/training/software_engineering_lora.json
```

## Canonical Row Format

Rows follow the standard SigerLM instruction schema:

```json
{
  "instruction": "...",
  "input": "...",
  "output": "...",
  "system": "...",
  "source": "siger_software_engineering_seed",
  "type": "software_engineering_capability"
}
```

## Capability Modules

### 1. Language-Agnostic Parser & Structure Generator

SIGERLLM should learn to inspect a project independently of programming language:

- scan folders and manifests
- identify package manager and framework
- normalize functions/classes/routes/models/dependencies into JSON
- prefer Tree-sitter/AST parsing when available
- generate stack-idiomatic project structure while preserving domain/service/repository separation

### 2. ISO/IEC 25010 Quality Blueprint

Generated apps should include:

- maintainability guard
- portability through Docker/Docker Compose
- reliability through global error handling, retry, timeout, circuit breaker, and health checks
- performance basics such as pagination and N+1 avoidance

### 3. ISO/IEC 27001 Security Controls

Generated apps should default to:

- schema-based input validation
- SQL injection, XSS, and SSRF prevention patterns
- Argon2id/bcrypt for passwords
- AES-256-GCM/KMS only for data that must be decrypted
- RBAC/auth middleware for sensitive endpoints
- secure headers, rate limits, and secret handling

### 4. Automated Testing & Verification

Generated apps should include:

- unit tests
- integration tests
- e2e tests for main flows
- edge-case tests
- fuzz/property tests for validators/parsers where useful
- target coverage guidance of at least 80% for business modules

### 5. Logging & Audit Trails

Generated apps should use:

- JSON structured logs
- timestamp, level, service, environment, correlation ID, event, message
- no secrets or raw sensitive PII in logs
- tamper-evident audit log design for critical actions

### 6. Documentation & Compliance Mapping

Generated apps should include:

- OpenAPI/Swagger docs
- README/runbook
- env.example
- COMPLIANCE.md mapping code features to ISO/IEC 25010 and ISO/IEC 27001 controls

## Build Commands

Build seed dataset:

```bash
python tools/build_software_engineering_seed.py
```

Build software-only instruction corpus:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/software_engineering_instruction.json
```

Build Indonesian + Kaggle + Lampung + software engineering corpus:

```bash
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle_software.json
```

Train software engineering LoRA:

```bash
python lora/run_lora.py --config configs/training/software_engineering_lora.json
```

## Important Boundary

This pack is a **training/instruction foundation**, not a full static analyzer implementation yet. Real AST execution and Tree-sitter parsing should be added later as tools under `tools/` or `evaluation/` so generated answers can be verified against real code.
