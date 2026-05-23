# SigerLM Learning Intake Data

This directory is for local runtime data submitted through `/v1/learning/*`.

Do not commit generated intake files. They may contain redacted user/app/web data,
privacy audit metadata, or reviewer notes.

Expected local files:

- `candidates_sanitized.jsonl`: sanitized rows eligible for human review
- `quarantine_redacted.jsonl`: rejected or sensitive rows, redacted
- `approved_training.jsonl`: rows approved for export
- `approval_log.jsonl`: reviewer decisions

Only export reviewed, non-sensitive rows into `data/corpus/` when they are safe
and allowed for training.
