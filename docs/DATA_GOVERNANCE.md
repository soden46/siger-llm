# SigerLM Data Governance

SigerLM can receive learning data from web or mobile apps through a privacy-first intake pipeline. This pipeline is designed to prevent credentials and personal data from entering training data by accident.

This is engineering guidance, not legal advice. Production use still needs policy review for applicable Indonesian and international privacy rules.

## Intake Flow

```txt
web/app data
  -> /v1/learning/intake
  -> privacy scan and redaction
  -> consent and training-purpose check
  -> candidates_sanitized.jsonl or quarantine_redacted.jsonl
  -> human review
  -> approved_training.jsonl
  -> tools/export_learning_intake.py
  -> training corpus
```

Raw sensitive values are not stored by the intake module. Stored rows contain sanitized/redacted payloads plus audit metadata.

## Sensitive Data Detected

- passwords, tokens, API keys, authorization headers
- private keys, JWTs, bearer tokens
- database URLs with embedded credentials
- URL basic-auth credentials
- email addresses
- Indonesian phone numbers
- possible Indonesian NIK / 16-digit identifiers
- possible NPWP identifiers
- payment card numbers that pass Luhn validation

Critical findings are quarantined and blocked from training approval.

## API Examples

Submit web/app data:

```bash
curl -X POST http://localhost:8000/v1/learning/intake \
  -H "Content-Type: application/json" \
  -H "X-Siger-API-Key: change-this-key" \
  -d '{
    "source_type": "app",
    "text": "User asked how to configure Laravel queues.",
    "app_id": "mobile-app",
    "session_id": "s1",
    "language": "id",
    "domain": "laravel",
    "consent": true,
    "allow_training": true,
    "metadata": {"screen": "support_chat"}
  }'
```

If the payload contains a credential such as `api_key=abc123secret`, the stored payload will contain `<redacted:credential_assignment>` and the row will be quarantined.

Approve a safe candidate:

```bash
curl -X POST http://localhost:8000/v1/learning/approval \
  -H "Content-Type: application/json" \
  -H "X-Siger-API-Key: change-this-key" \
  -d '{"intake_id":"...","reviewer":"admin","decision":"approve","note":"safe curated row"}'
```

Check stats:

```bash
curl http://localhost:8000/v1/learning/stats \
  -H "X-Siger-API-Key: change-this-key"
```

Export approved rows:

```powershell
python tools\export_learning_intake.py `
  --input data\intake\approved_training.jsonl `
  --output data\corpus\learning_intake_approved_train.jsonl
```

## Rules

- Do not train on raw web/app data.
- Do not train on data without consent or a valid training purpose.
- Do not approve rows with credentials, tokens, private keys, payment cards, or high-risk identifiers.
- Treat redaction as a safety net, not a substitute for review.
- Keep `data/intake/` local and out of git.
