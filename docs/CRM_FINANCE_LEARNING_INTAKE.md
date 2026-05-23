# CRM and Household Finance Learning Intake

This guide maps SigerLM learning intake to:

- Krayin CRM custom workflows
- WhatsApp integration through Evolution API
- Chat management data from Chatwoot
- future household finance tracking web/mobile apps

The goal is to learn safe workflow patterns, support templates, and product behavior signals without memorizing customer or family private data.

## Recommended Separation

```txt
Customer-specific facts
  -> tenant-local RAG / memory only
  -> do not train

CRM conversation patterns
  -> anonymize
  -> review
  -> train only generic support behavior

CRM user behavior
  -> aggregate events
  -> train workflow assistance

Household finance
  -> aggregate categories and budgeting patterns
  -> do not train raw transactions
```

## Krayin CRM + Chatwoot Conversations

Use this for anonymized support patterns, not raw customer history.

```json
{
  "source_type": "chatwoot",
  "domain": "crm_chat",
  "learning_mode": "anonymized",
  "text": "Customer asked about order delay. Agent explained SLA and created follow-up task.",
  "app_id": "krayin-crm",
  "session_id": "conversation_123",
  "language": "id",
  "consent": true,
  "allow_training": true,
  "metadata": {
    "provider": "chatwoot",
    "anonymized": true,
    "tenant_id": "tenant-redacted",
    "conversation_status": "resolved",
    "training_scope": "support_pattern"
  }
}
```

Avoid sending raw:

- customer name
- phone / WhatsApp JID
- email
- address
- order IDs tied to a real person
- payment proof
- screenshots
- exact complaint history unless anonymized

## Evolution API WhatsApp Events

Evolution API events often contain phone numbers and WhatsApp JIDs. Send only extracted, anonymized intent summaries for training.

Bad training payload:

```json
{
  "text": "62812xxxx@s.whatsapp.net said: alamat saya ..."
}
```

Better payload:

```json
{
  "source_type": "evolution_api",
  "domain": "crm_chat",
  "learning_mode": "anonymized",
  "text": "A lead asked whether the product can be delivered today. Agent should confirm area, stock, and delivery cutoff time.",
  "consent": true,
  "allow_training": true,
  "metadata": {
    "provider": "evolution_api",
    "anonymized": true,
    "training_scope": "lead_response_template"
  }
}
```

## CRM Behavior Events

Behavior events are safer than raw chat when you strip user and customer identifiers.

```json
{
  "source_type": "crm_behavior",
  "domain": "crm_behavior",
  "learning_mode": "training_candidate",
  "text": "User opened lead detail, changed status to follow-up, then created reminder.",
  "app_id": "krayin-crm",
  "language": "id",
  "consent": true,
  "allow_training": true,
  "metadata": {
    "event_type": "lead_workflow",
    "app_section": "leads",
    "role": "sales",
    "contains_customer_data": false
  }
}
```

Useful behavior signals:

- page or feature used
- workflow step
- anonymized role
- success/failure state
- time-to-complete bucket, not exact timestamps when unnecessary
- validation errors without raw input

Do not send:

- full user ID
- full customer ID
- customer contact fields
- private notes
- raw message body
- precise location

## Household Finance App

Use aggregate-only learning. Do not train raw transactions or family ledgers.

Good:

```json
{
  "source_type": "mobile_app",
  "domain": "household_finance",
  "learning_mode": "aggregate",
  "text": "Monthly budget pattern: groceries exceeded planned budget, transport stayed under budget, user wants weekly reminders.",
  "language": "id",
  "consent": true,
  "allow_training": true,
  "metadata": {
    "aggregate_only": true,
    "period_bucket": "monthly",
    "training_scope": "budgeting_advice_pattern"
  }
}
```

Avoid:

- exact salaries
- exact account balances
- bank account numbers
- payment cards
- lender names tied to family members
- exact transaction timestamps
- merchant names that identify personal routines

## API Flow

Submit:

```bash
curl -X POST http://localhost:8000/v1/learning/intake \
  -H "Content-Type: application/json" \
  -H "X-Siger-API-Key: change-this-key" \
  -d @payload.json
```

Rows with CRM chat or household finance are usually marked `needs_review` unless metadata says they are anonymized or aggregate-only and the privacy scan is clean.

Approve:

```bash
curl -X POST http://localhost:8000/v1/learning/approval \
  -H "Content-Type: application/json" \
  -H "X-Siger-API-Key: change-this-key" \
  -d '{"intake_id":"...","reviewer":"admin","decision":"approve","note":"anonymized pattern only"}'
```

Export:

```powershell
python tools\export_learning_intake.py `
  --input data\intake\approved_training.jsonl `
  --output data\corpus\learning_intake_approved_train.jsonl
```

## Practical Rule

For CRM and finance:

- RAG may use customer-specific or household-specific data locally per tenant/user.
- Training should only use anonymized patterns, templates, aggregates, and reviewed examples.
