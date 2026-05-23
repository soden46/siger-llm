# Strategic Roadmap: 8 SigerLM Targets

This roadmap maps the product goals to concrete SigerLM subsystems while keeping
the core model general, lightweight, and CPU/VPS-friendly.

## Targets

1. Very capable general assistant.
2. Easy to train and adapt.
3. Can learn from reviewed user feedback.
4. Token-efficient.
5. Fast response on modest hardware.
6. Can learn from external data through curated ingestion and retrieval.
7. Flexible and not rigid.
8. Can refuse harmful or unrealistic requests gently, with reasons.

## Design Principles

- Keep `model/` general. Do not hardcode Lampung, Laravel, or safety logic into
  the SSM backbone.
- Use dataset registries and training configs for behavior changes.
- Keep Lampung O as part of tokenizer and base data, then strengthen it with
  retrieval and LoRA experts.
- Use LoRA experts for heavier skills: coding, Laravel, math, debugging,
  translation, and safety.
- Use RAG/retrieval for facts and domain documents that should not be memorized.
- Validate every domain with harness cases before promoting a model.

## Target Architecture

```text
SIGER-Core
  SSM/Mamba-like trilingual backbone

SIGER-Tokenizer
  32k ByteLevel BPE for Indonesian, English, Lampung O, and code

SIGER-Router
  language detection + domain detection + safety guardrail

SIGER-LoRA Experts
  coding, Laravel, math, Lampung translator, debugging, safety

SIGER-RAG/Retrieval
  Lampung dictionary, grammar rules, Laravel docs, project docs, factual sources

SIGER-Harness
  language, domain, translation, coding, math, and safety regression checks
```

## Roadmap

### Phase 1: Base and Tokenizer

- Train or refresh tokenizer with trilingual and code-aware data.
- Build base corpus with the current target mix:
  - 30% Indonesian general
  - 25% English general
  - 20% Lampung O
  - 20% coding / technical docs
  - 3% math / reasoning
  - 2% safety / refusal
- Pretrain `SIGER-Base-Trilingual` with CLM.
- Continue pretrain on Lampung O + Indonesian identity data.

### Phase 2: Assistant Behavior

- Run multilingual instruction tuning.
- Keep responses concise, accurate, and language-matched.
- Use domain tags in instruction rows:
  - `<|lang:id|>`, `<|lang:en|>`, `<|lang:lampung_o|>`, `<|lang:mixed|>`
  - `<|domain:general|>`, `<|domain:code|>`, `<|domain:translation|>`,
    `<|domain:math|>`, `<|domain:safety|>`, and related expert tags.

### Phase 3: Domain Experts

- Train LoRA experts for:
  - Lampung O translation and conversation
  - coding and debugging
  - Laravel/PHP technical tasks
  - math/reasoning
  - safety/refusal style
- Prefer LoRA experts and retrieval over forcing every skill into the small
  base model.

### Phase 4: Alignment and Feedback

- Use DPO on reviewed preference pairs.
- Mine weak preference pairs only as a starting point; curate high-risk domains.
- Collect feedback to JSONL, review it, then batch into DPO/SFT data.
- Keep automatic continual learning behind manual approval until benchmarked.

### Phase 5: Speed and Deployment

- Keep CPU/VPS constraints in mind.
- Add quantization only after compatibility tests.
- Consider QLoRA for training when a stable quantized backend is selected.
- Use streaming generation and retrieval caches for perceived latency.

## Harness Requirements

Minimum suites:

- `id_general_qa`
- `en_general_qa`
- `lampung_o_translation`
- `lampung_o_conversation`
- `id_en_translation`
- `en_id_translation`
- `code_basic`
- `math_basic`
- `router_language_detection`
- `router_domain_detection`
- `safety_multilingual`

Smoke target: 100 to 300 cases per important language/domain.

Regression target: 1000+ cases per critical domain once data is available.

## Current Risk

The pipeline supports the target design, but current local data is still
imbalanced. Indonesian and Lampung O are well represented; English, code, math,
and safety need more curated rows before serious base pretraining.
