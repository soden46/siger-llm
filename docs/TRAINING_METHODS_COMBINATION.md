# SigerLM Training Methods Combination

This document describes how SigerLM should combine modern LLM training methods
without breaking its lightweight SSM-first design.

## Current Foundation

SigerLM already has:

- CLM-style language modeling for the base model.
- LoRA instruction tuning.
- Dataset registry and corpus builders.
- Lampung O lookup/retrieval plus model fallback.
- Curriculum LoRA pipeline.
- Evaluation harness for dataset fixtures, router behavior, and Lampung lookup.

Recent additions:

- SigerLM-native DPO LoRA trainer.
- Preference-pair mining utility.
- Router-level safety and critical-realism guardrail.
- Feedback collector and continual-learning scheduler skeletons.
- Trilingual/domain data mix registry.
- Static trilingual/domain/safety harness.

## Recommended Training Stack

### 1. Tokenizer Training

Use a ByteLevel BPE tokenizer with special tokens for chat, language, and domain
tags. For the target 30M+ model range, use:

```powershell
python tokenizer\train_hf_bpe.py --vocab-size 32000
```

The tokenizer should preserve:

- Indonesian and English Latin text.
- Lampung O words.
- code symbols such as `->`, `::`, `$`, `{}`, `[]`, `()`, and `;`.
- chat markers and domain tags.

### 2. Base Pretraining

Build the target mix:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_base_trilingual_mix.json
```

Target composition:

- 30% Indonesian general
- 25% English general
- 20% Lampung O
- 20% coding / technical docs
- 3% math / reasoning
- 2% safety / refusal

This keeps the model general while making Indonesian, English, Lampung O, and
technical text first-class parts of the base model.

### 3. Continue Pretraining for Local Identity

After the base stage, strengthen the local identity with Lampung O and
Indonesian:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_continue_lampung_identity.json
```

Current identity mix:

- 60% Lampung O
- 40% Indonesian

### 4. Instruction Tuning

Use LoRA SFT for multilingual assistant behavior. Keep the core model general;
domain-specific behavior should come from tags, retrieval, and adapters.

### 5. Domain LoRA Experts

Train separate LoRA experts for:

- Lampung O translator and conversation.
- coding and debugging.
- Laravel/PHP.
- math/reasoning.
- safety/refusal tone.

This is safer than forcing every skill into one small base model.

### 6. Preference Alignment with DPO

Use DPO after SFT when there are preference pairs:

```powershell
python lora\dpo.py --config configs\training\lampung_dpo.json
python lora\dpo.py --config configs\training\general_dpo.json
```

DPO is useful for:

- choosing complete answers over incomplete ones,
- preferring safe refusal over unsafe compliance,
- improving translation quality,
- reducing overly rigid responses.

Do not trust weak mined preferences blindly. Review and benchmark.

### 7. Retrieval and RAG

Use retrieval for facts and domain knowledge:

- Lampung dictionary and grammar rules.
- Laravel docs.
- project docs.
- safety policy snippets.

The base model should learn language patterns. Retrieval should provide current
or domain-specific facts.

## What Not To Do

- Do not add Lampung-specific logic to `model/`.
- Do not train a tiny base model with too much niche data and then expect strong
  general ability.
- Do not over-duplicate a tiny English/code/safety corpus just to satisfy a
  percentage target.
- Do not enable automatic continual learning without human review and harness
  regression.
- Do not claim DPO improvement without comparing against SFT on held-out tests.

## Evaluation Gates

Before promoting a model, run:

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_trilingual.json --allow-missing-model --only router_language_detection router_domain_detection safety_multilingual --no-fail
python evaluation\run_harness.py --config configs\evaluation\harness_smoke.json --only dataset_fixture_audit --no-fail
```

For model-backed evaluation, add checkpoint-specific runs for generation,
Lampung translation, coding, math, and safety refusal quality.

## Current Data Gap

The builder now encodes the target mix, but local data does not yet fully meet
it. Indonesian and Lampung O have useful volume. English, code, math, and safety
need more curated data before serious base pretraining.
