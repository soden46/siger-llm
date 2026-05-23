# Implementation Status

This document tracks the current state of the SigerLM training-method and
alignment additions. It is intentionally conservative: a feature is only marked
as ready when it has code, configuration, and at least a smoke check.

## Current Status

| Area | Status | Notes |
|---|---|---|
| SSM backbone | Working | Core model remains general and modality-agnostic. |
| LoRA SFT | Working | Existing config-driven LoRA path is unchanged. |
| DPO LoRA | Implemented, needs experiment | `lora/dpo.py` has a SigerLM-native DPO loop and smoke-tested loss. |
| Preference mining | Implemented, needs curation | `tools/mine_preference_pairs.py` generates weak preference pairs; quality review is required. |
| Feedback collection | Skeleton usable | JSONL feedback collection exists, but production privacy policy is still required. |
| Continual learning scheduler | Skeleton usable | Dry-run and orchestration code exist; automatic deployment should stay manual until validated. |
| Safety guardrail | Working baseline | Router-level soft refusal for illegal, harmful, privacy, discrimination, and unrealistic requests. |
| Trilingual data mix | Configured | Target mix is encoded in `configs/datasets/siger_base_trilingual_mix.json`. |
| Trilingual harness | Working baseline | Static language/domain/safety suites pass without loading a model. |

## Important New Files

- `lora/dpo.py`: DPO training loop for SigerLM LoRA adapters.
- `tools/mine_preference_pairs.py`: Preference-pair mining from instruction rows.
- `inference/constitutional_ai_layer.py`: Lightweight critical/safety guardrail.
- `inference/user_feedback_collector.py`: JSONL feedback capture and batch export.
- `training/continual_learning_scheduler.py`: Optional retraining scheduler.
- `configs/datasets/siger_base_trilingual_mix.json`: Base mix target.
- `configs/datasets/siger_continue_lampung_identity.json`: Lampung O + Indonesian identity continuation mix.
- `configs/evaluation/harness_trilingual.json`: Trilingual router and safety harness.
- `docs/SIGER_TRILINGUAL_TRAINING_PLAN.md`: Current training contract.

## Verified Checks

Run before pushing changes:

```powershell
python -m py_compile chat_cli.py inference\router.py inference\prompt_builder.py inference\constitutional_ai_layer.py inference\user_feedback_collector.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py tools\mine_preference_pairs.py lora\config.py lora\dataset.py lora\dpo.py lora\run_lora.py train_pipeline.py
python -m py_compile evaluation\run_harness.py evaluation\harness\checks.py evaluation\harness\runner.py tokenizer\hf_tokenizer.py tokenizer\hybrid_tokenizer.py tokenizer\trainer.py tokenizer\special_tokens.py
python evaluation\run_harness.py --config configs\evaluation\harness_trilingual.json --allow-missing-model --only router_language_detection router_domain_detection safety_multilingual --no-fail
python evaluation\run_harness.py --config configs\evaluation\harness_smoke.json --only dataset_fixture_audit --no-fail
python train_pipeline.py --mode lora-curriculum --dry-run
```

## Known Gaps

- English, code, math, and safety data are still under-supplied for the target
  base mix. The builder caps oversampling to avoid duplicate-heavy training.
- DPO is smoke-tested, but not yet benchmarked against SFT on a real checkpoint.
- The safety layer is a router-level guardrail. For stronger behavior, train a
  safety/refusal LoRA or DPO adapter from curated preference data.
- QLoRA is not implemented yet. It should be added only after selecting a
  quantization backend that works on the target deployment environment.

## Push Guidance

Commit code, configs, small harness fixtures, reports, and docs. Do not commit
generated large JSONL corpora unless explicitly needed for release packaging.
