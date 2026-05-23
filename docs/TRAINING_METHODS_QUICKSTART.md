# SigerLM Training Methods Quickstart

This guide lists the practical commands for the current SigerLM training-method
additions.

## 1. Build Trilingual Base Mix

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_base_trilingual_mix.json
```

Target mix:

- 30% Indonesian general
- 25% English general
- 20% Lampung O
- 20% coding / technical docs
- 3% math / reasoning
- 2% safety / refusal

Check the generated report before training:

```powershell
Get-Content data\corpus\siger_base_trilingual_mix_train.report.json
```

If many groups are capped by `max_oversample_factor`, add more data before a
serious base run.

## 2. Build Lampung Identity Continuation Mix

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_continue_lampung_identity.json
```

This builds a Lampung O + Indonesian continuation corpus.

## 3. Train or Refresh Tokenizer

For a new base model, train the 32k tokenizer:

```powershell
python tokenizer\train_hf_bpe.py --vocab-size 32000
```

Do this before training a new checkpoint that should understand the new
language/domain tags as special tokens.

## 4. Generate Preference Pairs

```powershell
python tools\mine_preference_pairs.py `
  --input data\corpus\lampung_instruction_train.jsonl `
  --output data\corpus\preference_pairs_lampung_train.jsonl `
  --strategies shorter_vs_complete language_correct_vs_wrong `
  --sample 0.5
```

Preference mining is a starting point. Review samples before using them for a
serious DPO run.

## 5. Train DPO LoRA

```powershell
python lora\dpo.py --config configs\training\lampung_dpo.json
python lora\dpo.py --config configs\training\general_dpo.json
```

The DPO trainer is SigerLM-native and targets SSM LoRA modules:

- `in_proj`
- `out_proj`
- `x_proj`
- `dt_proj`

## 6. Run Safety and Router Harness

```powershell
python evaluation\run_harness.py `
  --config configs\evaluation\harness_trilingual.json `
  --allow-missing-model `
  --only router_language_detection router_domain_detection safety_multilingual `
  --no-fail
```

This checks static language/domain detection and the safety guardrail without
requiring a loaded model.

## 7. Run Existing Smoke Checks

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_smoke.json --only dataset_fixture_audit --no-fail
python train_pipeline.py --mode lora-curriculum --dry-run
```

## Current Notes

- The guardrail is active at router level.
- The model still needs training data for refusal behavior to internalize it.
- QLoRA is not implemented yet.
- Generated large corpus JSONL files should not be committed unless explicitly
  required. Commit configs and reports instead.
