# Evaluation

Evaluation is currently split into general LM evaluation scaffolding and Lampung domain smoke/eval tools.

## General Evaluation

Existing modules:

```txt
evaluation/perplexity.py
evaluation/generation.py
evaluation/benchmarks.py
evaluation/indo_eval.py
evaluation/runner.py
evaluation/run_eval.py
```

These are intended for perplexity, generation quality, Indo tasks, and benchmark wrappers. Some parts are still scaffolding and should be verified before being treated as final metrics.

## Engineering Harness

The engineering harness is a config-driven layer around existing inference and evaluation pieces. It does not change the model core. It can audit datasets without loading a model, or load one checkpoint and run generation/router/Lampung regression suites.

Files:

```txt
evaluation/harness/
evaluation/run_harness.py
configs/evaluation/harness_smoke.json
configs/evaluation/harness_dataset_only.json
data/eval/harness/*.jsonl
```

Dataset-only audit:

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_smoke.json --only dataset_fixture_audit
```

Checkpoint-backed smoke:

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_smoke.json --checkpoint checkpoints\lora\model_general_merged.pt --device auto
```

For Kaggle/code-corpus work, point `configs/evaluation/harness_dataset_only.json` at the generated JSONL corpus or raw converted source JSONL, then run:

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_dataset_only.json --no-fail
```

Each run writes:

```txt
logs/eval/harness/<timestamp>_<name>/report.json
logs/eval/harness/<timestamp>_<name>/summary.md
```

Supported suite kinds:

- `dataset_audit`: JSONL validity, required fields, rough length, duplicate fingerprints, optional metadata license check.
- `generation`: direct `Generator.generate(...)` smoke cases.
- `router`: `SigerRouter` route and output regression cases.
- `lampung_lookup`: lookup-first Lampung translation regression cases.

## Lampung Lookup Evaluation

Recent Lampung work added lookup-first eval helpers:

```txt
evaluation/lampung_lookup_eval.py
evaluation/run_lampung_lookup_eval.py
evaluation/lampung_reasoning.py
```

The runtime pipeline uses:

```txt
retrieval/instruction_lookup.py
retrieval/compositional_translator.py
inference/lampung_pipeline.py
```

## Latest Smoke Results

CLI auto route:

```txt
Input: Nyak haga mengan manuk di warung paghek jalan
Output: aku mau makan ayam di warung dekat jalan
Route: lampung_to_id
Source: exact instruction lookup
```

Lampung O -> English:

```txt
Nyak haga mengan manuk di warung paghek jalan
-> i want to eat chicken at the stall near the road

Nyak ago belei buku di pasar
-> i want to buy a book at the market
```

## What Still Needs Evaluation

- Lampung O -> Indonesia exact/semantic accuracy
- Indonesia -> Lampung O semantic accuracy
- Lampung O -> English accuracy
- router false positives/false negatives
- general chat after `general_lora.json` training
- hallucination and refusal behavior
- CPU inference speed

## Suggested Commands

Compile checks:

```powershell
python -m py_compile evaluation\lampung_lookup_eval.py evaluation\run_lampung_lookup_eval.py inference\router.py inference\lampung_pipeline.py
```

CLI smoke:

```powershell
@'
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```
