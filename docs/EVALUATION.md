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
