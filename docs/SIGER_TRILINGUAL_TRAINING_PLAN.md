# SIGER Trilingual Training Plan

This is the training contract for the Indonesia + English + Lampung O direction.

## Target Data Mix

For `SIGER-Base-Trilingual`, use:

| Group | Target |
|---|---:|
| Indonesian general | 30% |
| English general | 25% |
| Lampung O | 20% |
| Coding / technical docs | 20% |
| Math / reasoning | 3% |
| Safety / refusal | 2% |

The executable registry is:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_base_trilingual_mix.json
```

This keeps the base model general, but gives four strong pillars: Indonesian
general, English general, Lampung O, and coding/technical data. Math/reasoning
and safety stay present as small stabilizers; deeper skill should come from
domain LoRA experts and harnessed DPO/SFT stages.

The builder writes a quality report beside the corpus. If a group does not have
enough rows, oversampling is capped by `max_oversample_factor` so a tiny domain
does not poison the base model with thousands of duplicates.

## Training Order

1. Train tokenizer with a 32k ByteLevel BPE:

```powershell
python tokenizer\train_hf_bpe.py --vocab-size 32000
```

2. Pretrain `SIGER-Base-Trilingual` on the mixed corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_base_trilingual_mix.json
```

3. Continue pretrain / adapt identity on Lampung O + Indonesian:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\siger_continue_lampung_identity.json
```

4. Instruction tune multilingual assistant behavior with the normal LoRA
curriculum.

5. Train domain LoRA experts separately:

- coding / technical docs
- Laravel
- math / reasoning
- Lampung translator
- debugging
- safety / refusal

6. Run the trilingual/domain harness:

```powershell
python evaluation\run_harness.py --config configs\evaluation\harness_trilingual.json --allow-missing-model --no-fail
```

## Tags

Rows built through the trilingual registries are prefixed with:

```text
<|lang:id|>
<|lang:en|>
<|lang:lampung_o|>
<|lang:mixed|>

<|domain:general|>
<|domain:code|>
<|domain:math|>
<|domain:laravel|>
<|domain:debug|>
<|domain:safety|>
<|domain:translation|>
```

These tags are registered as tokenizer special tokens. Once a tokenizer and
checkpoint are trained with them, keep the IDs stable.

## Current Data Gap

The local repository currently has strong Indonesian and Lampung O coverage, but
English, code, math, and safety are still under-supplied for the 30/25/20/15/5/5
target. The registry is ready, but the quality report must be checked before a
serious base run.
