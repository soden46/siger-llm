"""Print a concise summary of the SigerLM training-method roadmap.

This file is intentionally plain ASCII so it is safe to view in PowerShell,
terminals with legacy encodings, and GitHub diffs.
"""

SUMMARY = """
SigerLM Training Methods Summary
================================

Current stable base:
- SSM/Mamba-like language model backbone.
- LoRA instruction tuning.
- Dataset registry and corpus builder.
- Lampung O retrieval and lookup-first inference path.
- Evaluation harness for data and router smoke checks.

Recent additions:
- SigerLM-native DPO LoRA trainer.
- Preference-pair mining utility.
- Router-level safety and critical-realism guardrail.
- Feedback collector and continual-learning scheduler skeletons.
- Trilingual/domain data-mix registry.
- Static trilingual/domain/safety harness.

Recommended sequence:
1. Train or refresh a 32k ByteLevel BPE tokenizer.
2. Pretrain SIGER-Base-Trilingual on:
   - 30% Indonesian general
   - 25% English general
   - 20% Lampung O
   - 20% coding / technical docs
   - 3% math / reasoning
   - 2% safety / refusal
3. Continue pretrain on Lampung O + Indonesian identity data.
4. Run multilingual instruction tuning.
5. Train LoRA experts for Lampung, coding, Laravel, math, debugging, and safety.
6. Use DPO only after preference data is reviewed and benchmarked.
7. Validate every promoted checkpoint with harness suites.

Important commands:
- python tokenizer\\train_hf_bpe.py --vocab-size 32000
- python tools\\build_instruction_corpus.py --registry configs\\datasets\\siger_base_trilingual_mix.json
- python tools\\build_instruction_corpus.py --registry configs\\datasets\\siger_continue_lampung_identity.json
- python lora\\dpo.py --config configs\\training\\lampung_dpo.json
- python evaluation\\run_harness.py --config configs\\evaluation\\harness_trilingual.json --allow-missing-model --only router_language_detection router_domain_detection safety_multilingual --no-fail

Known gaps:
- English, code, math, and safety corpora need more curated rows.
- DPO has smoke tests, but still needs real checkpoint benchmarks.
- QLoRA is not implemented yet.
- Continual learning should require human approval until the full regression
  harness is reliable.
"""


def main() -> None:
    print(SUMMARY.strip())


if __name__ == "__main__":
    main()
