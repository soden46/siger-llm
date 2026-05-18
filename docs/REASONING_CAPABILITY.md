# Reasoning Capability Pack

Reasoning di SigerLM ditanam sebagai **instruction/LoRA capability**, bukan dengan mengubah core architecture.

## Goals

- Melatih jawaban runtut dengan format `<thought>...</thought>` saat analisis diperlukan.
- Mengajarkan pemilihan konteks dialek Lampung O/Nyo.
- Mengajarkan penjelasan struktur tata bahasa Lampung.
- Mengajarkan logic, security reasoning, dan software design reasoning.
- Menjaga core model tetap general.

## Files

```txt
tools/build_reasoning_seed.py
data/capabilities/reasoning_cot_seed.jsonl
configs/datasets/reasoning_instruction.json
configs/datasets/indonesian_hf_mix_plus_kaggle_reasoning.json
configs/training/reasoning_lora.json
configs/training/indonesian_hf_mix_plus_kaggle_reasoning_lora.json
```

## Chain-of-Thought Format

Preferred output format:

```txt
<thought> alasan runtut dan analisis konteks </thought> jawaban akhir
```

Contoh:

```json
{
  "instruction": "Terjemahkan ke Dialek O: Kamu mau pergi ke mana?",
  "output": "<thought> Pertanyaan meminta Dialek O. Pilih subjek, kata mau, verba pergi, dan bentuk tanya ke mana. </thought> Niku ago mit jow?"
}
```

## Generator Support

`inference/generator.py` now has thought-aware stopping:

- default `preserve_thought=True`
- if an EOS/stop token appears while `<thought>` is still open, generation continues
- generation can stop normally after `</thought>` appears

This helps avoid truncated reasoning during inference.

## Activation

`SigerConfig.activation` defaults to `silu`, matching the Mamba-style block already used by SigerLM.

Supported values:

```txt
silu
swish
gelu
relu
```

Keep `silu` as the default for reasoning experiments.

## Build Commands

Reasoning-only:

```bash
python tools/build_reasoning_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/reasoning_instruction.json
python lora/run_lora.py --config configs/training/reasoning_lora.json
```

Full Indonesian + Kaggle + Lampung + reasoning + software engineering:

```bash
python tools/build_software_engineering_seed.py
python tools/build_reasoning_seed.py
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle_reasoning.json
python lora/run_lora.py --config configs/training/indonesian_hf_mix_plus_kaggle_reasoning_lora.json
```

Existing miners and corpus builders can also emit CoT-formatted rows:

```bash
python tools/mine_indonesian_hf_mix.py --max-items-per-source 10000 --cot-ratio 0.25
python tools/mine_general_assistant_data.py --preset all --max-items 1000 --cot-ratio 0.25
python tools/build_instruction_corpus.py --registry configs/datasets/indonesian_hf_mix_plus_kaggle.json --cot-ratio 0.25
```

Recommended starting ratio is `0.15` to `0.30`. Keep some normal rows so the model still learns to answer directly when reasoning is not requested.

## Important Notes

- Seed data is intentionally small and should be expanded with many verified examples.
- CoT improves style and reasoning behavior only after the base model has enough language capacity.
- For stronger reasoning, prefer at least the base-size model profile (`d_model=512`, `n_layers=12`) when hardware allows.
- RLHF/RLAIF/GRPO is not implemented yet. Current work is supervised CoT fine-tuning foundation.
