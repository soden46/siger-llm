# SigerLM Expertise Curriculum

This curriculum separates training material by expertise domain while still
allowing one connected neural model through sequential LoRA merges.

## Domain Split

| Domain | Dataset registry | Output corpus | Merged checkpoint |
|---|---|---|---|
| Indonesian | `configs/datasets/expertise_indonesian.json` | `data/corpus/expertise_indonesian_train.jsonl` | `checkpoints/lora/model_expertise_indonesian_merged.pt` |
| Lampung | `configs/datasets/expertise_lampung.json` | `data/corpus/expertise_lampung_train.jsonl` | `checkpoints/lora/model_expertise_lampung_merged.pt` |
| General knowledge | `configs/datasets/expertise_general_knowledge.json` | `data/corpus/expertise_general_knowledge_train.jsonl` | `checkpoints/lora/model_expertise_general_knowledge_merged.pt` |
| Reasoning | `configs/datasets/expertise_reasoning.json` | `data/corpus/expertise_reasoning_train.jsonl` | `checkpoints/lora/model_expertise_reasoning_merged.pt` |
| Programming basic | `configs/datasets/expertise_programming_basic.json` | `data/corpus/expertise_programming_basic_train.jsonl` | `checkpoints/lora/model_expertise_programming_basic_merged.pt` |
| Programming intermediate | `configs/datasets/expertise_programming_intermediate.json` | `data/corpus/expertise_programming_intermediate_train.jsonl` | `checkpoints/lora/model_expertise_programming_intermediate_merged.pt` |
| Programming expert | `configs/datasets/expertise_programming_expert.json` | `data/corpus/expertise_programming_expert_train.jsonl` | `checkpoints/lora/model_expertise_programming_expert_merged.pt` |
| Neural mixer | `configs/datasets/expertise_neural_mixer.json` | `data/corpus/expertise_neural_mixer_train.jsonl` | `checkpoints/lora/model_expertise_neural_mixer_merged.pt` |

## Programming Dataset Paths

The programming registries expect normalized local JSONL files. Use this shape:

```json
{"instruction":"...","input":"","output":"...","system":"optional","source":"...","type":"..."}
```

Expected paths:

```txt
data/raw/code/mit/python_codes_25k/train.jsonl
data/raw/code/mit/code_instructions_120k/train.jsonl
data/raw/code/apache_2_0/python_code_explainer/train.jsonl
data/raw/code/apache_2_0/codefeedback_filtered_instruction/train.jsonl
data/raw/code/apache_2_0/mbpp/train.jsonl
data/raw/code/mit/python_state_changes/train.jsonl
data/raw/code/apache_2_0/arxiv_code_instructions_34k/train.jsonl
```

`FreedomIntelligence/evol-instruct-indonesian` is kept out of the commercial-safe
expertise chain. If needed for experiments only, normalize code slices into:

```txt
data/raw/instruction/noncommercial/evol_instruct_indonesian_code_slices.jsonl
```

and build:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\expertise_programming_indonesian_experimental.json
```

## Full Expertise Curriculum

Run the separated expertise curriculum:

```powershell
python train_pipeline.py --mode lora-curriculum --lora-curriculum-config configs\training\lora_expertise_curriculum.json
```

Use `--dry-run` first on Kaggle:

```powershell
python train_pipeline.py --mode lora-curriculum --lora-curriculum-config configs\training\lora_expertise_curriculum.json --dry-run
```

Final integrated checkpoint:

```txt
checkpoints/lora/model_expertise_neural_mixer_merged.pt
```

## Runtime General Expertise

Training separation alone is not enough. Inference also has a general
expertise coordinator:

```txt
user task
  -> ExpertiseOrchestrator
  -> task summary
  -> domain drafts: Lampung / Indonesian / reasoning / programming basic / intermediate / expert
  -> synthesized final answer
```

CLI:

```powershell
python chat_cli.py --mode expertise --prompt "Buat blueprint REST API todo dengan FastAPI, PostgreSQL, Docker, test, logging JSON, dan COMPLIANCE.md."
```

Code:

```python
from inference.expertise_router import ExpertiseOrchestrator

expertise = ExpertiseOrchestrator(gen, lampung)
response = expertise.route(user_prompt)
print(response.domains)
print(response.text)
```

## Extending New Domains

SigerLM is designed to grow by adding new expertise domains instead of trying to
learn every topic in the first training run.

For a new domain, add:

```txt
configs/datasets/expertise_<domain>.json
configs/training/expertise_<domain>_lora.json
```

Recommended flow:

```txt
current neural mixer checkpoint
  -> new domain LoRA
  -> new merged domain checkpoint
  -> neural mixer v2 with old domains + new domain
```

Example domains:

```txt
expertise_medical
expertise_legal
expertise_agriculture
expertise_cybersecurity
expertise_finance
expertise_regional_language_<name>
```

Keep the core model general. Domain specificity should enter through dataset
registries, LoRA adapters, retrieval tools, and the `ExpertiseOrchestrator`.

## Feedback Repair Loop

User feedback can become training data, but it should not be trained online
immediately. Store it, audit it, and periodically fine-tune a repair adapter.

Preferred feedback row:

```json
{
  "instruction": "original user request",
  "input": "optional previous model answer or context",
  "output": "curated corrected answer",
  "system": "Kamu adalah SigerLM, asisten yang memperbaiki jawaban berdasarkan feedback user.",
  "source": "user_feedback",
  "type": "feedback_repair"
}
```

Safe feedback pipeline:

```txt
user prompt + model answer + user correction
  -> raw feedback store
  -> dedupe
  -> toxicity/noise filtering
  -> domain tagging
  -> human or rule audit
  -> feedback repair JSONL
  -> periodic LoRA repair run
  -> regression tests
  -> mixer refresh
```

Do not train directly on every feedback item. Unfiltered feedback can teach the
model wrong facts, bad style, private data, or prompt-injection behavior.

## Single-Domain Runs

Build one domain corpus:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\expertise_lampung.json
```

Train one domain adapter:

```powershell
python lora\run_lora.py --config configs\training\expertise_lampung_lora.json
```

This keeps each expertise inspectable while still allowing the full curriculum
to connect them into one model.
