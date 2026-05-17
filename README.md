# SigerLM

SigerLM is an experimental custom LLM framework built with a State Space Model/Mamba-like architecture. The project target is a lightweight general LM that can be trained and deployed on modest hardware. Bahasa Lampung is currently used as a domain adapter and training testbed.

## Status

Experimental. The architecture, datasets, and training recipes are still changing quickly.

## What Exists Now

- Custom SSM/Mamba-like language model in PyTorch
- Hybrid tokenizer selector: custom HF ByteLevel BPE if available, fallback tokenizer otherwise
- Base training pipeline
- Custom LoRA implementation
- Config-driven LoRA runner
- Unified instruction corpus builder
- General/domain inference router
- Lampung lookup-first translation pipeline
- Lampung O/Nyo dataset pipeline
- PDF extraction tools for Lampung sources
- Evaluation and optimization scaffolding

## Architecture

```txt
raw data / domain data
  -> extraction or dataset registry
  -> unified instruction corpus
  -> tokenizer
  -> base training or LoRA
  -> merged checkpoint
  -> inference router
  -> general chat or domain tools
```

The core model is general. Lampung-specific behavior lives in:

```txt
data/lampung/
tools/build_lampung_dataset.py
tools/build_instruction_dataset.py
retrieval/
inference/lampung_pipeline.py
```

Routing between general chat and Lampung tools lives in:

```txt
inference/router.py
chat_cli.py
```

## Key Directories

```txt
config/       model configuration
model/        SigerLM and SSM blocks
tokenizer/    tokenizer wrappers and trainer
training/     base training and dataset registry
lora/         LoRA config, dataset, trainer, runner
inference/    generator, chat session, router, domain pipelines
retrieval/    Lampung lookup/rules/lexicon
tools/        dataset builders and extractors
configs/      dataset registries and training configs
data/         local corpora and generated datasets
evaluation/   evaluation scaffolding
optimization/ CPU, ONNX, quantization, cache experiments
```

## Latest Verified Data State

```txt
data/lampung/processed/percakapan_1000_pairs.jsonl: 3100 rows
data/lampung/processed/compositional_pairs.jsonl: 1968 rows
data/lampung/final/train.jsonl: 4325 rows
data/lampung/final/valid.jsonl: 541 rows
data/lampung/final/test.jsonl: 541 rows
data/lampung/final/train_augmented_instruction.jsonl: 32059 rows
data/corpus/lampung_instruction_train.jsonl: 30701 rows
data/corpus/general_instruction_train.jsonl: 30704 rows
```

`general_instruction_train.jsonl` is not yet a strong general-chat corpus. It is mostly Lampung plus small local Indonesian/English/code text files. Add larger sources to `configs/datasets/general_instruction.json` before expecting broad chatbot behavior.

## Dataset Pipeline

Build Lampung domain data:

```powershell
python tools\extract_percakapan_pdf.py
python tools\build_compositional_lampung_dataset.py
python tools\build_lampung_dataset.py
python tools\build_instruction_dataset.py
```

Build unified instruction corpora:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
```

Dataset registry source formats:

- `instruction_jsonl`
- `chat_jsonl`
- `text_completion`

Preferred instruction row:

```json
{"instruction":"Terjemahkan Lampung O ke Bahasa Indonesia","input":"api kabar niku","output":"apa kabar kamu","source":"manual","type":"translation"}
```

Preferred chat row:

```json
{"messages":[{"role":"user","content":"Apa itu AI?"},{"role":"assistant","content":"AI adalah teknologi yang membuat komputer dapat melakukan tugas yang biasanya membutuhkan kecerdasan manusia."}]}
```

## Training

Lampung-only LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\lampung_instruction.json
python lora\run_lora.py --config configs\training\lampung_lora.json
```

General LoRA:

```powershell
python tools\build_instruction_corpus.py --registry configs\datasets\general_instruction.json
python lora\run_lora.py --config configs\training\general_lora.json
```

Default fallback:

```powershell
python lora\run_lora.py
```

The default remains Lampung-safe for backward compatibility.

## Inference CLI

```powershell
python chat_cli.py
```

Modes:

```txt
0 auto/general router
1 Lampung O -> Indonesia
2 Indonesia -> Lampung O
3 Lampung O -> English
4 Lampung reasoning
5 general chat
6 Lampung word order
```

Latest smoke:

```txt
Mode: 0
Input: Nyak haga mengan manuk di warung paghek jalan
Assistant: aku mau makan ayam di warung dekat jalan
Route: lampung_to_id
Source: exact instruction lookup
```

Lampung O -> English examples:

```txt
Nyak haga mengan manuk di warung paghek jalan
-> i want to eat chicken at the stall near the road

Nyak ago belei buku di pasar
-> i want to buy a book at the market
```

## Development Checks

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py
python -m py_compile training\dataset_registry.py tools\build_instruction_corpus.py lora\config.py lora\dataset.py lora\run_lora.py
python lora\run_lora.py --help
```

## Roadmap

- Expand general instruction/chat corpus
- Add automated tests for corpus builder and router
- Train and evaluate `general_lora.json`
- Add stronger Lampung ID/EN evaluation
- Add native speaker validation workflow for Lampung data
- Improve CPU inference performance
- Export and test ONNX/quantized checkpoints
- Add FastAPI endpoints for router/domain tasks

## License

MIT License.
