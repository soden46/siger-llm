
---

## `PROJECT_CONTEXT.md`

```md
# PROJECT_CONTEXT.md

# SIGER_LLM Context

This file exists to help AI assistants and contributors understand the repository quickly.

---

# Project Summary

SIGER_LLM is a custom Python LLM framework designed for experimentation, multilingual language modeling, Indonesian language support, and efficient inference.

The project contains custom implementations for:

- tokenizer
- Siger/SSM model architecture
- training
- inference
- LoRA fine tuning
- evaluation
- optimization

---

# Main Objectives

1. Build a lightweight custom LLM framework.
2. Support multilingual workflows.
3. Improve Indonesian language capability.
4. Support efficient inference.
5. Keep the codebase modular and readable.

---

# Main Architecture

```txt
Dataset
   ↓
Tokenizer
   ↓
Training
   ↓
Checkpoint
   ↓
Evaluation
   ↓
LoRA
   ↓
Optimization
   ↓
Inference/API

Important Modules
tokenizer/

Handles:

encoding
decoding
vocab management
multilingual token handling

Main files:

tokenizer/tokenizer.py
tokenizer/trainer.py
model/

Handles:

Siger model
SSM blocks
sequence modeling

Main files:

model/siger_model.py
model/ssm_block.py
model/ssm_core.py
training/

Handles:

training loop
optimizer
checkpointing
logging

Main files:

training/trainer.py
training/optimizer.py
training/checkpoint.py
inference/

Handles:

generation
streaming
chat sessions
API inference

Main files:

inference/generator.py
inference/chat.py
inference/api.py
lora/

Handles:

LoRA injection
adapter training
adapter merging

Main files:

lora/layer.py
lora/model.py
lora/trainer.py
evaluation/

Handles:

perplexity
MMLU
ARC
BLEU
ROUGE
Indo benchmarks
optimization/

Handles:

kv cache
quantization
ONNX export
benchmarking
Main Dependencies

Expected dependencies:

torch
numpy
fastapi
uvicorn
pydantic
datasets
tqdm
onnx
onnxruntime
Important Notes
This project is experimental.
The architecture may evolve frequently.
Some modules may still be incomplete.
Prefer incremental improvements.
Do not rewrite the architecture drastically.
Current Priorities
stabilize setup
improve docs
improve training stability
improve inference speed
improve evaluation coverage
improve optimization tools
AI Assistant Instructions

When working on this repository:

inspect related files before editing
avoid hallucinating missing features
keep changes modular
explain changes clearly
update docs when necessary

If something is unclear:

Need confirmation from project owner.