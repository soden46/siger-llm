# CLAUDE.md

You are working on SIGER_LLM, a custom Python LLM framework.

## Main Task
Read the full codebase, understand the architecture, improve documentation, and fix incomplete or broken code incrementally.

## Rules
- Do not rewrite the whole project.
- Keep existing modular structure.
- Fix bugs with minimal changes.
- Update docs when behavior changes.
- Do not invent missing features.
- If a module is incomplete, mark it clearly and propose a safe implementation.
- Always run or suggest relevant commands after editing.

## Project Structure
- model/: Mamba/SSM model implementation
- tokenizer/: tokenizer implementation
- training/: base model training
- inference/: generator, sampler, chat, API
- lora/: LoRA fine tuning
- evaluation/: benchmarks and metrics
- optimization/: KV cache, ONNX, quantization

## Priority
1. Fix import errors.
2. Make `python main.py` run.
3. Create/fix requirements.txt.
4. Complete docs in docs/.
5. Add missing __init__.py if needed.
6. Add simple tests.
7. Improve error handling.