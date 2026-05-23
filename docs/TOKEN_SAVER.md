# SigerLM Token Saver

SigerLM Token Saver is an optional context layer for compressing verbose tool output before it reaches chat memory or the model prompt. It is inspired by RTK/9Router-style tool-result compression, but it stays outside the core model so it does not damage SigerLM's base language ability.

## Where It Fits

```txt
tool output
  -> ToolResultCompressor
  -> chat memory / retrieval context
  -> router / prompt builder
  -> SigerLM
```

This means the model still receives normal text during training and inference. Only noisy context such as logs, diffs, grep results, directory listings, and install output is compacted.

## Supported Output

- `git status`: branch, staged/unstaged files, untracked files
- `git diff`: changed files, add/remove counts, hunk headers, important changed lines
- `rg` / `grep`: file, line number, matched text, capped per file
- `ls` / `tree` / `find`: compact directory and file map
- test logs and stack traces: failures, errors, assertions, traceback lines, tail
- package logs: error, warning, failed, missing, conflict, tail

Unknown output falls back to a conservative head/signal/tail summary only when it is long enough to justify compression.

## Safety Rules

- Keep this layer optional and outside `model/`, `training/`, and tokenizer training.
- Do not compress normal user messages by default.
- Do not treat compressed output as ground-truth training data without human approval.
- If the compressed result is larger than the original, the compressor stores the original.
- Every compressed block includes metadata: mode, original chars, compressed chars, saved ratio, and `lossless=false`.

## CLI Smoke

```powershell
git diff | python tools\compress_tool_result.py --command "git diff" --stats
rg -n "TODO|FIXME" . | python tools\compress_tool_result.py --command "rg TODO FIXME" --stats
```

## Python Usage

```python
from inference.tool_result_compressor import compress_tool_result

result = compress_tool_result(long_output, command="git diff")
print(result.text)
print(result.metadata())
```

For chat sessions:

```python
session.add_tool_result(output=long_output, command="pytest -q")
```

The compressed text is stored as retrievable memory, so the prompt remains smaller and cleaner.
