# Inference

SigerLM inference has three layers:

1. Raw generation/chat with `Generator` and `ChatSession`.
2. Routed inference with `SigerRouter`, which can call general chat or Lampung domain tools.
3. Expertise orchestration with `ExpertiseOrchestrator`, which decomposes a user task into domain expertise drafts and synthesizes a final answer.

## Generator

```python
from inference.generator import Generator
from tokenizer.hybrid_tokenizer import build_tokenizer

tok = build_tokenizer("auto")
gen = Generator(model, tok, device="cpu")

output = gen.generate(
    "Jelaskan apa itu machine learning:",
    max_new_tokens=120,
    temperature=0.3,
    top_k=20,
    top_p=0.8,
)
```

## ChatSession

```python
from inference.chat import ChatSession

chat = ChatSession(
    gen,
    max_context_tokens=1024,
    retrieval_top_k=5,
    retrieval_token_budget=360,
    long_input_threshold_chars=1200,
)
reply = chat.chat("Apa itu AI?", max_new_tokens=80)
```

## Long-Context Memory

SigerLM does not need to put every token from a large document into the prompt.
Long context is handled with a retrieval budget:

```txt
large document or long user message
  -> chunk into overlapping word windows
  -> store in SessionMemory
  -> retrieve top-k relevant chunks for each question
  -> fit retrieved chunks, summaries, recent turns, and user prompt into max_context_tokens
```

This gives "large context" behavior while keeping the actual model prompt small.
It is the preferred path for CPU/VPS and small SSM checkpoints.

Load a document in Python:

```python
chat.add_document(
    long_text,
    metadata={"source": "project_spec.md", "title": "Project Spec"},
)
reply = chat.chat("Ringkas bagian deployment dan security.", max_new_tokens=160)
```

Load a document from CLI:

```powershell
python chat_cli.py --checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt --context-file docs\ARCHITECTURE.md --mode chat --prompt "Apa poin arsitektur yang relevan untuk inference?"
```

Interactive CLI:

```txt
/doc docs/ARCHITECTURE.md
/memory
Apa bagian yang menjelaskan domain Lampung?
```

Useful long-context knobs:

```txt
--max-context-tokens             hard prompt budget, default 1024
--retrieval-top-k                number of chunks to retrieve, default 5
--retrieval-token-budget         token budget for retrieved chunks
--recent-turn-token-budget       token budget for recent chat turns
--long-input-threshold-chars     long user messages above this are stored as documents
```

For current small checkpoints, keep native prompt budget modest:

```txt
CPU safe: 768-1024 max_context_tokens
GPU test: 1024-2048 max_context_tokens
```

## Lampung Pipeline

`LampungPipeline` is lookup-first:

1. exact instruction lookup
2. word-order/bag-of-words lookup
3. compositional rules
4. model generation fallback

```python
from inference.lampung_pipeline import LampungPipeline

lampung = LampungPipeline(gen, tok)
response = lampung.translate("Lampung O", "English", "Nyak haga mengan manuk di warung paghek jalan")
print(response.text)
print(response.source)
```

## Router

`SigerRouter` keeps the CLI general while preserving Lampung domain accuracy.

```python
from inference.router import SigerRouter

router = SigerRouter(chat, lampung)
response = router.route("Nyak haga mengan manuk di warung paghek jalan")
print(response.route)
print(response.text)
```

Routes:

```txt
general_chat
lampung_to_id
id_to_lampung
lampung_to_en
```

## Expertise Orchestrator

`ExpertiseOrchestrator` is the runtime counterpart of the separated expertise
curriculum. It acts like a general coordinator:

1. summarize the user task
2. detect relevant domains
3. ask focused expertise prompts
4. synthesize the final answer

It does not put domain rules into the core model. Lampung still uses the
lookup-first pipeline when relevant.

```python
from inference.expertise_router import ExpertiseOrchestrator

expertise = ExpertiseOrchestrator(gen, lampung)
response = expertise.route(
    "Buat blueprint REST API todo dengan FastAPI, PostgreSQL, Docker, test, logging JSON, dan COMPLIANCE.md."
)
print(response.domains)
print(response.task_summary)
print(response.text)
```

Known expertise domains:

```txt
indonesian
lampung
general_knowledge
reasoning
programming_basic
programming_intermediate
programming_expert
```

## CLI

Installable command:

```powershell
pip install -e .
siger
siger ask "Jelaskan REST API secara ringkas."
siger chat --context-file docs\ARCHITECTURE.md
siger config set checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt
```

See `docs/CLI.md` for the full command guide.

Legacy script:

```powershell
python chat_cli.py
```

The CLI accepts direct questions by default. `SigerRouter` decides whether to use general chat or a Lampung domain tool.
The installable `siger` command defaults to `dynamic` mode, which can choose
general chat, Lampung tools, or the expertise orchestrator.

```txt
You: Nyak haga mengan manuk di warung paghek jalan
Assistant: aku mau makan ayam di warung dekat jalan
Route: lampung_to_id
Source: exact instruction lookup
```

Optional commands are still available for manual testing:

```txt
/help      show commands
/lo-id     Lampung O -> Indonesia
/id-lo     Indonesia -> Lampung O
/lo-en     Lampung O -> English
/reason    Lampung reasoning
/chat      force general chat
/reorder   Lampung word order
/expertise general expertise orchestrator
```

Legacy numeric modes are also supported: `0` auto, `1` LO->ID, `2` ID->LO, `3` LO->EN, `4` reasoning, `5` chat, `6` word order, `7` expertise.

Short route commands in `siger`:

```txt
/code TASK      developer/coding expertise
/basic TASK     programming basic
/debug TASK     algorithm/debug expertise
/expert TASK    software engineering expert
/reasoning TASK reasoning
/lampung TASK   Lampung expertise
/general TASK   general knowledge
```

They also work as one-shot subcommands:

```powershell
siger code "buat FastAPI todo lengkap dengan PostgreSQL"
siger expert "rancang arsitektur API production"
```

Expertise mode:

```powershell
python chat_cli.py --mode expertise --prompt "Jelaskan struktur kalimat Lampung berikut dan buat contoh FastAPI endpoint untuk menyimpan hasil terjemahan: Nyak haga mengan manuk di warung paghek jalan"
```

Lampung O -> English:

```txt
/lo-en
Input: Nyak haga mengan manuk di warung paghek jalan
English: i want to eat chicken at the stall near the road
Source: exact instruction lookup
```

## Sampling Defaults

For lookup-backed translation, temperature should be deterministic:

```txt
temperature=0.0
top_k=0
top_p=1.0
```

For general chat:

```txt
temperature=0.3
top_k=20
top_p=0.8
```

These defaults are conservative because current checkpoints are still small.

## Verification

```powershell
python -m py_compile chat_cli.py inference\router.py inference\lampung_pipeline.py retrieval\instruction_lookup.py retrieval\compositional_translator.py

@'
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```
