# Inference

SigerLM inference has two layers:

1. Raw generation/chat with `Generator` and `ChatSession`.
2. Routed inference with `SigerRouter`, which can call general chat or Lampung domain tools.

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

chat = ChatSession(gen, max_context_tokens=1024)
reply = chat.chat("Apa itu AI?", max_new_tokens=80)
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

## CLI

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

Lampung O -> English:

```txt
Mode: 3
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
0
Nyak haga mengan manuk di warung paghek jalan
exit
'@ | python chat_cli.py
```
