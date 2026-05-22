# Long-Context Strategy

SigerLM should treat very large context as a memory and retrieval problem first,
not as a requirement to place every token inside the native model window.

## Design

```txt
long text / document / session history
  -> overlapping chunks
  -> lexical retrieval
  -> compact extractive summary
  -> token-budgeted prompt assembly
  -> answer generation
```

This gives SigerLM a practical path toward document-scale usage on CPU/VPS
hardware while native context training grows gradually.

## Runtime Components

- `memory/chunk_store.py`: overlapping chunk storage and retrieval scoring.
- `memory/session_memory.py`: document ingestion, long user input ingestion, and summaries.
- `memory/context_manager.py`: prompt assembly with separate budgets for retrieved chunks and recent turns.
- `inference/chat.py`: automatically stores huge user messages as retrievable context.
- `inference/expertise_router.py`: uses the same memory for domain expertise orchestration.

## CLI Usage

Load a large document before asking:

```powershell
python chat_cli.py ^
  --checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt ^
  --context-file docs\ARCHITECTURE.md ^
  --max-context-tokens 1024 ^
  --retrieval-top-k 6 ^
  --retrieval-token-budget 420 ^
  --mode chat ^
  --prompt "Ringkas bagian inference dan routing."
```

Use the expertise orchestrator with long context:

```powershell
python chat_cli.py ^
  --checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt ^
  --context-file docs\ARCHITECTURE.md ^
  --mode expertise ^
  --prompt "Jelaskan bagaimana router Lampung dan general expertise saling melengkapi."
```

Interactive:

```txt
/doc docs/ARCHITECTURE.md
/memory
Jelaskan bagian MoE dan routing secara singkat.
```

## Practical Limits

Current small checkpoints should stay conservative:

```txt
max_context_tokens: 768-1024 on CPU
retrieval_top_k: 4-8
retrieval_token_budget: 256-512
max_new_tokens: 80-180
```

The long document can be much larger than this because only relevant chunks are
inserted into the prompt.

## Native Context Roadmap

Grow native context gradually:

```txt
512 -> 1024 -> 2048 -> 4096
```

Each jump needs:

- training or LoRA examples at that length
- clean long-context QA/summarization data
- evals for retrieval accuracy and needle-in-haystack behavior
- memory checks on CPU/GPU

Do not jump directly to huge native windows. For SigerLM, the strongest path is:

```txt
moderate native context + retrieval memory + summarization + expertise routing
```

## Dataset Ideas

Create long-context instruction rows like:

```json
{
  "instruction": "Jawab pertanyaan berdasarkan dokumen panjang berikut.",
  "input": "Dokumen: ...\n\nPertanyaan: ...",
  "output": "Jawaban yang hanya memakai bagian relevan dari dokumen.",
  "source": "long_context_synthetic",
  "type": "long_context_qa"
}
```

Keep outputs grounded and short. The goal is to teach the model to use retrieved
evidence, not to memorize huge documents.
