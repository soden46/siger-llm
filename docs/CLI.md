# Siger CLI

SigerLM can be installed as a local terminal command named `siger`.

## Install

From the repository root:

```powershell
pip install -e .
```

After that, the command is available from CMD or PowerShell:

```powershell
siger --help
```

## Interactive Chat

Start an interactive assistant session:

```powershell
siger
```

or:

```powershell
siger chat
```

Useful commands inside the session:

```txt
/help
/exit
/memory
/doc docs\ARCHITECTURE.md
/mode dynamic
/mode auto
/mode chat
/mode expertise
/checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt
```

Default mode is `dynamic`: users can type normal requests and Siger chooses the
route automatically.

Power users can force a route with slash commands:

```txt
/code buat FastAPI todo lengkap dengan PostgreSQL
/basic buat fungsi Python untuk membalik string
/debug debug kode rekursi ini: ...
/expert rancang arsitektur API production
/reasoning jelaskan langkah logika soal ini
/lampung jelaskan struktur kalimat: Nyak haga mengan manuk
/general jelaskan konsep machine learning
```

The same routes also work as one-shot subcommands:

```powershell
siger code "buat FastAPI todo lengkap dengan PostgreSQL"
siger expert "rancang arsitektur API production"
siger reasoning "pecahkan soal logika ini"
```

## Ask Once

```powershell
siger ask "Jelaskan REST API secara ringkas."
```

With a checkpoint:

```powershell
siger ask --checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt "Apa itu machine learning?"
```

With long-context document memory:

```powershell
siger ask ^
  --checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt ^
  --context-file docs\ARCHITECTURE.md ^
  --mode expertise ^
  "Ringkas arsitektur inference dan routing."
```

## Config

Siger CLI reads defaults from:

```txt
~\.siger\config.json
```

Show config:

```powershell
siger config show
```

Set default checkpoint:

```powershell
siger config set checkpoint checkpoints\lora\model_cpu_repair_general_merged.pt
```

Set default mode:

```powershell
siger config set mode expertise
```

Set context budget:

```powershell
siger config set max-context-tokens 1024
siger config set retrieval-top-k 6
siger config set retrieval-token-budget 420
```

Unset a value:

```powershell
siger config unset checkpoint
```

## Modes

```txt
dynamic   choose general, Lampung, or expertise route automatically
auto       route automatically between general chat and Lampung tools
chat       direct general chat
expertise  general expertise orchestrator
lo-id      Lampung O -> Bahasa Indonesia
id-lo      Bahasa Indonesia -> Lampung O
lo-en      Lampung O -> English
reason     Lampung explanation/reasoning
reorder    Lampung word reorder
```
