# SigerLM Distributed Training & Hardware Scaling Roadmap

This roadmap is intentionally separated from `ROADMAP.md`.

- `ROADMAP.md` focuses on **serious fine-tuning correctness and experiment quality**.
- This document focuses on **hardware scaling, distributed execution, multi-GPU training, and future cluster readiness**.

The purpose is to prepare SigerLM for a future where it can scale from:

```txt
single CPU / single GPU
  -> single-node multi-GPU
  -> multi-node cluster execution
  -> larger-model memory scaling with sharding strategies
```

This roadmap should be implemented gradually. SigerLM does **not** need to adopt every distributed technique immediately. The first goal is to make the training architecture **distributed-ready without breaking the current local workflow**.

Current implementation status: SigerLM has a cluster-ready experimental stack for CPU, single GPU, single-node multi-GPU, and `torchrun`/DDP-style launch detection. It is not yet a fully managed production cluster platform.

Experimental features now present:

- FSDP opt-in runtime wrapping.
- Sharded checkpoint utility.
- Graceful preemption handling for safer recovery.
- Activation/gradient checkpointing in the SigerLM block stack.
- Distributed validation loss aggregation helper.
- Conservative VRAM-aware batch-size suggestion.

Remaining gaps:

- No elastic multi-node job requeue orchestration yet.
- No automatic recovery from arbitrary node failure mid-step.
- No full validation dataset wiring in the default base trainer yet.
- No aggressive OOM-search batch tuner; current VRAM tuning is conservative.
- No production-grade sharded checkpoint consolidation policy yet.

---

# Why This Roadmap Exists

SigerLM currently emphasizes:

- custom SSM/Mamba-like language model development
- base training and LoRA fine-tuning
- Lampung domain adaptation
- lightweight experimentation on modest hardware
- CPU/VPS optimization and future ONNX/quantization work

As the project grows, future contributors may want to:

- run training on 2–8 GPUs
- run jobs on rented GPU machines
- use `torchrun` launchers
- scale to multi-node clusters
- train larger SigerLM variants
- reduce memory pressure with sharded training
- benchmark hardware efficiency across devices

This roadmap defines the engineering work needed to move safely in that direction.

---

# Scope

This roadmap covers:

- distributed training runtime design
- multi-GPU DDP-style execution
- distributed samplers and global batch semantics
- rank-aware logging and checkpointing
- distributed evaluation aggregation
- multi-node launch recipes
- optional future FSDP / ZeRO-style scaling
- activation checkpointing and memory-aware training
- hardware profiling and parity tests
- cluster documentation and contribution pathways

This roadmap applies to both:

1. **Base LM training**
2. **LoRA / adapter fine-tuning**

---

# Non-Goals for the First Distributed Milestone

The first milestone is **not**:

- immediate support for huge foundation-model-scale training
- automatic Kubernetes orchestration
- a full experiment-tracking platform
- mandatory DeepSpeed/FSDP adoption
- replacing the current simple local trainer path

The initial objective is more modest:

```txt
Make SigerLM training code cleanly support distributed execution
without harming the current single-device developer experience.
```

---

# Design Principles

## 1. Preserve Single-Device Simplicity

The existing local workflow must continue to work:

```powershell
python lora\run_lora.py --config ...
python main.py
```

Distributed support should be additive, not disruptive.

---

## 2. Runtime Should Be Explicit

Training code should know whether it is running in:

- single-process CPU
- single-process CUDA
- multi-process single-node GPU
- multi-node distributed mode

This state should be represented cleanly through a runtime/distributed context object rather than scattered environment checks.

---

## 3. Rank 0 Owns User-Facing Outputs

Only the main process should:

- write primary logs
- save normal checkpoints
- write summary files
- print most progress output

Other ranks should participate in computation, not spam outputs or corrupt files.

---

## 4. Batch Semantics Must Stay Correct

In distributed mode:

```txt
global_effective_batch =
per_device_batch_size
× gradient_accumulation_steps
× world_size
```

All config summaries and experiment metadata should report this accurately.

---

## 5. Evaluation Must Be Globally Correct

Validation loss, token counts, and derived metrics must represent **all participating ranks**, not only local process metrics.

---

## 6. Support Reproducibility Across Ranks

Random seeds, shuffling, and sampler state should be handled in a way that supports repeatable experiments as much as practical.

---

## 7. Introduce Heavy Scaling Techniques Only When Needed

DDP-style execution should be the first distributed target.  
FSDP, ZeRO, and other memory-sharding techniques should be introduced later when model size or memory pressure justifies them.

---

# Current Starting Point

SigerLM already has useful prerequisites:

- training/fine-tuning config files
- checkpointing logic
- hardware detection helpers
- CPU threading optimization
- single-device training loops
- LoRA fine-tuning pipeline
- future optimization scaffolding

What is still missing for distributed scale:

- distributed runtime abstraction
- `torchrun`-friendly entrypoints
- `DistributedSampler`
- rank-aware logging/checkpointing
- global metric aggregation
- distributed resume behavior
- cluster launch documentation
- multi-GPU correctness tests

---

# Priority Legend

| Priority | Meaning |
|---|---|
| **P0** | Required before distributed training can be trusted |
| **P1** | Strongly recommended for practical multi-GPU and cluster runs |
| **P2** | Future scaling and production-grade hardware support |

---

# P0 — Distributed-Ready Foundations

## 1. Create a Distributed Runtime Context

### Problem

Current code paths mostly reason about:

```txt
cpu or cuda
```

Cluster-ready training needs richer context:

```txt
is_distributed
rank
local_rank
world_size
device
backend
is_main_process
```

### Required Improvement

Add a small runtime abstraction, for example:

```txt
optimization/distributed/runtime.py
```

or:

```txt
training/distributed.py
```

It should expose:

```python
DistributedContext(
    enabled: bool,
    rank: int,
    local_rank: int,
    world_size: int,
    device: str,
    backend: str | None,
    is_main_process: bool,
)
```

### Expected Outcome

Training and fine-tuning code can become distributed-aware without duplicating environment logic everywhere.

---

## 2. Add Distributed Initialization and Cleanup Utilities

### Problem

Distributed training requires coordinated process-group lifecycle management.

### Required Improvement

Implement helpers such as:

```python
init_distributed(...)
barrier(...)
cleanup_distributed(...)
```

The code should:

- detect whether distributed launch environment variables exist
- initialize a process group only when needed
- select the correct local GPU device when CUDA is used
- clean up safely at the end of training

### Expected Outcome

Training scripts can support both local and `torchrun` execution.

---

## 3. Make Training Entrypoints Launcher-Friendly

### Problem

Current scripts are intended primarily for direct Python execution.

### Required Improvement

Ensure relevant entrypoints can work with a distributed launcher pattern such as:

```bash
torchrun --nproc_per_node=4 lora/run_lora.py --config ...
```

and future base-training equivalents.

This includes:

- proper device selection
- process-aware startup output
- safe initialization order
- graceful exit

### Expected Outcome

SigerLM can be launched on multiple GPUs without redesigning the training logic.

---

## 4. Add Distributed Sampler Support

### Problem

Without rank-aware sampling, each GPU/process may read overlapping data unnecessarily.

### Required Improvement

When distributed mode is enabled:

- use `DistributedSampler`
- ensure epoch changes call `sampler.set_epoch(epoch)`
- preserve standard shuffling in non-distributed mode
- apply this to both training and validation loaders as appropriate

### Expected Outcome

Data is partitioned consistently across devices.

---

## 5. Make Logging Rank-Aware

### Problem

If every rank prints all logs, console output becomes noisy and debugging becomes harder.

### Required Improvement

Create helper behavior such as:

```python
print_rank0(...)
log_rank0(...)
```

Only rank 0 should handle most:

- progress lines
- summary output
- artifact paths
- user-facing warnings

Other ranks may log only when explicitly debugging.

### Expected Outcome

Distributed output remains readable.

---

## 6. Make Checkpoint Saving Rank-Safe

### Problem

Multiple processes writing the same checkpoint path can corrupt outputs.

### Required Improvement

Only rank 0 should write standard artifacts:

- LoRA checkpoints
- best checkpoints
- last checkpoints
- merged model exports
- run metadata

Other ranks should synchronize where necessary.

### Expected Outcome

Checkpoint files remain valid and deterministic.

---

## 7. Report Global Effective Batch Size Correctly

### Problem

A config summary based only on local batch size becomes misleading in distributed mode.

### Required Improvement

Whenever training begins, print/report:

```txt
per_device_batch_size
grad_accum
world_size
global_effective_batch_size
```

Formula:

```txt
global_effective_batch_size =
per_device_batch_size × grad_accum × world_size
```

### Expected Outcome

Experiment records remain interpretable when scaling hardware.

---

## 8. Make Step Accounting Compatible With Distributed Training

### Problem

The fine-tuning roadmap already requires correct separation of:

- micro-step
- optimizer/global step

Distributed training must preserve that fix.

### Required Improvement

Ensure:

- `max_steps` means optimizer update steps
- accumulation semantics stay stable as world size changes
- scheduler steps are aligned with optimizer steps
- save/eval intervals remain optimizer-step based

### Expected Outcome

A run on 1 GPU and a run on 4 GPUs use comparable training semantics.

---

# P1 — Practical Multi-GPU and Cluster Execution

## 9. Wrap Models for DDP-Style Multi-GPU Training

### Problem

After process initialization and sampler support, models still need a distributed wrapper for gradient synchronization.

### Required Improvement

Add a clean model wrapping path for distributed data parallel execution.

This should be supported for:

- base LM training
- LoRA training

The implementation should:

- wrap only when distributed mode is active
- avoid breaking existing single-device code
- work with trainable LoRA parameters
- preserve access to underlying modules when saving/exporting

### Expected Outcome

SigerLM can train across multiple GPUs on a single machine.

---

## 10. Add Distributed Validation Metric Aggregation

### Problem

Validation metrics computed locally on each rank are incomplete.

### Required Improvement

Aggregate across ranks:

- sum of validation loss numerator
- number of valid target tokens or examples
- final mean loss
- final perplexity from global loss

### Expected Outcome

Validation metrics reflect the whole distributed validation pass.

---

## 11. Add Distributed-Aware Resume Logic

### Problem

Resuming from checkpoints in distributed mode requires consistency across ranks.

### Required Improvement

Resume should ensure:

- all ranks restore the same training position
- checkpoint metadata is consistent
- optimizer and scheduler state match the run state
- resumed distributed runs do not diverge due to mismatched step counters

### Expected Outcome

Interrupted multi-GPU training can resume reliably.

---

## 12. Add Reproducibility Rules for Distributed Runs

### Problem

Distributed shuffling and rank-local RNG behavior can reduce reproducibility.

### Required Improvement

Define and document:

- global base seed
- rank-adjusted seed strategy where appropriate
- DataLoader worker seed behavior
- deterministic sampler behavior when feasible

### Expected Outcome

Distributed experiments become more repeatable.

---

## 13. Add Distributed Smoke Tests

### Problem

Distributed code can appear correct but fail in edge cases.

### Required Improvement

Add minimal tests or scripts for:

- 2-process local run
- sampler partition sanity
- rank 0 only checkpoint writes
- global metric aggregation
- correct batch/step reporting
- successful shutdown

These tests may initially be script-based rather than full CI GPU tests.

### Expected Outcome

Distributed changes are less likely to silently regress.

---

## 14. Add Single-Node Multi-GPU Launch Recipes

### Problem

Contributors need clear commands.

### Required Improvement

Create documentation examples such as:

```bash
torchrun --nproc_per_node=2 lora/run_lora.py --config ...
torchrun --nproc_per_node=4 training/run_base.py --config ...
```

Include:

- CUDA-visible-devices usage
- expected logs
- troubleshooting notes
- when to prefer smaller batch sizes

### Expected Outcome

Contributors can actually use multi-GPU support.

---

## 15. Add Multi-Node Cluster Launch Documentation

### Problem

Cluster execution requires launch recipes beyond single-node examples.

### Required Improvement

Document examples for:

- multi-node `torchrun`
- environment variables
- node rank
- rendezvous address/port
- world size thinking
- synchronization expectations

Include an example structure similar to:

```bash
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=<rank> \
  --master_addr=<host> \
  --master_port=<port> \
  lora/run_lora.py --config ...
```

### Expected Outcome

SigerLM has a clear path to cluster experimentation.

---

## 16. Add Optional SLURM Job Templates

### Problem

Many research and academic GPU clusters use schedulers.

### Required Improvement

Add optional examples such as:

```txt
scripts/slurm/lora_single_node_multi_gpu.sbatch
scripts/slurm/lora_multi_node.sbatch
```

or documentation-only templates.

### Expected Outcome

The project becomes easier to run in shared cluster environments.

---

# P2 — Larger Model Scaling and Advanced Hardware Efficiency

## 17. Evaluate FSDP Integration for Larger SigerLM Variants

### Problem

If future SigerLM variants become too large for comfortable per-GPU replication, DDP-style training may become memory-limited.

### Required Improvement

Investigate a future FSDP path for:

- sharded parameters
- sharded gradients
- sharded optimizer state
- compatibility with SigerLM model blocks
- checkpoint save/load strategies

This should be treated as an optional later-stage path, not a replacement for simpler DDP.

### Expected Outcome

SigerLM has a roadmap toward larger model variants.

---

## 18. Evaluate DeepSpeed / ZeRO-Style Scaling as an Optional Backend

### Problem

Some contributors may prefer alternative distributed optimization stacks for large-model training.

### Required Improvement

Investigate optional integration boundaries for:

- ZeRO-style optimizer state partitioning
- memory offload possibilities
- launcher/config coexistence
- whether it meaningfully benefits SigerLM workloads

This should remain optional and should not hard-couple the project to one external framework too early.

### Expected Outcome

SigerLM remains open to advanced scaling approaches without overcommitting.

---

## 19. Add Activation Checkpointing Support

### Problem

Longer sequence lengths and larger models increase activation memory costs.

### Required Improvement

Evaluate and optionally add:

- activation checkpointing in model blocks
- config toggles
- memory/performance notes
- compatibility checks with distributed training

### Expected Outcome

Larger experiments become more feasible under memory constraints.

---

## 20. Add Sharded or Distributed Checkpoint Strategy

### Problem

Large distributed models may outgrow single-file checkpoint workflows.

### Required Improvement

Plan a future checkpointing layer that can support:

- normal single-file checkpoints for small runs
- sharded checkpoints for large distributed runs
- metadata manifests
- checkpoint consolidation when exporting

### Expected Outcome

Checkpoints remain manageable as model/hardware scale grows.

---

## 21. Add Hardware Profiling and Scaling Benchmarks

### Problem

Scaling should be measured, not assumed.

### Required Improvement

Benchmark and record:

- tokens/sec
- samples/sec
- GPU memory usage
- scaling efficiency from 1 → 2 → 4 → 8 GPUs
- validation cost overhead
- checkpoint save overhead

Store benchmark outputs in a reproducible form.

### Expected Outcome

The project can compare hardware scaling decisions with evidence.

---

## 22. Add Distributed Parity Tests Against Single-Device Runs

### Problem

Distributed logic should not significantly change expected training behavior in small controlled experiments.

### Required Improvement

Create small parity experiments comparing:

```txt
1 GPU
vs
2 GPUs distributed
```

For a tiny seed-fixed dataset/config, compare:

- total examples consumed
- global batch interpretation
- loss trend sanity
- checkpoint resume behavior
- validation metric consistency

### Expected Outcome

Distributed implementation can be trusted.

---

## 23. Add Failure Handling and Synchronization Safeguards

### Problem

Distributed runs can fail unevenly across ranks.

### Required Improvement

Add and document:

- graceful cleanup on exceptions
- barriers only where necessary
- meaningful distributed error messages
- timeout awareness
- avoiding deadlocks caused by rank-specific control flow

### Expected Outcome

Cluster jobs are easier to debug.

---

## 24. Add Cluster Readiness Documentation

### Problem

Hardware scaling is not only a code change; it requires contributor-facing guidance.

### Required Improvement

Create:

```txt
docs/DISTRIBUTED_TRAINING.md
```

or equivalent documentation covering:

- distributed concepts used in SigerLM
- single-node multi-GPU commands
- multi-node examples
- SLURM templates
- supported/unsupported modes
- troubleshooting
- scaling recommendations
- decision guide: DDP vs future FSDP/ZeRO

### Expected Outcome

The scaling path is understandable to contributors.

---

# Suggested Implementation Phases

## Phase 0 — Finish Fine-Tuning Correctness First

Before serious distributed training work, finish the core fine-tuning quality roadmap:

- step accounting
- validation
- resume
- tokenizer safety
- dataset quality checks
- structured logging

Distributed training should build on a correct single-device trainer.

---

## Phase 1 — Distributed-Ready Abstractions

Implement:

1. distributed runtime context
2. init/cleanup helpers
3. launcher-friendly entrypoints
4. rank-aware logging/checkpointing
5. global batch reporting

---

## Phase 2 — Single-Node Multi-GPU

Implement:

6. distributed samplers
7. DDP-style wrapping
8. distributed validation aggregation
9. distributed resume
10. smoke tests
11. single-node launch docs

---

## Phase 3 — Multi-Node Cluster Support

Implement:

12. multi-node launch docs
13. SLURM templates
14. synchronization/failure notes
15. cluster debugging guidance

---

## Phase 4 — Larger-Model Scaling Research

Investigate:

16. FSDP path
17. ZeRO/DeepSpeed path
18. activation checkpointing
19. sharded checkpoints
20. hardware scaling benchmarks

---

# Roadmap Status Tracker

| No. | Item | Priority | Status |
|---|---|---|---|
| 1 | Distributed runtime context | P0 | Pending |
| 2 | Distributed init/cleanup helpers | P0 | Pending |
| 3 | Launcher-friendly training entrypoints | P0 | Pending |
| 4 | Distributed sampler support | P0 | Pending |
| 5 | Rank-aware logging | P0 | Pending |
| 6 | Rank-safe checkpoint saving | P0 | Pending |
| 7 | Global effective batch reporting | P0 | Pending |
| 8 | Distributed-safe step accounting | P0 | Pending |
| 9 | DDP-style multi-GPU model wrapping | P1 | Pending |
| 10 | Distributed validation metric aggregation | P1 | Pending |
| 11 | Distributed-aware resume logic | P1 | Pending |
| 12 | Distributed reproducibility rules | P1 | Pending |
| 13 | Distributed smoke tests | P1 | Pending |
| 14 | Single-node multi-GPU launch docs | P1 | Pending |
| 15 | Multi-node cluster launch docs | P1 | Pending |
| 16 | Optional SLURM job templates | P1 | Pending |
| 17 | Evaluate FSDP integration | P2 | Pending |
| 18 | Evaluate DeepSpeed / ZeRO-style scaling | P2 | Pending |
| 19 | Activation checkpointing support | P2 | Pending |
| 20 | Sharded/distributed checkpoint strategy | P2 | Pending |
| 21 | Hardware profiling and scaling benchmarks | P2 | Pending |
| 22 | Distributed parity tests | P2 | Pending |
| 23 | Failure handling and synchronization safeguards | P2 | Pending |
| 24 | Cluster readiness documentation | P2 | Pending |

---

# Definition of “Distributed-Ready”

SigerLM can be considered **distributed-ready** when:

- local single-device runs still work unchanged
- distributed launchers can start training cleanly
- each rank receives coherent runtime state
- datasets are sharded properly across ranks
- only rank 0 writes user-facing artifacts
- global batch size is reported correctly
- validation metrics aggregate across ranks
- checkpoints and resumes are safe
- basic multi-GPU smoke tests pass
- contributors have launch documentation

---

# Definition of “Cluster-Ready”

SigerLM can be considered **cluster-ready** when:

- multi-node launch documentation exists
- training can run across multiple machines
- rank/world-size semantics are documented
- validation/checkpoint behavior remains correct
- distributed failures are easier to diagnose
- optional scheduler templates exist
- scaling benchmarks are recorded
- advanced memory-scaling paths are assessed for larger model variants

---

# Suggested Future Codex Task Batches

## Codex Batch D — Distributed Runtime Foundation

```txt
Create the first distributed-training foundation for SigerLM without changing default single-device behavior.

Scope:
- Add a small distributed runtime/context module.
- Support detection of rank, local_rank, world_size, distributed enabled state, backend, and is_main_process.
- Add init_distributed() and cleanup_distributed() helpers.
- Keep direct `python ...` execution behavior unchanged.
- Prepare `lora/run_lora.py` and future base-training entrypoints to consume the runtime context cleanly.
- Add rank-0-safe printing helper(s).

Do not add DDP wrapping yet.
Do not add multi-node docs yet.
Focus only on clean distributed-aware runtime foundations.
```

### Roadmap Coverage

- #1
- #2
- partial #3
- partial #5

---

## Codex Batch E — Multi-GPU Fine-Tuning Enablement

```txt
Extend SigerLM LoRA fine-tuning to support single-node multi-GPU distributed training.

Build on the distributed runtime/context module.

Implement:
1. launcher-friendly behavior for `torchrun --nproc_per_node=N lora/run_lora.py --config ...`
2. DistributedSampler for training and validation datasets when distributed mode is enabled
3. DDP-style model wrapping for LoRA training
4. rank-0-only checkpoint saves and artifact writes
5. global effective batch size reporting:
   per_device_batch_size × grad_accum × world_size
6. distributed-aware logging behavior
7. distributed validation metric aggregation across ranks

Keep single-device behavior backward compatible.
```

### Roadmap Coverage

- #3
- #4
- #5
- #6
- #7
- #9
- #10

---

## Codex Batch F — Distributed Resume, Tests, and Documentation

```txt
Improve reliability and contributor usability for SigerLM distributed training.

Implement:
1. distributed-aware resume behavior for LoRA training checkpoints
2. reproducibility rules for distributed seeds and sampler epochs
3. distributed smoke-test scripts or test utilities for:
   - sampler partitioning
   - rank-0 checkpoint writing
   - global metric aggregation
4. documentation for single-node multi-GPU launches
5. initial documentation outline for multi-node launches
6. clear troubleshooting notes for common distributed execution mistakes

Do not implement FSDP or DeepSpeed yet.
```

### Roadmap Coverage

- #11
- #12
- #13
- #14
- partial #15
- partial #24

---

# Notes on Advanced Scaling

Future work may explore:

- FSDP-style sharded training for models that no longer comfortably fit replicated on each GPU
- ZeRO-style optimizer and memory partitioning approaches
- activation checkpointing to reduce memory use at the cost of recomputation
- sharded checkpoint formats
- benchmark-driven decisions about when each technique is worthwhile

These should remain opt-in advanced paths, introduced only after the single-device and DDP-style foundations are stable.

---

# Final Note

SigerLM does not need to become a full cluster-first training framework immediately.

The recommended evolution is:

```txt
correct single-device training
  -> reliable serious fine-tuning
  -> distributed-ready abstractions
  -> multi-GPU training
  -> multi-node cluster workflows
  -> optional large-model sharding strategies
```

This staged path keeps the project practical for today's contributors while leaving a credible route toward future hardware scale.
