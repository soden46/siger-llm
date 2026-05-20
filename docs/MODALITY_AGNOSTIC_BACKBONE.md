# Modality-Agnostic Siger Backbone

SigerLM starts as a text language model, but the core SSM stack should remain a generic sequence backbone. The long-term target is:

```txt
raw modality
  -> modality adapter / tokenizer / encoder
  -> sequence embeddings shaped (batch, length, d_model)
  -> Siger backbone
  -> modality-specific head / decoder / tool layer
```

The core model must not know whether a timestep came from text, pixels, audio frames, table cells, graph walks, robot states, or event streams. It should only process ordered hidden vectors.

## Core Boundary

`model/siger_model.py` now exposes:

```python
hidden = model.forward_hidden(input_ids=token_ids)
hidden = model.forward_hidden(inputs_embeds=modality_embeddings)
logits = model.project_logits(hidden)
```

Text training still uses:

```python
logits, loss = model(input_ids, targets=labels)
```

Non-text modalities should not be hardcoded into `model/`. They should live in adapter modules and call `forward_hidden(inputs_embeds=...)`.

## Adapter Contract

The common adapter contract lives in:

```txt
modalities/base.py
modalities/registry.py
```

Every modality adapter is responsible for:

- converting raw inputs into `(B, L, d_model)` embeddings
- defining target format
- defining decoder/head if needed
- defining loss if it is not text cross-entropy

## Capability Map

| Capability | Adapter Direction | Typical Objective |
|---|---|---|
| Text generator | token IDs -> Siger -> token logits | causal LM |
| Code model | code tokens -> Siger -> code tokens | causal LM / instruction |
| Pixel language model | image patches/tokens -> Siger -> patch/token logits | next/blocked patch prediction |
| Vision encoder | image patches -> Siger -> pooled embedding | contrastive / classification |
| Vision-language model | image patches + text tokens -> Siger -> text/head | captioning / VQA |
| Image generator | text/condition -> Siger -> image latent/tokens | diffusion or AR image tokens |
| Speech to text | audio frames -> Siger -> text tokens | CTC / seq2seq |
| Text to speech | text tokens -> Siger -> codec/mel tokens | codec generation |
| Audio classification | audio frames -> Siger -> label | classification |
| Speaker recognition | audio frames -> Siger -> speaker embedding | contrastive / classification |
| Emotion from voice | audio frames -> Siger -> emotion label | classification |
| Audio captioning | audio frames -> Siger -> text tokens | captioning |
| Voice assistant | audio -> text/dialogue -> speech | ASR + dialogue + TTS |
| Video model | tubelet patches -> Siger -> label/tokens | video understanding |
| Audio-visual omni | mixed stream -> Siger -> mixed outputs | multimodal instruction |
| Action / robotics | observation/state -> Siger -> actions | behavior cloning |
| Time series / sensor | numeric timesteps -> Siger -> forecast/label | forecasting |
| Structured/tabular | typed cells/rows -> Siger -> label/text | table QA / prediction |
| Graph / KG | node-edge paths -> Siger -> relation/node | link prediction |
| Retrieval/memory/agent | query + memory + tool events -> Siger | RAG / tool use |
| Document AI/OCR | page patches + layout -> Siger -> text/bbox | OCR / extraction |
| Music/symbolic sequence | MIDI/codec tokens -> Siger -> tokens | continuation |
| Biological sequence | DNA/RNA/protein tokens -> Siger -> label/tokens | sequence modeling |
| Financial/event sequence | event/numeric stream -> Siger -> forecast/event | forecasting |

## Build Order

Do not attempt all modalities at once. Grow Siger in rings:

1. **Text/code/retrieval**: already closest to the current stack.
2. **Vision encoder and document OCR**: patch embeddings into `forward_hidden`.
3. **Audio encoder and ASR**: log-mel/audio codec frontend into `forward_hidden`.
4. **Vision-language and audio captioning**: combine projected modality embeddings with text embeddings.
5. **Generators**: image/TTS/music require their own decoders and losses.
6. **Action/time-series/graph/bio/finance**: numeric or symbolic sequence adapters.
7. **Omni model**: mixed modality stream with typed embeddings and multiple heads.

## Design Rules

- Keep `model/ssm_core.py`, `model/ssm_block.py`, and the SSM stack modality-neutral.
- Add modality-specific preprocessing, losses, and decoders outside `model/`.
- Do not make text tokenizer IDs the only input path.
- Preserve text checkpoint compatibility by keeping text embedding and LM head names stable.
- Use `SigerConfig.input_modality`, `output_modality`, and `model_kind` as metadata, not as hardcoded branching in the SSM core.
- Prefer adapters that project into `d_model` over separate model forks.

## Minimal Adapter Skeleton

```python
class ImagePatchAdapter(ModalityAdapter):
    modality = "vision_encoder"

    def __init__(self, d_model: int, patch_dim: int):
        super().__init__(d_model)
        self.proj = nn.Linear(patch_dim, d_model)

    def encode(self, batch: ModalityBatch) -> torch.Tensor:
        return self.proj(batch.values)
```

Then:

```python
embeds = adapter.encode(batch)
hidden = siger.forward_hidden(inputs_embeds=embeds)
```

This is the central foundation: Siger becomes a sequence backbone first, while each modality owns its own edge logic.
