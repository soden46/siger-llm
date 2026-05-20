from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModalitySpec:
    """Declarative contract for one Siger modality family."""

    name: str
    family: str
    input_unit: str
    output_unit: str
    adapter_kind: str
    objective: str
    status: str = "planned"
    notes: str = ""


@dataclass
class ModalityBatch:
    """Common batch container after a modality-specific loader."""

    values: Any
    attention_mask: torch.Tensor | None = None
    targets: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModalityAdapter(nn.Module, ABC):
    """Base class for adapters that bridge raw modalities to Siger hidden states.

    The core Siger backbone consumes a sequence of vectors shaped
    ``(batch, length, d_model)``. Every modality owns its raw preprocessing,
    encoder/patchifier/quantizer, target formatting, and decoder head outside
    the backbone.
    """

    modality: str

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    @abstractmethod
    def encode(self, batch: ModalityBatch) -> torch.Tensor:
        """Return embeddings with shape ``(batch, length, d_model)``."""

    def decode(self, hidden_states: torch.Tensor, **kwargs: Any) -> Any:
        """Optional modality-specific decoder head."""
        raise NotImplementedError(f"{self.__class__.__name__} has no decoder.")

    def loss(self, outputs: Any, batch: ModalityBatch) -> torch.Tensor:
        """Optional modality-specific loss."""
        raise NotImplementedError(f"{self.__class__.__name__} has no loss.")
