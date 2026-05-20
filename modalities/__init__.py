"""Modality adapter contracts for Siger's modality-agnostic backbone."""

from .base import ModalityAdapter, ModalityBatch, ModalitySpec
from .registry import MODALITY_SPECS, get_modality_spec, list_modalities

__all__ = [
    "MODALITY_SPECS",
    "ModalityAdapter",
    "ModalityBatch",
    "ModalitySpec",
    "get_modality_spec",
    "list_modalities",
]
