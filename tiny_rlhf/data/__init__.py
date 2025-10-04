"""Dataset adapter registry."""
from __future__ import annotations

from typing import Dict, Type

from tiny_rlhf.config import DatasetConfig

from .base import DatasetAdapter, DatasetSplits
from .freeform import FreeformAdapter
from .multiple_choice import MultipleChoiceAdapter
from .preference_pairs import PreferencePairsAdapter

_ADAPTERS: Dict[str, Type[DatasetAdapter]] = {
    "multiple_choice": MultipleChoiceAdapter,
    "freeform": FreeformAdapter,
    "preference_pairs": PreferencePairsAdapter,
}


def build_dataset(config: DatasetConfig) -> DatasetSplits:
    adapter_cls = _ADAPTERS.get(config.type)
    if adapter_cls is None:
        raise KeyError(f"Unknown dataset type: {config.type}")
    adapter = adapter_cls(config)
    return adapter.build()


__all__ = ["DatasetSplits", "build_dataset"]
