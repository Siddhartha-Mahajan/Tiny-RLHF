"""LoRA adapter registry."""
from __future__ import annotations

from typing import Any, Dict, Type

from tiny_rlhf.config import LoRAConfig

from .base import LoRAAdapter
from .peft_adapter import PeftLoRAAdapter
from .unsloth_adapter import UnslothLoRAAdapter

class _NoOpAdapter(LoRAAdapter):
    def prepare_model(self) -> None:  # pragma: no cover - nothing to do
        return

_ADAPTERS: Dict[str, Type[LoRAAdapter]] = {
    "peft": PeftLoRAAdapter,
    "unsloth": UnslothLoRAAdapter,
    "none": _NoOpAdapter,
}


def build_adapter(config: LoRAConfig, model: Any) -> LoRAAdapter:
    adapter_cls = _ADAPTERS.get(config.provider)
    if adapter_cls is None:
        raise KeyError(f"Unknown LoRA provider: {config.provider}")
    return adapter_cls(config, model)


__all__ = ["LoRAAdapter", "build_adapter"]
