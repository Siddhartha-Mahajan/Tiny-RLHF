"""Model backend factory."""
from __future__ import annotations

from typing import Dict, Type

from tiny_rlhf.config import LoRAConfig, ModelConfig

from .base import ModelBackend, ModelHandle
from .transformers_backend import TransformersBackend
from .unsloth_backend import UnslothBackend

_BACKENDS: Dict[str, Type[ModelBackend]] = {
    "transformers": TransformersBackend,
    "unsloth": UnslothBackend,
}


def load_model(config: ModelConfig, lora: LoRAConfig | None = None) -> ModelHandle:
    backend_cls = _BACKENDS.get(config.provider)
    if backend_cls is None:
        raise KeyError(f"Unknown model provider: {config.provider}")
    backend = backend_cls(config, lora)
    return backend.load_model()


def save_model(config: ModelConfig, handle: ModelHandle, output_dir: str) -> None:
    backend_cls = _BACKENDS.get(config.provider)
    if backend_cls is None:
        raise KeyError(f"Unknown model provider: {config.provider}")
    backend = backend_cls(config)
    backend.save_checkpoint(handle, output_dir)


__all__ = ["ModelHandle", "load_model", "save_model"]
