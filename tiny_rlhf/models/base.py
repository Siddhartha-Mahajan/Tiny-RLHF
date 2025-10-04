"""Model backend abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from tiny_rlhf.config import LoRAConfig, ModelConfig


@dataclass
class ModelHandle:
    model: Any
    tokenizer: Any


class ModelBackend:
    def __init__(self, config: ModelConfig, lora: Optional[LoRAConfig] = None):
        self.config = config
        self.lora = lora

    def load_model(self) -> ModelHandle:
        raise NotImplementedError

    def save_checkpoint(self, handle: ModelHandle, output_dir: str) -> None:
        raise NotImplementedError
