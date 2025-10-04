"""Base classes for LoRA adapters."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tiny_rlhf.config import LoRAConfig


class LoRAAdapter(ABC):
    def __init__(self, config: LoRAConfig, model: Any):
        self.config = config
        self.model = model

    @abstractmethod
    def prepare_model(self) -> None:
        """Attach LoRA adapters to the model."""
