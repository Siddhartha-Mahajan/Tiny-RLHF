"""Unsloth LoRA adapter implementation."""
from __future__ import annotations

import logging

from tiny_rlhf.config import LoRAConfig

from .base import LoRAAdapter

logger = logging.getLogger(__name__)


class UnslothLoRAAdapter(LoRAAdapter):
    def prepare_model(self) -> None:
        try:
            from unsloth import to_peft
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install tiny-rlhf[unsloth] to use this adapter") from exc

        logger.info("Converting Unsloth model to PEFT-compatible LoRA")
        to_peft(
            self.model,
            r=self.config.r,
            lora_alpha=self.config.alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias=self.config.bias,
            task_type=self.config.task_type,
        )
