"""PEFT LoRA adapter implementation."""
from __future__ import annotations

import logging

from tiny_rlhf.config import LoRAConfig

from .base import LoRAAdapter

logger = logging.getLogger(__name__)


class PeftLoRAAdapter(LoRAAdapter):
    def prepare_model(self) -> None:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install peft to use the PEFT adapter") from exc

        lora_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            bias=self.config.bias,
            task_type=self.config.task_type,
        )
        get_peft_model(self.model, lora_config)
        logger.info("Attached PEFT LoRA adapters")
