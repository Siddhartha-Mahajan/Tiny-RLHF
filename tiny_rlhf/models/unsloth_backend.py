"""Unsloth backend implementation."""
from __future__ import annotations

import logging
from typing import Any

from tiny_rlhf.config import LoRAConfig, ModelConfig

from .base import ModelBackend, ModelHandle
from .lora import build_adapter

logger = logging.getLogger(__name__)


class UnslothBackend(ModelBackend):
    def load_model(self) -> ModelHandle:
        try:
            from unsloth import FastLanguageModel  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install tiny-rlhf[unsloth] to use this backend") from exc

        model_name = self.config.pretrained_model_name_or_path
        logger.info("Loading Unsloth model %s", model_name)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name,
            max_seq_length=self.config.max_seq_length or 4096,
            dtype=self.config.torch_dtype or "float16",
            load_in_4bit=self.config.load_in_4bit,
        )

        if self.lora and self.lora.provider != "none":
            adapter = build_adapter(self.lora, model)
            adapter.prepare_model()

        return ModelHandle(model=model, tokenizer=tokenizer)

    def save_checkpoint(self, handle: ModelHandle, output_dir: str) -> None:
        if hasattr(handle.model, "save_pretrained"):
            handle.model.save_pretrained(output_dir)
        else:
            raise RuntimeError("Unsloth model does not support save_pretrained")
