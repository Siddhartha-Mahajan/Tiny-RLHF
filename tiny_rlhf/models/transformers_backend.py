"""Transformers-based backend implementation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from tiny_rlhf.config import LoRAConfig, ModelConfig

from .base import ModelBackend, ModelHandle
from .lora import build_adapter

logger = logging.getLogger(__name__)


class TransformersBackend(ModelBackend):
    def load_model(self) -> ModelHandle:
        model_name = self.config.pretrained_model_name_or_path
        tokenizer = None

        if model_name.lower() == "mock":
            logger.warning("Using mock model – no real weights will be loaded")
            model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
        else:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("transformers is required for this backend") from exc

            logger.info("Loading model %s", model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=self._resolve_dtype(),
                trust_remote_code=self.config.trust_remote_code,
                load_in_8bit=self.config.load_in_8bit,
                load_in_4bit=self.config.load_in_4bit,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=self.config.trust_remote_code)

        if self.lora and self.lora.provider != "none":
            adapter = build_adapter(self.lora, model)
            adapter.prepare_model()

        return ModelHandle(model=model, tokenizer=tokenizer)

    def save_checkpoint(self, handle: ModelHandle, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if hasattr(handle.model, "save_pretrained"):
            handle.model.save_pretrained(output)
        else:
            torch.save(handle.model.state_dict(), output / "model.pt")
        if handle.tokenizer is not None and hasattr(handle.tokenizer, "save_pretrained"):
            handle.tokenizer.save_pretrained(output)

    def _resolve_dtype(self) -> Any:
        mapping = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.config.torch_dtype is None or self.config.torch_dtype == "auto":
            return None
        return mapping.get(self.config.torch_dtype.lower(), None)
