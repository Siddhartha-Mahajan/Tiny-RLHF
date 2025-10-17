"""Model loading and saving utilities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any


_DTYPE_LOOKUP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_dtype(value: Optional[str]) -> Any:
    if value is None or value == "auto":
        return None
    return _DTYPE_LOOKUP.get(value.lower())


def load_model(config: ModelConfig) -> ModelBundle:
    """Instantiate the base model + tokenizer."""

    torch_dtype = _resolve_dtype(config.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.pretrained,
        torch_dtype=torch_dtype,
        trust_remote_code=config.trust_remote_code,
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained, trust_remote_code=config.trust_remote_code)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if config.use_peft:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("`peft` is required when `use_peft=True`. Install peft>=0.11.") from exc

        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    return ModelBundle(model=model, tokenizer=tokenizer)


def save_model(bundle: ModelBundle, output_dir: str) -> None:
    """Persist the model/tokenizer pair to disk."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    bundle.model.save_pretrained(path)
    if hasattr(bundle.tokenizer, "save_pretrained"):
        bundle.tokenizer.save_pretrained(path)
