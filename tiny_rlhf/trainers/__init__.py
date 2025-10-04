"""Trainer registry."""
from __future__ import annotations

from typing import Dict, Type

from tiny_rlhf.config import TrainerConfig
from tiny_rlhf.models import ModelHandle

from .base import BaseTrainer
from .dpo_trainer import DPOTrainer
from .grpo_trainer import GRPOTrainer
from .sft_trainer import SFTTrainer

_TRAINERS: Dict[str, Type[BaseTrainer]] = {
    "sft": SFTTrainer,
    "grpo": GRPOTrainer,
    "dpo": DPOTrainer,
}


def build_trainer(config: TrainerConfig, model: ModelHandle) -> BaseTrainer:
    trainer_cls = _TRAINERS.get(config.algorithm)
    if trainer_cls is None:
        raise KeyError(f"Unknown trainer algorithm: {config.algorithm}")
    return trainer_cls(model, config)


__all__ = ["BaseTrainer", "build_trainer"]
