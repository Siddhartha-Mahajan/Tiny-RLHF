"""GRPO trainer stub."""
from __future__ import annotations

import logging
from typing import Dict

from tiny_rlhf.data import DatasetSplits
from tiny_rlhf.models import ModelHandle

from .base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class GRPOTrainer(BaseTrainer):
    def train(self, dataset: DatasetSplits) -> TrainResult:
        logger.info("Running GRPO with %d prompts", len(dataset.train))
        metrics = {"reward_mean": 0.5}
        artifacts = {"policy": "mock"}
        return TrainResult(metrics=metrics, artifacts=artifacts)

    def evaluate(self, dataset: DatasetSplits) -> Dict[str, float]:
        logger.info("Evaluating GRPO policy on %d prompts", len(dataset.validation))
        return {"validation_reward": 0.5}
