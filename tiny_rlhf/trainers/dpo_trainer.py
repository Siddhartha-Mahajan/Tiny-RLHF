"""DPO trainer stub."""
from __future__ import annotations

import logging
from typing import Dict

from tiny_rlhf.data import DatasetSplits
from tiny_rlhf.models import ModelHandle

from .base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class DPOTrainer(BaseTrainer):
    def train(self, dataset: DatasetSplits) -> TrainResult:
        logger.info("Training DPO on %d preference pairs", len(dataset.train))
        metrics = {"dpo_loss": 0.1}
        artifacts = {"policy": "mock"}
        return TrainResult(metrics=metrics, artifacts=artifacts)

    def evaluate(self, dataset: DatasetSplits) -> Dict[str, float]:
        logger.info("Evaluating DPO on %d preference pairs", len(dataset.validation))
        return {"validation_reward": 0.1}
