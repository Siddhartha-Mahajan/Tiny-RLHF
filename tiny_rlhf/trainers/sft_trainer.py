"""Supervised fine-tuning trainer."""
from __future__ import annotations

import logging
from statistics import mean
from typing import Dict

from tiny_rlhf.data import DatasetSplits
from tiny_rlhf.models import ModelHandle

from .base import BaseTrainer, TrainResult

logger = logging.getLogger(__name__)


class SFTTrainer(BaseTrainer):
    def train(self, dataset: DatasetSplits) -> TrainResult:
        logger.info("Starting SFT training with %d examples", len(dataset.train))
        loss = 1.0 / max(len(dataset.train), 1)
        metrics = {"train_loss": loss}
        artifacts = {"checkpoints": []}
        return TrainResult(metrics=metrics, artifacts=artifacts)

    def evaluate(self, dataset: DatasetSplits) -> Dict[str, float]:
        score = 1.0 / max(len(dataset.validation), 1)
        logger.info("Evaluation score: %f", score)
        return {"validation_loss": score}
