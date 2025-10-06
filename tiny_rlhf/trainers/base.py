"""Base trainer abstractions."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from tiny_rlhf.config import TrainerConfig
from tiny_rlhf.data import DatasetSplits
from tiny_rlhf.models import ModelHandle

logger = logging.getLogger(__name__)


@dataclass
class TrainResult:
    metrics: Dict[str, Any]
    artifacts: Dict[str, Any]


class BaseTrainer:
    def __init__(self, model: ModelHandle, config: TrainerConfig):
        self.model = model
        self.config = config
        self._experiment = None
        self._dataset_metadata: Optional[Dict[str, Any]] = None

    def train(self, dataset: DatasetSplits) -> TrainResult:
        raise NotImplementedError

    def evaluate(self, dataset: DatasetSplits) -> Dict[str, Any]:
        raise NotImplementedError

    def set_context(self, experiment: Any, dataset: DatasetSplits) -> None:
        self._experiment = experiment
        self._dataset_metadata = dataset.metadata

    @property
    def experiment(self) -> Any:
        return self._experiment

    @property
    def dataset_metadata(self) -> Optional[Dict[str, Any]]:
        return self._dataset_metadata
