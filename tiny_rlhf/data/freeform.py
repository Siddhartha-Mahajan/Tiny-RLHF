"""Free-form dataset adapter."""
from __future__ import annotations

from typing import Dict, List

from tiny_rlhf.config import DatasetConfig

from .base import DatasetAdapter, DatasetSplits, _ensure_required_fields


class FreeformAdapter(DatasetAdapter):
    def build(self) -> DatasetSplits:
        train = self._load_jsonl(self.config.path)
        validation = self._load_jsonl(self.config.validation_path) if self.config.validation_path else []
        test = self._load_jsonl(self.config.test_path) if self.config.test_path else []

        required = [self.config.input_field or "prompt", self.config.target_field or "completion"]
        for split_name, data in {"train": train, "validation": validation, "test": test}.items():
            for row in data:
                _ensure_required_fields(row, required, f"{split_name} example")

        return DatasetSplits(train=train, validation=validation, test=test)
