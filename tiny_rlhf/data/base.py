"""Base dataset adapter primitives."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tiny_rlhf.config import DatasetConfig


@dataclass
class DatasetSplits:
    train: List[Dict]
    validation: List[Dict]
    test: Optional[List[Dict]] = None

    def summary(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test or []),
        }


class DatasetAdapter(ABC):
    def __init__(self, config: DatasetConfig):
        self.config = config

    @abstractmethod
    def build(self) -> DatasetSplits:
        """Load dataset splits according to the config."""

    def _load_jsonl(self, path: str | Path) -> List[Dict]:
        records: List[Dict] = []
        if path is None:
            return records
        full_path = Path(path)
        if not full_path.exists():
            raise FileNotFoundError(full_path)
        with open(full_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records


def _ensure_required_fields(example: Dict, fields: Iterable[str], context: str) -> None:
    missing = [field for field in fields if field not in example]
    if missing:
        raise KeyError(f"Missing fields {missing} in {context}")
