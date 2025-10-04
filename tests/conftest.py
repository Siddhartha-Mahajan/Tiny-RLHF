from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pytest  # type: ignore


@pytest.fixture()
def synthetic_mc_dataset(tmp_path: Path) -> Dict[str, Path]:
    data_dir = tmp_path / "mc"
    data_dir.mkdir(parents=True, exist_ok=True)
    train = [{"question": "Q?", "options": ["A", "B"], "answer": "A"} for _ in range(4)]
    validation = [{"question": "Q?", "options": ["A", "B"], "answer": "A"} for _ in range(2)]

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "validation.jsonl"

    for path, records in [(train_path, train), (val_path, validation)]:
        with open(path, "w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row) + "\n")

    return {"train": train_path, "validation": val_path}
