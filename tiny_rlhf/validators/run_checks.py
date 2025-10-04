"""Pre-flight validation checks."""
from __future__ import annotations

from pathlib import Path

from tiny_rlhf.config import ExperimentConfig


def run_preflight_checks(config: ExperimentConfig) -> None:
    dataset = config.dataset
    for path in [dataset.path, dataset.validation_path, dataset.test_path]:
        if path is None:
            continue
        if not Path(path).exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
