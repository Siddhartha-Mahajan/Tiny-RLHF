from __future__ import annotations

from pathlib import Path

from tiny_rlhf.config import load_experiment_config


def test_load_quickstart_sft(tmp_path: Path) -> None:
    config_path = Path("configs/experiments/quickstart_sft.yaml").resolve()
    config = load_experiment_config(config_path)
    assert config.dataset.name == "multiple_choice_default"
    assert config.run.pipeline == "sft"
