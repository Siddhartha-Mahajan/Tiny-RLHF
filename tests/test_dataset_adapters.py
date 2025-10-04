from __future__ import annotations

from tiny_rlhf.config import DatasetConfig
from tiny_rlhf.data import build_dataset


def test_multiple_choice_adapter(synthetic_mc_dataset):
    config = DatasetConfig(
        name="mc",
        type="multiple_choice",
        path=str(synthetic_mc_dataset["train"]),
        validation_path=str(synthetic_mc_dataset["validation"]),
        text_field="question",
        choice_field="options",
        answer_field="answer",
    )
    dataset = build_dataset(config)
    assert len(dataset.train) == 4
    assert len(dataset.validation) == 2
