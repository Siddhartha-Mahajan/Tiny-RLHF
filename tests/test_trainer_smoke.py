from __future__ import annotations

from tiny_rlhf.config import DatasetConfig, ExperimentConfig, LoggingConfig, ModelConfig, RunConfig, TrainerConfig
from tiny_rlhf.data import build_dataset
from tiny_rlhf.models import load_model
from tiny_rlhf.trainers import build_trainer


def test_sft_trainer_smoke(monkeypatch):
    dataset_config = DatasetConfig(
        name="mc",
        type="multiple_choice",
        path="tests/fixtures/mc_train.jsonl",
        validation_path="tests/fixtures/mc_validation.jsonl",
        text_field="question",
        choice_field="options",
        answer_field="answer",
    )
    model_config = ModelConfig(
        name="mock",
        provider="transformers",
        pretrained_model_name_or_path="mock",
    )
    trainer_config = TrainerConfig(
        name="sft_test",
        algorithm="sft",
        max_steps=1,
        per_device_train_batch_size=1,
    )
    dataset = build_dataset(dataset_config)
    model = load_model(model_config)
    trainer = build_trainer(trainer_config, model)
    result = trainer.train(dataset)
    assert "train_loss" in result.metrics
