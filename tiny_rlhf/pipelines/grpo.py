"""GRPO pipeline."""
from __future__ import annotations

import logging
from pathlib import Path

from tiny_rlhf.config import ExperimentConfig
from tiny_rlhf.data import build_dataset
from tiny_rlhf.logging import configure_logging
from tiny_rlhf.logging.wandb import WandbSession
from tiny_rlhf.models import load_model, save_model
from tiny_rlhf.rewards import build_reward
from tiny_rlhf.trainers import build_trainer

logger = logging.getLogger(__name__)


def run(config: ExperimentConfig) -> None:
    configure_logging()
    if config.reward is None:
        raise ValueError("GRPO pipeline requires a reward configuration")

    dataset = build_dataset(config.dataset)
    logger.info("Dataset summary: %s", dataset.summary())

    reward_fn = build_reward(config.reward.strategy, config.reward.weights)
    dummy_prompts = ["" for _ in dataset.train]
    dummy_responses = ["A" for _ in dataset.train]
    scores = reward_fn.score(dummy_prompts, dummy_responses)
    logger.info("Initial reward stats – mean: %.2f", sum(scores) / max(len(scores), 1))

    model_handle = load_model(config.model, config.lora)
    trainer = build_trainer(config.trainer, model_handle)

    wandb_config = config.run.logging
    if wandb_config.use_wandb and not wandb_config.project:
        raise ValueError("WandB logging requested but no project specified")

    output_dir = Path(config.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with WandbSession(wandb_config.project, wandb_config.run_name) if wandb_config.use_wandb else _null_context() as session:
        result = trainer.train(dataset)
        eval_metrics = trainer.evaluate(dataset)
        if session:
            session.log({**result.metrics, **eval_metrics})
        logger.info("Train metrics: %s", result.metrics)
        logger.info("Eval metrics: %s", eval_metrics)

    save_model(config.model, model_handle, str(output_dir / "checkpoints"))
    logger.info("Artifacts stored in %s", output_dir)


class _null_context:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False
