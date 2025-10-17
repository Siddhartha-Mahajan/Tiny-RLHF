"""Simple training entrypoints for SFT, DPO, and GRPO."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from .config import ExperimentConfig
from .data import DatasetBundle, load_dataset
from .modeling import ModelBundle, load_model, save_model
from .rewards import build_reward_functions
from .utils import configure_logging, maybe_wandb, set_seed

logger = logging.getLogger("tiny_rlhf")


def run_experiment(config: ExperimentConfig) -> Dict[str, float]:
    """Dispatch to the appropriate algorithm."""

    algorithm = config.trainer.algorithm.lower()
    if algorithm == "sft":
        return run_sft(config)
    if algorithm == "dpo":
        return run_dpo(config)
    if algorithm == "grpo":
        return run_grpo(config)
    raise ValueError(f"Unsupported algorithm: {config.trainer.algorithm}")


def _prepare(config: ExperimentConfig) -> tuple[ModelBundle, DatasetBundle]:
    model = load_model(config.model)
    data = load_dataset(config.dataset, tokenizer=model.tokenizer, algorithm=config.trainer.algorithm.lower())
    return model, data


def _finalise(model: ModelBundle, output_dir: str) -> None:
    save_model(model, output_dir)
    logger.info("Saved model to %s", output_dir)


def _ensure_output_dir(path: str) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_sft(config: ExperimentConfig) -> Dict[str, float]:
    configure_logging()
    set_seed(config.run.seed)
    model, data = _prepare(config)

    training_args = SFTConfig(
        output_dir=config.run.output_dir,
        dataset_text_field="text",
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_train_epochs=(
            config.trainer.num_train_epochs if config.trainer.num_train_epochs is not None else 1.0
        ),
        max_steps=config.trainer.max_steps or -1,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        warmup_ratio=config.trainer.warmup_ratio,
        logging_steps=config.run.logging_steps,
        eval_steps=config.run.eval_steps,
        save_steps=config.run.save_steps,
        evaluation_strategy="steps" if data.validation and config.run.eval_steps else "no",
        max_seq_length=config.trainer.max_prompt_length,
        report_to="wandb" if config.run.use_wandb else "none",
    )

    with maybe_wandb(config.run):
        trainer = SFTTrainer(
            model=model.model,
            args=training_args,
            tokenizer=model.tokenizer,
            train_dataset=data.train,
            eval_dataset=data.validation,
        )
        train_output = trainer.train()
        metrics = {"train_loss": float(train_output.training_loss)}
        if data.validation:
            metrics.update({f"eval_{k}": float(v) for k, v in trainer.evaluate().items()})
        _ensure_output_dir(config.run.output_dir)
        trainer.save_model(config.run.output_dir)

    _finalise(model, config.run.output_dir)
    return metrics


def run_dpo(config: ExperimentConfig) -> Dict[str, float]:
    configure_logging()
    set_seed(config.run.seed)
    model, data = _prepare(config)

    training_args = DPOConfig(
        output_dir=config.run.output_dir,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        num_train_epochs=(
            config.trainer.num_train_epochs if config.trainer.num_train_epochs is not None else 1.0
        ),
        max_steps=config.trainer.max_steps or -1,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        warmup_ratio=config.trainer.warmup_ratio,
        logging_steps=config.run.logging_steps,
        eval_steps=config.run.eval_steps,
        save_steps=config.run.save_steps,
        evaluation_strategy="steps" if data.validation and config.run.eval_steps else "no",
        beta=config.trainer.beta,
    )

    with maybe_wandb(config.run):
        trainer = DPOTrainer(
            model=model.model,
            ref_model=None,
            args=training_args,
            train_dataset=data.train,
            eval_dataset=data.validation,
            tokenizer=model.tokenizer,
        )
        train_output = trainer.train()
        metrics = {"train_loss": float(train_output.training_loss)}
        if data.validation:
            metrics.update({f"eval_{k}": float(v) for k, v in trainer.evaluate().items()})
        _ensure_output_dir(config.run.output_dir)
        trainer.save_model(config.run.output_dir)

    _finalise(model, config.run.output_dir)
    return metrics


def run_grpo(config: ExperimentConfig) -> Dict[str, float]:
    configure_logging()
    set_seed(config.run.seed)
    model, data = _prepare(config)

    reward_functions = build_reward_functions(config.reward, data.reward_targets)
    max_prompt_length = config.trainer.max_prompt_length or 512
    max_completion_length = config.trainer.max_completion_length or 256

    training_args = GRPOConfig(
        output_dir=config.run.output_dir,
        per_device_train_batch_size=config.trainer.per_device_train_batch_size,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        warmup_ratio=config.trainer.warmup_ratio,
        logging_steps=config.run.logging_steps,
        save_steps=config.run.save_steps,
        max_steps=config.trainer.max_steps or -1,
        num_generations=config.trainer.num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        report_to="wandb" if config.run.use_wandb else "none",
        use_vllm=False,
    )

    if hasattr(model.tokenizer, "padding_side"):
        model.tokenizer.padding_side = "left"

    with maybe_wandb(config.run):
        trainer = GRPOTrainer(
            model=model.model,
            reward_funcs=reward_functions,
            args=training_args,
            train_dataset=data.train,
            processing_class=model.tokenizer,
        )
        train_output = trainer.train()
        metrics = {"train_loss": float(train_output.training_loss)}
        _ensure_output_dir(config.run.output_dir)
        trainer.save_model(config.run.output_dir)

    _finalise(model, config.run.output_dir)
    return metrics
