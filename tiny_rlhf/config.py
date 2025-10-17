"""Minimal configuration primitives for TinyRLHF."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RunConfig:
    """Runtime settings not tied to a specific algorithm."""

    output_dir: str = "outputs"
    logging_steps: int = 10
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    seed: Optional[int] = 42
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class DatasetConfig:
    """Dataset layout description.

    format controls how examples are interpreted:
    - "chat": expect a ``messages`` list compatible with chat templates.
    - "instruction": expect free-form ``prompt``/``completion`` strings.
    - "preference": expect ``prompt``/``chosen``/``rejected`` strings.
    - "grpo": expect ``prompt`` strings and optional ``answer`` targets.
    """

    format: str = "instruction"
    train_file: str = ""
    validation_file: Optional[str] = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    messages_field: str = "messages"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    answer_field: str = "answer"


@dataclass
class ModelConfig:
    """How to load and optionally wrap the base model."""

    pretrained: str
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_peft: bool = False
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


@dataclass
class TrainerConfig:
    """Common trainer hyper-parameters."""

    algorithm: str
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_train_epochs: Optional[float] = None
    max_steps: Optional[int] = None
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    beta: float = 0.1  # Used by DPO
    num_generations: int = 4  # Used by GRPO
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None


@dataclass
class RewardConfig:
    """Reward shaping options for GRPO."""

    type: str = "exact_match"  # exact_match | keyword | script | none
    keyword: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    script_path: Optional[str] = None
    function_name: str = "build_reward"


@dataclass
class ExperimentConfig:
    """Full experiment description."""

    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    run: RunConfig = field(default_factory=RunConfig)
    reward: Optional[RewardConfig] = None


def _filter_kwargs(data: Dict[str, Any], cls: type) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {key: value for key, value in (data or {}).items() if key in allowed}


def _build(cls: type, data: Dict[str, Any] | None) -> Any:
    if data is None:
        return cls(**{})
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping to construct {cls.__name__}, got {type(data).__name__}")
    return cls(**_filter_kwargs(data, cls))


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Parse a YAML experiment file into typed configuration."""

    with open(Path(path), "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Experiment config must be a mapping at the top level.")

    dataset = _build(DatasetConfig, payload.get("dataset"))
    model = _build(ModelConfig, payload.get("model"))
    trainer = _build(TrainerConfig, payload.get("trainer"))
    run = _build(RunConfig, payload.get("run"))
    reward_cfg = payload.get("reward")
    reward = _build(RewardConfig, reward_cfg) if reward_cfg is not None else None

    return ExperimentConfig(dataset=dataset, model=model, trainer=trainer, run=run, reward=reward)


__all__ = [
    "DatasetConfig",
    "ExperimentConfig",
    "ModelConfig",
    "RewardConfig",
    "RunConfig",
    "TrainerConfig",
    "load_experiment",
]
