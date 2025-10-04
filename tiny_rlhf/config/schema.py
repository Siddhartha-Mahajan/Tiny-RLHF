"""Configuration schemas for TinyRLHF."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, root_validator


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project: Optional[str] = None
    run_name: Optional[str] = None


class RunConfig(BaseModel):
    pipeline: str
    output_dir: str = "outputs/default"
    seed: int = 42
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    extra: Dict[str, Any] = Field(default_factory=dict)


class DatasetConfig(BaseModel):
    name: str
    type: Literal["multiple_choice", "freeform", "preference_pairs"]
    path: str
    validation_path: Optional[str] = None
    test_path: Optional[str] = None
    text_field: Optional[str] = None
    choice_field: Optional[str] = None
    answer_field: Optional[str] = None
    input_field: Optional[str] = None
    target_field: Optional[str] = None
    input_template: Optional[str] = None
    target_template: Optional[str] = None
    system_prompt: Optional[str] = None
    format: Optional[str] = None
    fields: Dict[str, str] = Field(default_factory=dict)


class LoRAConfig(BaseModel):
    provider: Literal["peft", "unsloth", "none"] = "peft"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


class ModelConfig(BaseModel):
    name: str
    provider: Literal["transformers", "unsloth"] = "transformers"
    pretrained_model_name_or_path: str
    torch_dtype: Optional[str] = None
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    max_seq_length: Optional[int] = None


class TrainerConfig(BaseModel):
    name: str
    algorithm: Literal["sft", "grpo", "dpo", "ppo"]
    max_steps: Optional[int] = None
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: Optional[int] = None
    rollout_batch_size: Optional[int] = None
    num_generations: Optional[int] = None
    kl_coeff: Optional[float] = None
    reward_clip: Optional[float] = None
    beta: Optional[float] = None


class JudgeConfig(BaseModel):
    provider: Literal["placeholder", "openai", "local"] = "placeholder"
    model_name: Optional[str] = None
    api_key: Optional[str] = None


class RewardConfig(BaseModel):
    strategy: Literal["format_only", "format_and_accuracy", "learned", "judge_only"] = "format_only"
    judge: Optional[JudgeConfig] = None
    weights: Dict[str, float] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    run: RunConfig
    lora: Optional[LoRAConfig] = None
    reward: Optional[RewardConfig] = None

    @root_validator(pre=True)
    def ensure_paths_are_strings(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        for section in ("dataset", "model", "run"):
            if section in values and isinstance(values[section], dict):
                for key, val in values[section].items():
                    if isinstance(val, Path):
                        values[section][key] = str(val)
        return values


class RegistryConfig(BaseModel):
    datasets: Dict[str, str]
    models: Dict[str, str]
    trainers: Dict[str, str]
    pipelines: Dict[str, str]
