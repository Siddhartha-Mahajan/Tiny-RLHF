"""Handles optional heavy dependencies so other modules can import safely."""
from __future__ import annotations

try:  # Unsloth accelerated model loader
    from unsloth import FastLanguageModel
except Exception:  # pragma: no cover - optional dependency
    FastLanguageModel = None  # type: ignore

try:  # TRL trainers/configs
    from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
except Exception:  # pragma: no cover - optional dependency
    SFTTrainer = SFTConfig = GRPOTrainer = GRPOConfig = None  # type: ignore

try:  # vLLM sampling params
    from vllm import SamplingParams
except Exception:  # pragma: no cover - optional dependency
    SamplingParams = None  # type: ignore

__all__ = [
    "FastLanguageModel",
    "SFTTrainer",
    "SFTConfig",
    "GRPOTrainer",
    "GRPOConfig",
    "SamplingParams",
]
