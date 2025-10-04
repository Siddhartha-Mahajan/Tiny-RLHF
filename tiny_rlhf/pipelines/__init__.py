"""Pipeline registry."""
from __future__ import annotations

from typing import Callable, Dict

from tiny_rlhf.config import ExperimentConfig

from .dpo import run as run_dpo
from .grpo import run as run_grpo
from .sft import run as run_sft

PipelineFn = Callable[[ExperimentConfig], None]

_PIPELINES: Dict[str, PipelineFn] = {
    "sft": run_sft,
    "grpo": run_grpo,
    "dpo": run_dpo,
}


def get_pipeline(name: str) -> PipelineFn:
    pipeline = _PIPELINES.get(name)
    if pipeline is None:
        raise KeyError(f"Unknown pipeline: {name}")
    return pipeline


__all__ = ["get_pipeline"]
