"""Judge registry."""
from __future__ import annotations

from typing import Dict, Type

from tiny_rlhf.config import JudgeConfig

from .base import JudgeBackend
from .local_model_judge import LocalModelJudge
from .openai_judge import OpenAIJudge

_JUDGES: Dict[str, Type[JudgeBackend]] = {
    "placeholder": LocalModelJudge,
    "local": LocalModelJudge,
    "openai": OpenAIJudge,
}


def build_judge(config: JudgeConfig) -> JudgeBackend:
    judge_cls = _JUDGES.get(config.provider)
    if judge_cls is None:
        raise KeyError(f"Unknown judge provider: {config.provider}")
    if judge_cls is OpenAIJudge:
        return judge_cls(model_name=config.model_name or "gpt-4o-mini", api_key=config.api_key)
    return judge_cls()


__all__ = ["build_judge"]
