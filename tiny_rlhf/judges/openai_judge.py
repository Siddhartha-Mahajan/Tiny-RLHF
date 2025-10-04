"""OpenAI judge placeholder."""
from __future__ import annotations

from typing import List

from .base import JudgeBackend


class OpenAIJudge(JudgeBackend):
    def __init__(self, model_name: str, api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        raise NotImplementedError("OpenAI judge integration is not implemented yet")
