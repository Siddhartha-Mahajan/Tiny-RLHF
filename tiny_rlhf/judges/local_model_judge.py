"""Local model judge placeholder."""
from __future__ import annotations

from typing import List

from .base import JudgeBackend


class LocalModelJudge(JudgeBackend):
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        return [0.0 for _ in responses]
