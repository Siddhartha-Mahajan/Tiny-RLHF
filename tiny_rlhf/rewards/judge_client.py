"""LLM judge client placeholder."""
from __future__ import annotations

from typing import List

from .base import RewardFunction


class PlaceholderJudge(RewardFunction):
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        return [0.5 for _ in responses]
