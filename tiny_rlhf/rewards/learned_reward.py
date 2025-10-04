"""Learned reward model placeholder."""
from __future__ import annotations

from typing import List

from .base import RewardFunction


class LearnedReward(RewardFunction):
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        return [0.2 for _ in responses]
