"""Reward function abstractions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List


class RewardFunction(ABC):
    @abstractmethod
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Return a reward score per response."""


class ConstantReward(RewardFunction):
    def __init__(self, value: float = 0.0):
        self.value = value

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        return [self.value for _ in responses]
