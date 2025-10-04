"""Simple formatting rewards."""
from __future__ import annotations

from typing import List

from .base import RewardFunction


class LabelFormatReward(RewardFunction):
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        scores = []
        for response in responses:
            response = response.strip()
            if len(response) == 1 and response.isalpha():
                scores.append(1.0)
            else:
                scores.append(0.0)
        return scores
