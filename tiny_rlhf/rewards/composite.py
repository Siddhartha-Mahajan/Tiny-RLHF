"""Composite reward assembly."""
from __future__ import annotations

from typing import Dict, List

from .base import RewardFunction
from .formatting import LabelFormatReward
from .judge_client import PlaceholderJudge
from .learned_reward import LearnedReward


class CompositeReward(RewardFunction):
    def __init__(self, components: Dict[str, RewardFunction], weights: Dict[str, float] | None = None):
        self.components = components
        self.weights = weights or {name: 1.0 for name in components}

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        total = [0.0 for _ in responses]
        for name, component in self.components.items():
            scores = component.score(prompts, responses)
            weight = self.weights.get(name, 1.0)
            total = [t + weight * s for t, s in zip(total, scores)]
        return total


def build_reward(strategy: str, weights: Dict[str, float] | None = None) -> RewardFunction:
    if strategy == "format_only":
        return LabelFormatReward()
    if strategy == "judge_only":
        return PlaceholderJudge()
    if strategy == "learned":
        return LearnedReward()
    if strategy == "format_and_accuracy":
        components = {
            "format": LabelFormatReward(),
            "judge": PlaceholderJudge(),
        }
        return CompositeReward(components, weights)
    raise KeyError(f"Unknown reward strategy: {strategy}")
