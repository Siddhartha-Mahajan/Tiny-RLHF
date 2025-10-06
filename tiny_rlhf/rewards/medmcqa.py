"""MedMCQA reward helpers and reward functions."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .base import RewardFunction

_DEFAULT_MARKERS: Dict[str, str] = {
    "reasoning_start": "<start_working_out>",
    "reasoning_end": "<end_working_out>",
    "solution_start": "<SOLUTION>",
    "solution_end": "</SOLUTION>",
}


def _normalise_markers(markers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    combined = dict(_DEFAULT_MARKERS)
    if markers:
        combined.update({k: v for k, v in markers.items() if isinstance(v, str)})
    return combined


def _extract_between(text: str, start: str, end: str) -> Optional[str]:
    if not text:
        return None
    start_idx = text.find(start)
    if start_idx == -1:
        return None
    start_idx += len(start)
    end_idx = text.find(end, start_idx)
    if end_idx == -1:
        return text[start_idx:].strip() or None
    return text[start_idx:end_idx].strip() or None


def _normalise_label(candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    cleaned = candidate.strip()
    if not cleaned:
        return None
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned.upper()
    match = re.search(r"\b([A-H])\b", cleaned.upper())
    if match:
        return match.group(1)
    # Look for patterns like "Option: A" or "Answer - B"
    match = re.search(r"[\(\[:\s]([A-H])[\)\]\s\.]", cleaned.upper())
    if match:
        return match.group(1)
    return None


def _extract_answer_text(candidate: Optional[str]) -> Optional[str]:
    if candidate is None:
        return None
    cleaned = candidate.strip()
    return cleaned or None


@dataclass
class ParsedMedMCQAResponse:
    """Structured representation of a MedMCQA model response."""

    raw: str
    reasoning: Optional[str]
    solution: Optional[str]
    answer_label: Optional[str]
    answer_text: Optional[str]


def parse_medmcqa_response(response: str, markers: Optional[Dict[str, str]] = None) -> ParsedMedMCQAResponse:
    markers = _normalise_markers(markers)
    reasoning = _extract_between(response, markers["reasoning_start"], markers["reasoning_end"])
    solution = _extract_between(response, markers["solution_start"], markers["solution_end"])
    answer_label = _normalise_label(solution)
    answer_text = _extract_answer_text(solution)
    return ParsedMedMCQAResponse(
        raw=response,
        reasoning=reasoning,
        solution=solution,
        answer_label=answer_label,
        answer_text=answer_text,
    )


def format_reward_score(parsed: ParsedMedMCQAResponse) -> float:
    """Score format compliance: reasoning markers + labelled answer."""
    score = 0.0
    if parsed.reasoning:
        score += 0.5
    if parsed.answer_label:
        score += 0.5
    return score


def accuracy_reward_score(
    parsed: ParsedMedMCQAResponse,
    reference: Optional[Dict[str, Optional[str]]] = None,
) -> float:
    """Compare extracted answer against reference label/text."""
    if reference is None:
        return 0.0

    gold_label = reference.get("answer_label") if reference else None
    gold_text = reference.get("answer_text") if reference else None

    if gold_label and parsed.answer_label:
        return 1.0 if parsed.answer_label == gold_label else 0.0

    if gold_text and parsed.answer_text:
        return 1.0 if parsed.answer_text.strip().lower() == gold_text.strip().lower() else 0.0

    return 0.0


class MedMCQAFormatReward(RewardFunction):
    """Reward that encourages the model to obey MedMCQA formatting rules."""

    def __init__(self, markers: Optional[Dict[str, str]] = None):
        self.markers = _normalise_markers(markers)

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        scores: List[float] = []
        for response in responses:
            parsed = parse_medmcqa_response(response, self.markers)
            scores.append(format_reward_score(parsed))
        return scores


class MedMCQAAccuracyReward(RewardFunction):
    """Reward that scores answers by accuracy given references."""

    def __init__(
        self,
        references: Iterable[Dict[str, Optional[str]]],
        markers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.references: List[Dict[str, Optional[str]]] = [dict(ref) for ref in references]
        self.markers = _normalise_markers(markers)

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        if len(responses) != len(self.references):
            raise ValueError("Number of responses must match number of references for accuracy scoring")
        scores: List[float] = []
        for response, reference in zip(responses, self.references):
            parsed = parse_medmcqa_response(response, self.markers)
            scores.append(accuracy_reward_score(parsed, reference))
        return scores


class MedMCQAReward(RewardFunction):
    """Combined MedMCQA reward aggregating format and accuracy components."""

    def __init__(
        self,
        references: Optional[Iterable[Dict[str, Optional[str]]]] = None,
        markers: Optional[Dict[str, str]] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.references = list(references) if references is not None else None
        self.markers = _normalise_markers(markers)
        base_weights = {"format": 1.0, "accuracy": 1.0}
        if weights:
            base_weights.update(weights)
        self.weights = base_weights

    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        scores: List[float] = []
        for idx, response in enumerate(responses):
            parsed = parse_medmcqa_response(response, self.markers)
            total = self.weights.get("format", 1.0) * format_reward_score(parsed)
            if self.references is not None:
                if idx >= len(self.references):
                    raise ValueError("Not enough references supplied for MedMCQAReward")
                total += self.weights.get("accuracy", 1.0) * accuracy_reward_score(parsed, self.references[idx])
            scores.append(total)
        return scores


__all__ = [
    "ParsedMedMCQAResponse",
    "parse_medmcqa_response",
    "format_reward_score",
    "accuracy_reward_score",
    "MedMCQAFormatReward",
    "MedMCQAAccuracyReward",
    "MedMCQAReward",
]
