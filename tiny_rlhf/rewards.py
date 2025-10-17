"""Reward builders for GRPO training."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Callable, Iterable, List, Optional

from .config import RewardConfig

RewardFn = Callable[..., List[float]]


def _load_script(path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location("tiny_rlhf_reward_module", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"Unable to load reward script at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _exact_match_reward(targets: Iterable[str]) -> RewardFn:
    answers = list(targets)

    def reward(completions, prompts=None, completion_ids=None, **kwargs):  # type: ignore[override]
        scores: List[float] = []
        for idx, completion in enumerate(completions):
            target_idx = _resolve_index(idx, completion_ids)
            target = answers[target_idx] if 0 <= target_idx < len(answers) else ""
            if not target:
                scores.append(0.0)
            else:
                scores.append(float(target.lower() in str(completion).lower()))
        return scores

    return reward


def _keyword_reward(keywords: Iterable[str]) -> RewardFn:
    tokens = [token.lower() for token in keywords if token]

    def reward(completions, **kwargs):  # type: ignore[override]
        scores: List[float] = []
        for completion in completions:
            text = str(completion).lower()
            scores.append(float(any(token in text for token in tokens)))
        return scores

    return reward


def _resolve_index(default_index: int, completion_ids) -> int:
    if completion_ids is None:
        return default_index
    try:
        candidate = completion_ids[default_index]
        if isinstance(candidate, (list, tuple)) and candidate:
            candidate = candidate[0]
        return int(candidate)
    except Exception:
        try:
            return int(completion_ids)
        except Exception:
            return default_index


def build_reward_functions(config: Optional[RewardConfig], targets: Optional[List[str]]) -> List[RewardFn]:
    """Create reward callables used by GRPOTrainer."""

    if config is None or config.type == "exact_match":
        targets = targets or []
        return [_exact_match_reward(targets)]

    if config.type == "keyword":
        keywords = config.keywords or ([] if config.keyword is None else [config.keyword])
        if not keywords:
            raise ValueError("Keyword reward requested but no keywords supplied.")
        return [_keyword_reward(keywords)]

    if config.type == "script":
        if not config.script_path:
            raise ValueError("Reward script path must be provided when type='script'.")
        module = _load_script(config.script_path)
        fn = getattr(module, config.function_name, None)
        if not callable(fn):
            raise AttributeError(
                f"Reward module {config.script_path} has no callable '{config.function_name}'."
            )
        reward_fn = fn(targets)
        if not callable(reward_fn):
            raise TypeError("Reward factory must return a callable")
        return [reward_fn]

    if config.type == "none":
        return [lambda completions, **_: [0.0 for _ in completions]]

    raise ValueError(f"Unsupported reward type: {config.type}")
