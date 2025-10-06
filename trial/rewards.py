"""Reward helpers for GRPO training (self-contained inside ``trial``)."""
from __future__ import annotations

import os
import re
from typing import Optional, Tuple

USE_EMBEDDINGS = bool(int(os.environ.get("TRIAL_REWARDS_USE_EMBEDDINGS", "0")))

if USE_EMBEDDINGS:  # pragma: no cover - optional dependency path
    try:
        from sentence_transformers import SentenceTransformer, util  # type: ignore
    except Exception:  # fallback gracefully if library unavailable
        SentenceTransformer = util = None  # type: ignore
        USE_EMBEDDINGS = False
else:  # pragma: no cover - env disabled
    SentenceTransformer = util = None  # type: ignore

_EMBEDDING_MODEL = None
_EMBEDDING_TRIED = False


def extract_between(text: Optional[str], start: Optional[str], end: Optional[str]) -> Optional[str]:
    if not text or not start or not end:
        return None
    s = text.find(start)
    if s == -1:
        return None
    s2 = text.find(end, s + len(start))
    if s2 == -1:
        return text[s + len(start) :].strip()
    return text[s + len(start) : s2].strip()


def _normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _jaccard_similarity(a: Optional[str], b: Optional[str]) -> float:
    a_norm = _normalize_text(a)
    b_norm = _normalize_text(b)
    if not a_norm or not b_norm:
        return 0.0
    set_a = set(a_norm.split())
    set_b = set(b_norm.split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _embedding_similarity(a: Optional[str], b: Optional[str]) -> float:
    global _EMBEDDING_MODEL, _EMBEDDING_TRIED

    if not USE_EMBEDDINGS or SentenceTransformer is None or util is None:
        return _jaccard_similarity(a, b)
    if not a or not b:
        return 0.0

    if not _EMBEDDING_TRIED:
        _EMBEDDING_TRIED = True
        try:
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _EMBEDDING_MODEL = None

    if _EMBEDDING_MODEL is None:
        return _jaccard_similarity(a, b)

    emb_a = _EMBEDDING_MODEL.encode(a, convert_to_tensor=True)
    emb_b = _EMBEDDING_MODEL.encode(b, convert_to_tensor=True)
    cos = util.cos_sim(emb_a, emb_b).item()
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


def _derive_gold(example: dict) -> Tuple[Optional[str], Optional[str]]:
    gold_label = example.get("answer_label")
    gold_text = example.get("answer_text")
    raw = example.get("raw") or {}

    if isinstance(raw, dict):
        if gold_label is None and raw.get("answer_label"):
            gold_label = raw.get("answer_label")
        if gold_text is None and raw.get("answer_text"):
            gold_text = raw.get("answer_text")
        if gold_text is None and raw.get("answer"):
            gold_text = raw.get("answer")

        cop = raw.get("cop")
        options = raw.get("options") or example.get("options") or []
        if cop is not None and cop not in ("", None) and gold_label is None and gold_text is None:
            try:
                idx = int(cop)
                if idx == -1:
                    pass
                elif 0 <= idx < len(options):
                    gold_label = chr(ord("A") + idx)
                    gold_text = options[idx]
                elif 1 <= idx <= len(options):
                    gold_label = chr(ord("A") + (idx - 1))
                    gold_text = options[idx - 1]
            except Exception:
                cop_str = str(cop).strip()
                if cop_str:
                    gold_text = gold_text or cop_str

    return gold_label, gold_text


def match_format_exactly(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    gold_label, _ = _derive_gold(example)
    if not gold_label:
        return 0.0

    pred = extract_between(gen_text or "", markers.get("solution_start"), markers.get("solution_end"))
    if not pred:
        match = re.search(r"\b([A-Z])\b", (gen_text or "").upper())
        pred = match.group(1) if match else None

    if pred and pred.strip().upper() == str(gold_label).strip().upper():
        return 1.0
    if pred and str(gold_label).strip().upper() in pred.strip().upper():
        return 1.0
    return 0.0


def match_explanation_similarity(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    raw = example.get("raw") or {}
    gold_exp = None
    for key in ("exp", "explanation", "explain"):
        if isinstance(raw, dict) and raw.get(key):
            gold_exp = str(raw.get(key)).strip()
            break
    if not gold_exp:
        return 0.0

    reasoning = extract_between(gen_text or "", markers.get("reasoning_start"), markers.get("reasoning_end"))
    if not reasoning:
        sol_idx = (gen_text or "").find(markers.get("solution_start") or "")
        if sol_idx > 0:
            candidate = (gen_text or "")[:sol_idx].strip()
            reasoning = candidate or None
    if not reasoning:
        return 0.0

    return _embedding_similarity(reasoning, gold_exp)


def check_answer(
    gen_text: str,
    prompt: str,
    example: dict,
    markers: dict,
    label_weight: float = 1.0,
    exp_weight: float = 0.4,
) -> float:
    label_score = match_format_exactly(gen_text, prompt, example, markers)
    exp_score = match_explanation_similarity(gen_text, prompt, example, markers)
    return float(label_weight * label_score + exp_weight * exp_score)


def match_format_approximately(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    return check_answer(gen_text, prompt, example, markers, label_weight=0.9, exp_weight=0.3)


def check_numbers(gen_text: str, prompt: str, example: dict, markers: dict) -> float:
    return 0.0


__all__ = [
    "match_format_exactly",
    "match_format_approximately",
    "check_answer",
    "check_numbers",
    "extract_between",
]
