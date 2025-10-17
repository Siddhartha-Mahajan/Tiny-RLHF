"""Dataset helpers for TinyRLHF."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import json

from datasets import Dataset

from .config import DatasetConfig


@dataclass
class DatasetBundle:
    """Container for train / validation splits and optional reward targets."""

    train: Dataset
    validation: Optional[Dataset] = None
    reward_targets: Optional[List[str]] = None


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(Path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _render_messages(tokenizer: Any, messages: Iterable[Dict[str, str]], *, add_generation_prompt: bool) -> str:
    messages_list = list(messages)
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except Exception:
            pass

    rendered: List[str] = []
    for message in messages_list:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        rendered.append(f"{role}: {content}")
    if add_generation_prompt:
        rendered.append("ASSISTANT:")
    return "\n\n".join(rendered)


def _prepare_sft_records(
    rows: List[Dict[str, Any]],
    config: DatasetConfig,
    tokenizer: Any,
) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for row in rows:
        if config.format == "chat":
            messages = row.get(config.messages_field)
            if not isinstance(messages, list):
                continue
            text = _render_messages(tokenizer, messages, add_generation_prompt=False)
        else:
            prompt = row.get(config.prompt_field, "")
            completion = row.get(config.completion_field, "")
            if prompt is None or completion is None:
                continue
            text = f"{prompt}{completion}"
        records.append({"text": text})
    return records


def _prepare_preference_records(rows: List[Dict[str, Any]], config: DatasetConfig) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for row in rows:
        prompt = row.get(config.prompt_field)
        chosen = row.get(config.chosen_field)
        rejected = row.get(config.rejected_field)
        if prompt is None or chosen is None or rejected is None:
            continue
        records.append({
            "prompt": str(prompt),
            "chosen": str(chosen),
            "rejected": str(rejected),
        })
    return records


def _prepare_grpo_records(rows: List[Dict[str, Any]], config: DatasetConfig, tokenizer: Any) -> DatasetBundle:
    prompts: List[str] = []
    answers: List[str] = []
    for row in rows:
        if config.format == "chat":
            messages = row.get(config.messages_field)
            if not isinstance(messages, list):
                continue
            prompt_text = _render_messages(tokenizer, messages, add_generation_prompt=True)
        else:
            prompt_text = row.get(config.prompt_field)
            if prompt_text is None:
                continue
        prompts.append(str(prompt_text))
        answers.append(str(row.get(config.answer_field, "")))
    train_dataset = Dataset.from_dict({"prompt": prompts, "answer": answers})
    return DatasetBundle(train=train_dataset, reward_targets=answers)


def load_dataset(config: DatasetConfig, *, tokenizer: Any, algorithm: str) -> DatasetBundle:
    """Load raw JSONL files into HF datasets suitable for the chosen pipeline."""

    train_rows = _read_jsonl(config.train_file)
    val_rows = _read_jsonl(config.validation_file) if config.validation_file else []

    if algorithm == "sft":
        records_train = _prepare_sft_records(train_rows, config, tokenizer)
        records_val = _prepare_sft_records(val_rows, config, tokenizer) if val_rows else []
        if not records_train:
            raise ValueError(f"No SFT records could be built from {config.train_file}")
        train_dataset = Dataset.from_list(records_train)
        val_dataset = Dataset.from_list(records_val) if records_val else None
        return DatasetBundle(train=train_dataset, validation=val_dataset)

    if algorithm == "dpo":
        records_train = _prepare_preference_records(train_rows, config)
        records_val = _prepare_preference_records(val_rows, config) if val_rows else []
        if not records_train:
            raise ValueError(f"No DPO records could be built from {config.train_file}")
        train_dataset = Dataset.from_list(records_train)
        val_dataset = Dataset.from_list(records_val) if records_val else None
        return DatasetBundle(train=train_dataset, validation=val_dataset)

    if algorithm == "grpo":
        bundle = _prepare_grpo_records(train_rows, config, tokenizer)
        if len(bundle.train) == 0:
            raise ValueError(f"No GRPO records could be built from {config.train_file}")
        return bundle

    raise ValueError(f"Unsupported algorithm: {algorithm}")
