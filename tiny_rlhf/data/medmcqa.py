"""MedMCQA dataset adapter."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer

from .base import DatasetAdapter, DatasetSplits

logger = logging.getLogger(__name__)


def _normalize_option(text: Any) -> str:
    return "" if text is None else str(text).strip()


def _extract_options(raw_example: Dict[str, Any]) -> List[str]:
    options: List[str] = []
    if isinstance(raw_example.get("options"), (list, tuple)):
        options = [_normalize_option(o) for o in raw_example["options"]]
    elif isinstance(raw_example.get("choices"), (list, tuple)):
        options = [_normalize_option(o) for o in raw_example["choices"]]
    else:
        # Look for keys like option1 / op1 etc.
        candidates: List[str] = []
        for key, value in raw_example.items():
            if not isinstance(key, str):
                continue
            key_lower = key.lower()
            if key_lower.startswith("option") or key_lower.startswith("op"):
                candidates.append((key_lower, _normalize_option(value)))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            options = [value for _, value in candidates]
    if not options:
        # explicit fallback
        for key in ("option1", "option2", "option3", "option4"):
            if raw_example.get(key):
                options.append(_normalize_option(raw_example[key]))
    return [opt for opt in options if opt]


def _determine_answer(raw_example: Dict[str, Any], options: List[str]) -> Dict[str, Optional[str]]:
    answer_label: Optional[str] = None
    answer_text: Optional[str] = None

    def _label_from_index(index: int) -> Optional[str]:
        if 0 <= index < len(options):
            return chr(ord("A") + index)
        return None

    possible = None
    for key in ("cop", "answer", "label", "ans", "correct", "correct_option"):
        if key in raw_example and raw_example[key] not in (None, ""):
            possible = raw_example[key]
            break

    if possible is not None:
        try:
            idx = int(possible)
            if idx == -1:
                answer_label = None
            else:
                if 0 <= idx < len(options):
                    answer_label = _label_from_index(idx)
                elif 1 <= idx <= len(options):
                    answer_label = _label_from_index(idx - 1)
        except (TypeError, ValueError):
            possible_str = str(possible).strip()
            if len(possible_str) == 1 and possible_str.isalpha():
                answer_label = possible_str.upper()
            else:
                # try match text
                for opt_index, option in enumerate(options):
                    if possible_str.lower() == option.lower() or possible_str in option:
                        answer_label = _label_from_index(opt_index)
                        answer_text = option
                        break
                if answer_text is None:
                    answer_text = possible_str

    if answer_label and answer_text is None:
        index = ord(answer_label) - ord("A")
        if 0 <= index < len(options):
            answer_text = options[index]
    if answer_text and not answer_label and options:
        answer_text_norm = answer_text.strip().lower()
        for opt_index, option in enumerate(options):
            if answer_text_norm == option.lower() or answer_text_norm in option.lower():
                answer_label = _label_from_index(opt_index)
                break

    # fallback explicit stored answers
    if answer_text is None and isinstance(raw_example.get("answer"), str):
        answer_text = raw_example["answer"].strip()

    return {"answer_label": answer_label, "answer_text": answer_text}


def _build_chat_template(tokenizer) -> Dict[str, str]:
    reasoning_start = "<start_working_out>"
    reasoning_end = "<end_working_out>"
    solution_start = "<SOLUTION>"
    solution_end = "</SOLUTION>"

    system_prompt = (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    chat_template = (
        "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
        "{% set loop_messages = messages[1:] %}"
        "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
        "{% set loop_messages = messages %}"
        "{% endif %}"
        "{% for message in loop_messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ message['content'] + eos_token }}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}" + reasoning_start + "{% endif %}"
    )

    try:
        tokenizer.chat_template = chat_template
    except Exception:
        setattr(tokenizer, "chat_template", chat_template)

    return {
        "system_prompt": system_prompt,
        "reasoning_start": reasoning_start,
        "reasoning_end": reasoning_end,
        "solution_start": solution_start,
        "solution_end": solution_end,
    }


def _format_messages(
    question: str,
    options: List[str],
    markers: Dict[str, str],
    answer_label: Optional[str],
    answer_text: Optional[str],
    raw_example: Dict[str, Any],
) -> List[Dict[str, str]]:
    system_prompt = markers["system_prompt"]
    reasoning_start = markers["reasoning_start"]
    reasoning_end = markers["reasoning_end"]
    solution_start = markers["solution_start"]
    solution_end = markers["solution_end"]

    if options:
        labeled = [f"{chr(ord('A') + idx)}. {opt}" for idx, opt in enumerate(options)]
        choice_text = "\n".join(labeled)
        user_prompt = (
            f"{question}\n\n"
            f"Options:\n{choice_text}\n\n"
            f"Please answer with the OPTION LABEL only (for example 'A' or 'B').\n"
            f"Place the final option label between {solution_start}{solution_end}."
        )
    else:
        user_prompt = (
            f"{question}\n\n"
            f"Please place the final answer between {solution_start}{solution_end}."
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    raw_fields = raw_example.get("raw") if isinstance(raw_example.get("raw"), dict) else raw_example
    gold_reasoning: Optional[str] = None
    if isinstance(raw_fields, dict):
        for key in ("exp", "explanation", "explain"):
            if raw_fields.get(key):
                gold_reasoning = str(raw_fields[key]).strip()
                break

    if answer_label:
        solution_text = answer_label
    elif answer_text:
        solution_text = answer_text
    else:
        solution_text = ""

    if gold_reasoning or solution_text:
        assistant_content = (
            f"{reasoning_start}{gold_reasoning or ''}{reasoning_end}"
            f"{solution_start}{solution_text}{solution_end}"
        )
        messages.append({"role": "assistant", "content": assistant_content})

    return messages


def _process_split(entries: Iterable[Dict[str, Any]], markers: Dict[str, str]) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for idx, raw in enumerate(entries):
        question = raw.get("question") or raw.get("Question") or raw.get("prompt") or ""
        options = _extract_options(raw)
        answers = _determine_answer(raw, options)
        processed_example = {
            "id": raw.get("id", idx),
            "question": question,
            "options": options,
            "answer_label": answers.get("answer_label"),
            "answer_text": answers.get("answer_text"),
            "raw": raw,
        }
        messages = _format_messages(
            question,
            options,
            markers,
            processed_example["answer_label"],
            processed_example["answer_text"],
            processed_example,
        )
        processed_example["messages"] = messages
        formatted.append(processed_example)
    return formatted


class MedMCQAAdapter(DatasetAdapter):
    """Adapter that downloads and formats the MedMCQA dataset."""

    DATASET_NAME = "openlifescienceai/medmcqa"

    def _load_dataset(self) -> DatasetDict:
        dataset_name = self.config.extras.get("dataset_name", self.DATASET_NAME)
        logger.info("Loading MedMCQA dataset: %s", dataset_name)
        return load_dataset(dataset_name)

    def build(self) -> DatasetSplits:
        cache_root = Path(self.config.path)
        cache_root.mkdir(parents=True, exist_ok=True)

        tokenizer_name = self.config.extras.get("tokenizer_name")
        if not tokenizer_name:
            tokenizer_name = self.config.extras.get("model_name")
        if not tokenizer_name:
            tokenizer_name = "Qwen/Qwen2-1.5B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        markers = _build_chat_template(tokenizer)

        raw_dataset = self._load_dataset()
        max_train_samples = self.config.extras.get("max_train_samples")
        max_eval_samples = self.config.extras.get("max_eval_samples")

        def _truncate(split_examples: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
            if limit is None or limit <= 0:
                return split_examples
            return split_examples[:limit]

        train_split = raw_dataset.get("train") or raw_dataset.get("training")
        validation_split = raw_dataset.get("validation") or raw_dataset.get("dev")
        test_split = raw_dataset.get("test")

        train_entries = _truncate(train_split.to_list(), max_train_samples) if train_split is not None else []
        validation_entries = _truncate(validation_split.to_list(), max_eval_samples) if validation_split is not None else []
        test_entries = test_split.to_list() if test_split is not None else []

        train = _process_split(train_entries, markers)
        validation = _process_split(validation_entries, markers)
        test = _process_split(test_entries, markers) if test_entries else None

        metadata = {
            "markers": markers,
            "tokenizer_name": tokenizer_name,
            "max_seq_length": self.config.extras.get("max_seq_length", self.config.extras.get("seq_len", 2048)),
        }

        logger.info(
            "Prepared MedMCQA dataset – train=%d, validation=%d, test=%d",
            len(train),
            len(validation),
            len(test or []),
        )

        return DatasetSplits(train=train, validation=validation, test=test, metadata=metadata)