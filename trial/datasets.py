"""Dataset download and preparation utilities."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from datasets import load_dataset  # type: ignore
from .utils import makedirs


def download_medmcqa(out_dir: str) -> Dict[str, str]:
    """Download the MedMCQA dataset and dump processed JSONL splits."""
    makedirs(out_dir)
    print("Loading dataset: openlifescienceai/medmcqa ...")
    ds_all = load_dataset("openlifescienceai/medmcqa")
    saved: Dict[str, str] = {}

    for split, dataset in ds_all.items():
        processed: List[Dict[str, object]] = []
        for idx, example in enumerate(dataset):
            question = (
                example.get("question")
                or example.get("Question")
                or example.get("prompt")
                or ""
            )

            options: List[str] = []
            if example.get("options") and isinstance(example.get("options"), (list, tuple)):
                options = [str(x).strip() for x in example["options"]]
            elif example.get("choices") and isinstance(example.get("choices"), (list, tuple)):
                options = [str(x).strip() for x in example["choices"]]
            else:
                option_candidates = []
                for key, value in example.items():
                    if isinstance(key, str) and key.lower().startswith("op") and value is not None:
                        option_candidates.append((key, str(value).strip()))
                if option_candidates:
                    option_candidates.sort(key=lambda pair: pair[0])
                    options = [text for _, text in option_candidates]

            if not options:
                for key in ["option1", "option2", "option3", "option4"]:
                    if example.get(key):
                        options.append(str(example[key]).strip())

            answer_label: Optional[str] = None
            answer_text: Optional[str] = None
            possible_answer = None
            for candidate in [
                "cop",
                "answer",
                "label",
                "ans",
                "correct",
                "correct_option",
            ]:
                if candidate in example and example[candidate] not in (None, ""):
                    possible_answer = example[candidate]
                    break

            if possible_answer is not None:
                try:
                    idx = int(possible_answer)
                    if 0 <= idx < len(options):
                        chosen_idx = idx
                    elif 1 <= idx <= len(options):
                        chosen_idx = idx - 1
                    else:
                        chosen_idx = None
                    if chosen_idx is not None:
                        answer_text = options[chosen_idx]
                        answer_label = chr(ord("A") + chosen_idx)
                except Exception:
                    text_value = str(possible_answer).strip()
                    if len(text_value) == 1 and text_value.isalpha():
                        letter = text_value.upper()
                        letter_idx = ord(letter) - ord("A")
                        if 0 <= letter_idx < len(options):
                            answer_label = letter
                            answer_text = options[letter_idx]
                    else:
                        text_value = text_value.strip()
                        for opt_idx, option in enumerate(options):
                            option_norm = option.strip().lower()
                            if text_value.lower() == option_norm or text_value in option:
                                answer_text = option
                                answer_label = chr(ord("A") + opt_idx)
                                break
                        if answer_text is None:
                            answer_text = text_value
                            answer_label = None

            processed.append(
                {
                    "id": idx,
                    "question": question,
                    "options": options,
                    "answer_label": answer_label,
                    "answer_text": answer_text,
                    "raw": dict(example),
                }
            )

        out_path = os.path.join(out_dir, f"medmcqa_processed.{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as fh:
            for row in processed:
                fh.write(json.dumps(row) + "\n")
        print(f"Saved processed split '{split}' to {out_path} (n={len(processed)})")
        saved[split] = out_path

    return saved


def format_medmcqa_for_chat(
    in_jsonl: str,
    out_jsonl: str,
    tokenizer,
    markers: Dict[str, str],
    subset_limit: Optional[int] = None,
):
    """Convert processed MedMCQA JSONL rows into chat-formatted JSONL."""
    makedirs(os.path.dirname(out_jsonl) or ".")
    system_prompt = markers["system_prompt"]
    reasoning_start = markers["reasoning_start"]
    reasoning_end = markers["reasoning_end"]
    solution_start = markers["solution_start"]
    solution_end = markers["solution_end"]

    formatted: List[Dict[str, object]] = []
    with open(in_jsonl, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if subset_limit and idx >= subset_limit:
                break
            example = json.loads(line)
            question = example["question"]
            options = example.get("options") or []

            if options:
                labeled = [f"{chr(ord('A') + opt_idx)}. {opt}" for opt_idx, opt in enumerate(options)]
                choices_text = "\n".join(labeled)
                prompt_text = (
                    f"{question}\n\nOptions:\n{choices_text}\n\n"
                    f"Please answer with the OPTION LABEL only (for example 'A' or 'B').\n"
                    f"Place the final option label between {solution_start} and {solution_end}."
                )
            else:
                prompt_text = (
                    f"{question}\n\nPlease place the final answer between {solution_start} and {solution_end}."
                )

            messages: List[Dict[str, str]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text},
            ]

            answer_label = example.get("answer_label")
            answer_text = example.get("answer_text")

            gold_explanation: Optional[str] = None
            raw = example.get("raw", {}) if isinstance(example.get("raw"), dict) else {}
            for key in ("exp", "explanation", "explain"):
                if isinstance(raw, dict) and raw.get(key):
                    gold_explanation = str(raw.get(key)).strip()
                    break

            if answer_label is not None:
                sol_text = answer_label
            elif answer_text is not None and options:
                matched_label = None
                for opt_idx, opt in enumerate(options):
                    if answer_text.strip().lower() == opt.strip().lower() or answer_text.strip() in opt:
                        matched_label = chr(ord("A") + opt_idx)
                        break
                sol_text = matched_label or answer_text
            else:
                sol_text = ""

            reasoning_content = gold_explanation or ""
            assistant_payload = (
                f"{reasoning_start}{reasoning_content}{reasoning_end}"
                f"{solution_start}{sol_text}{solution_end}"
            )

            if sol_text or reasoning_content:
                messages.append({"role": "assistant", "content": assistant_payload})

            formatted.append({"id": example.get("id"), "messages": messages, "raw": example})

    with open(out_jsonl, "w", encoding="utf-8") as fh:
        for row in formatted:
            fh.write(json.dumps(row) + "\n")
    print(f"Wrote formatted dataset to {out_jsonl} (n={len(formatted)})")
    return formatted


__all__ = ["download_medmcqa", "format_medmcqa_for_chat"]
