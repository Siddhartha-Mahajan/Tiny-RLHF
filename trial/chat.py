"""Chat templating helpers for the MedMCQA pipeline."""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from transformers import AutoTokenizer  # type: ignore


def build_chat_template(tokenizer: AutoTokenizer) -> Dict[str, str]:
    """Attach the custom chat template to *tokenizer* and return tag markers."""
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


def _fallback_system_prompt() -> str:
    return (
        "You are given a problem.\n"
        "Think about the problem and provide your working out.\n"
        "Place it between <start_working_out> and <end_working_out>.\n"
        "Then, provide your solution between <SOLUTION></SOLUTION>"
    )


def apply_chat_template_to_messages(
    tokenizer: AutoTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
    tokenize: bool = False,
):
    """Apply the configured chat template with a robust manual fallback."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=tokenize,
            )
        except Exception:
            pass

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        system_prompt = _fallback_system_prompt()
    else:
        try:
            if "{{ '" in chat_template and "' + eos_token" in chat_template:
                system_prompt = chat_template.split("{{ '")[1].split("' +")[0]
            else:
                system_prompt = _fallback_system_prompt()
        except Exception:
            system_prompt = _fallback_system_prompt()

    pieces: List[str] = []
    if messages and messages[0].get("role") == "system":
        pieces.append(messages[0]["content"])
        loop_messages = messages[1:]
    else:
        pieces.append(system_prompt)
        loop_messages = messages

    for msg in loop_messages:
        pieces.append(msg.get("content", ""))

    if add_generation_prompt:
        pieces.append("<start_working_out>")

    raw = "\n".join(pieces)

    if tokenize:
        try:
            enc = tokenizer(raw, return_tensors="pt")
            if "input_ids" in enc:
                return enc["input_ids"][0].tolist()
            return enc
        except Exception:
            return raw

    return raw


def extract_solution_from_text(text: Optional[str], sol_start: str, sol_end: str) -> Optional[str]:
    if text is None:
        return None
    start_idx = text.find(sol_start)
    if start_idx == -1:
        return None
    end_idx = text.find(sol_end, start_idx + len(sol_start))
    if end_idx == -1:
        return text[start_idx + len(sol_start):].strip()
    return text[start_idx + len(sol_start):end_idx].strip()


def extract_pred_with_fallback(
    out: Optional[str],
    sol_start: str,
    sol_end: str,
    options: Optional[List[str]] = None,
) -> Optional[str]:
    if not out:
        return None

    sol = extract_solution_from_text(out, sol_start, sol_end)
    if sol:
        return sol.strip()

    match = re.search(r"(answer|ans|final)\s*[:\-]?\s*([A-D])\b", out, re.IGNORECASE)
    if match:
        return match.group(2).upper()

    single_letter = re.search(r"\b([A-D])\b", out, re.IGNORECASE)
    if single_letter:
        return single_letter.group(1).upper()

    if options:
        out_lower = out.lower()
        for idx, option in enumerate(options):
            if option and option.lower() in out_lower:
                return chr(ord("A") + idx)

    return None

__all__ = [
    "build_chat_template",
    "apply_chat_template_to_messages",
    "extract_solution_from_text",
    "extract_pred_with_fallback",
]
