"""Chat templating helpers."""
from __future__ import annotations

from typing import Any, Dict, List


def render_chat_messages(tokenizer: Any, messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """Render chat messages to a single string using the tokenizer template when available."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None) is not None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=add_generation_prompt,
                tokenize=False,
            )
        except Exception:
            pass

    # Manual fallback: simple concatenation
    parts: List[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("ASSISTANT:")
    return "\n\n".join(parts)
