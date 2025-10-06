"""Small helpers for inspecting experiment results."""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from .utils import json_load


def load_eval_summary(path: str = "experiments_v2/eval_summary.json") -> Optional[Dict[str, float]]:
    """Return the evaluation summary if it exists, otherwise ``None``."""
    if not os.path.exists(path):
        return None
    return json_load(path)


def print_eval_summary(path: str = "experiments_v2/eval_summary.json") -> None:
    """Pretty-print the stored evaluation summary, if present."""
    summary = load_eval_summary(path)
    if summary is None:
        print("No evaluation summary found at", path)
        return

    print("=== Stored Evaluation Summary ===")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")


__all__ = ["load_eval_summary", "print_eval_summary"]
