"""Checkpoint utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_adapter_state(model: Any, output_dir: str) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    torch.save(state, output / "adapter.pt")
