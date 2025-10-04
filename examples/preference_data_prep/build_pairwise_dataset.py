#!/usr/bin/env python3
"""Create a synthetic preference dataset for DPO experiments."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("data/preferences")
DATA_DIR.mkdir(parents=True, exist_ok=True)

records = []
for idx in range(10):
    prompt = f"Synthetic prompt {idx}: describe the benefits of hydration."
    chosen = "Hydration supports cellular function, temperature regulation, and cognitive clarity."
    rejected = "Hydration might be neat sometimes, I guess."
    records.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as fh:
    for item in records:
        fh.write(json.dumps(item) + "\n")

print(f"Created {len(records)} preference pairs in {DATA_DIR}")
