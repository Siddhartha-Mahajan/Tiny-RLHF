#!/usr/bin/env python3
"""Create a tiny prompt/completion dataset for free-form SFT."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("data/freeform")
DATA_DIR.mkdir(parents=True, exist_ok=True)

train = []
validation = []
for idx in range(30):
    prompt = f"Explain concept #{idx} in one sentence."
    completion = f"Concept #{idx} can be understood as a simple example in this synthetic dataset."
    record = {"prompt": prompt, "completion": completion}
    if idx < 24:
        train.append(record)
    else:
        validation.append(record)

for name, items in {"train": train, "validation": validation}.items():
    with open(DATA_DIR / f"{name}.jsonl", "w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item) + "\n")

print(f"Wrote free-form dataset to {DATA_DIR}")
