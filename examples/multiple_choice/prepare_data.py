#!/usr/bin/env python3
"""Generate a tiny synthetic multiple-choice dataset for smoke testing."""
from __future__ import annotations

import json
from pathlib import Path

DATA_DIR = Path("data/multiple_choice")
DATA_DIR.mkdir(parents=True, exist_ok=True)

train = []
validation = []
options = ["A. Heart", "B. Lung", "C. Kidney", "D. Brain"]
for idx in range(20):
    example = {
        "id": idx,
        "question": f"Which organ is #{idx} in this synthetic example?",
        "options": options,
        "answer": "A",
    }
    if idx < 16:
        train.append(example)
    else:
        validation.append(example)

with open(DATA_DIR / "train.jsonl", "w", encoding="utf-8") as fh:
    for item in train:
        fh.write(json.dumps(item) + "\n")

with open(DATA_DIR / "validation.jsonl", "w", encoding="utf-8") as fh:
    for item in validation:
        fh.write(json.dumps(item) + "\n")

print(f"Created {len(train)} train and {len(validation)} validation examples in {DATA_DIR}.")
