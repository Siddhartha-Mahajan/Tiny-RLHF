"""Small utility helpers shared across modules."""
from __future__ import annotations

import json
import pathlib
from typing import Any


def makedirs(*paths: str) -> None:
    """Create one or more directories if they do not already exist."""
    for path in paths:
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    """Write *obj* as JSON to *path* using UTF-8 and nice indentation."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def json_load(path: str) -> Any:
    """Read JSON from *path* and return the decoded Python object."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

__all__ = ["makedirs", "json_dump", "json_load"]
