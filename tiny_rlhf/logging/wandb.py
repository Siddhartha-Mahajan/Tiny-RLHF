"""Optional Weights & Biases integration."""
from __future__ import annotations

import contextlib
from typing import Dict, Iterator, Optional

class WandbSession:
    def __init__(self, project: str, run_name: Optional[str] = None):
        self.project = project
        self.run_name = run_name
        self._run = None

    def __enter__(self):
        try:
            import wandb  # type: ignore
        except Exception:  # pragma: no cover - optional dependency
            return self
        self._run = wandb.init(project=self.project, name=self.run_name, reinit=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log(self, metrics: Dict[str, float]) -> None:
        if self._run is not None:
            self._run.log(metrics)
