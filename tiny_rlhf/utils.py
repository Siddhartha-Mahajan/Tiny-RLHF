"""Assorted helpers used across the package."""
from __future__ import annotations

import contextlib
import logging
import random
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import torch

from .config import RunConfig


logger = logging.getLogger("tiny_rlhf")


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class WandbSession:
    """Lightweight context manager for optional Weights & Biases logging."""

    enabled: bool
    project: Optional[str]
    run_name: Optional[str]

    def __enter__(self):
        if not self.enabled:
            logger.info("W&B logging disabled.")
            return None
        if not self.project:
            raise ValueError("run.use_wandb=True but no project provided.")
        try:
            import wandb
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install wandb to enable Weights & Biases integration." ) from exc

        wandb.init(project=self.project, name=self.run_name)
        logger.info("W&B run initialised: project=%s name=%s", self.project, self.run_name)
        return wandb

    def __exit__(self, exc_type, exc, tb) -> bool:
        if not self.enabled:
            return False
        try:
            import wandb

            wandb.finish()
        except Exception:  # pragma: no cover - best effort shutdown
            logger.warning("Failed to finish W&B session cleanly.")
        return False


def maybe_wandb(run_cfg: RunConfig) -> contextlib.AbstractContextManager:
    return WandbSession(run_cfg.use_wandb, run_cfg.wandb_project, run_cfg.wandb_run_name)
