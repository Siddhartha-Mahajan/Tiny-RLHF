"""Logging utilities."""
from __future__ import annotations

import logging

DEFAULT_LOG_LEVEL = logging.INFO


def configure_logging(level: int = DEFAULT_LOG_LEVEL) -> None:
    logging.basicConfig(level=level, format="[%(levelname)s] %(name)s: %(message)s")
