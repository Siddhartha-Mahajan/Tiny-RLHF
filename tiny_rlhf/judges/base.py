"""Judge backend contracts."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class JudgeBackend(ABC):
    @abstractmethod
    def score(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Produce scalar scores."""
