"""Task interface."""

from __future__ import annotations

from typing import Protocol

from social_embodied.core.scorer import Scorer
from social_embodied.core.types import TaskSpec


class Task(Protocol):
    """A benchmark task family or concrete task instance."""

    spec: TaskSpec
    scorer: Scorer

    def setup(self, env: object) -> None:
        """Configure the environment for this task."""

