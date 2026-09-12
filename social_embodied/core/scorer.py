"""Scoring interface."""

from __future__ import annotations

from typing import Any, Protocol

from social_embodied.core.types import Action, Observation, TaskSpec


class Scorer(Protocol):
    """Task-specific evaluator."""

    def reset(self, task_spec: TaskSpec) -> None:
        """Reset per-episode scoring state."""

    def update(self, observation: Observation, action: Action) -> None:
        """Consume one observation/action pair."""

    def final_score(self, final_observation: Observation | None = None) -> dict[str, Any]:
        """Return final metrics."""

