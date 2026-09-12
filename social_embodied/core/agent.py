"""Agent interface."""

from __future__ import annotations

from typing import Protocol

from social_embodied.core.types import Action, Observation, TaskSpec


class Agent(Protocol):
    """Common interface for scripted, LLM/VLM, oracle, and learned agents."""

    name: str

    def reset(self, task_spec: TaskSpec) -> None:
        """Reset any per-episode state before a task starts."""

    def act(self, observation: Observation) -> Action:
        """Choose the next symbolic action from the current observation."""

