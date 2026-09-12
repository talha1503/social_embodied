"""Deterministic baseline agents."""

from __future__ import annotations

from collections.abc import Iterable

from social_embodied.core.types import Action, Observation, TaskSpec


class ScriptedAgent:
    """A deterministic agent that replays a provided action list."""

    name = "scripted"

    def __init__(self, actions: Iterable[Action]) -> None:
        self._initial_actions = list(actions)
        self._actions: list[Action] = []

    def reset(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self._actions = list(self._initial_actions)

    def act(self, observation: Observation) -> Action:
        if not self._actions:
            return Action(action_type="wait")
        return self._actions.pop(0)

