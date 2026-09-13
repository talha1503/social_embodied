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


class SocialCueOracleAgent:
    """Oracle baseline that follows E's explicit gaze/gesture target metadata."""

    name = "social_cue_oracle"

    def reset(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        self._target_id: int | None = None
        target = task_spec.objects.get("target", {})
        self._object_class: str = str(target.get("class_name", "object"))
        self._phase = "move"

    def act(self, observation: Observation) -> Action:
        if self._target_id is None:
            for event in observation.events:
                if event.actor == "E" and event.event_type in {"gesture", "gaze"} and event.target_object_id is not None:
                    self._target_id = event.target_object_id
                    break

        if self._target_id is None:
            return Action(action_type="ask", utterance="Which one do you mean?")

        for obj in observation.visible_objects:
            if obj.object_id == self._target_id:
                self._object_class = obj.class_name
                break

        if self._phase == "move":
            self._phase = "pick_up"
            return Action(
                action_type="move_to",
                target_object_id=self._target_id,
                params={"object_class": self._object_class},
            )

        self._phase = "done"
        return Action(
            action_type="pick_up",
            target_object_id=self._target_id,
            params={"object_class": self._object_class},
        )
