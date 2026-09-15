"""Simple random baseline agent."""

from __future__ import annotations

import random

from social_embodied.core.types import Action, Observation, TaskSpec


class RandomAgent:
    """Randomly chooses from a small symbolic action set."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def reset(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec

    def act(self, observation: Observation) -> Action:
        candidates = [
            Action(action_type="wait"),
            Action(action_type="ask", utterance="Which one do you mean?"),
        ]
        for obj in observation.visible_objects[:10]:
            if obj.object_id is None:
                continue
            candidates.append(
                Action(
                    action_type="pick_up",
                    target_object_id=obj.object_id,
                    params={"object_class": obj.class_name},
                )
            )
        return self._rng.choice(candidates)