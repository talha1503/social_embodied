"""Task 0: ambiguous reference resolution from social cues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from social_embodied.core.types import Action, Observation, TaskSpec


def build_task_0_spec() -> TaskSpec:
    """Return a minimal dry-run task spec for ambiguous reference resolution.

    Real scenario setup will replace these object ids with ids read from
    VirtualHome after scene reset.
    """

    return TaskSpec(
        task_id="ambiguous_reference_0000",
        task_family="ambiguous_reference",
        scene_id=4,
        prompt="E asks T to bring the intended mug, but language alone is ambiguous.",
        objects={
            "candidates": [
                {"id": 101, "class_name": "mug", "label": "mug_left"},
                {"id": 102, "class_name": "mug", "label": "mug_right"},
            ],
            "target": {"id": 102, "class_name": "mug", "label": "mug_right"},
        },
        e_behavior={
            "speech": "Can you bring me that mug?",
            "gaze_target_id": 102,
            "gaze_duration": 1.5,
            "gesture": "point",
            "gesture_target_id": 102,
            "gesture_duration": 1.0,
            "gesture_intensity": 0.8,
        },
        success={
            "target_object_id": 102,
            "acceptable_actions": ["pick_up", "give_to", "move_to"],
        },
        metadata={"requires": ["fpv", "speech", "gaze", "gesture", "object_interaction"]},
    )


@dataclass
class AmbiguousReferenceScorer:
    """Score whether T selected the object indicated by E's social cue."""

    target_object_id: int | None = None
    selected_object_id: int | None = None
    asked_clarification: bool = False
    action_count: int = 0
    errors: list[str] | None = None

    def reset(self, task_spec: TaskSpec) -> None:
        self.target_object_id = task_spec.success.get("target_object_id")
        self.selected_object_id = None
        self.asked_clarification = False
        self.action_count = 0
        self.errors = []

    def update(self, observation: Observation, action: Action) -> None:
        self.action_count += 1
        if action.action_type == "ask":
            self.asked_clarification = True
        if action.action_type in {"pick_up", "give_to", "move_to"} and action.target_object_id is not None:
            if self.selected_object_id is None:
                self.selected_object_id = action.target_object_id

        error = observation.metadata.get("error")
        if error and self.errors is not None:
            self.errors.append(str(error))

    def final_score(self, final_observation: Observation | None = None) -> dict[str, Any]:
        target_success = self.selected_object_id == self.target_object_id
        return {
            "success": bool(target_success),
            "target_object_id": self.target_object_id,
            "selected_object_id": self.selected_object_id,
            "asked_clarification": self.asked_clarification,
            "action_count": self.action_count,
            "errors": self.errors or [],
        }

