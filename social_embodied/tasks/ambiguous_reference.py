"""Task 0: ambiguous reference resolution from social cues."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_embodied.core.types import Action, Observation, TaskSpec
from social_embodied.tasks.io import load_task_spec


DEFAULT_TASK_0_INSTANCE = (
    Path(__file__).resolve().parents[2] / "benchmark" / "tasks" / "task_0" / "instances" / "ambiguous_reference_0000.json"
)
DEFAULT_TASK_1_1_INSTANCE = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "tasks"
    / "task_1"
    / "instances"
    / "reference_no_history_s1_0000.json"
)


def build_task_0_spec() -> TaskSpec:
    """Return the default JSON-backed Task 0 instance."""

    return load_task_spec(DEFAULT_TASK_0_INSTANCE)


def build_task_1_1_spec() -> TaskSpec:
    """Return Task 1.1: no-history reference ambiguity with E watching TV."""

    return load_task_spec(DEFAULT_TASK_1_1_INSTANCE)


@dataclass
class AmbiguousReferenceScorer:
    """Score whether T selected the object indicated by E's social cue."""

    target_object_id: int | None = None
    selected_object_id: int | None = None
    asked_clarification: bool = False
    action_count: int = 0
    errors: list[str] | None = None
    acceptable_actions: set[str] | None = None
    require_final_hold: bool = False

    def reset(self, task_spec: TaskSpec) -> None:
        self.target_object_id = task_spec.success.get("target_object_id")
        self.selected_object_id = None
        self.asked_clarification = False
        self.action_count = 0
        self.errors = []
        self.acceptable_actions = set(task_spec.success.get("acceptable_actions", ["pick_up", "give_to"]))
        self.require_final_hold = bool(task_spec.success.get("require_final_hold", False))

    def update(self, observation: Observation, action: Action) -> None:
        self.action_count += 1
        if action.action_type == "ask":
            self.asked_clarification = True
        acceptable_actions = self.acceptable_actions or {"pick_up", "give_to"}
        if action.action_type in acceptable_actions and action.target_object_id is not None:
            if self.selected_object_id is None:
                self.selected_object_id = action.target_object_id

        error = observation.metadata.get("error")
        if error and self.errors is not None:
            self.errors.append(str(error))

    def final_score(self, final_observation: Observation | None = None) -> dict[str, Any]:
        final_hold = _target_held_by_target_agent(final_observation, self.target_object_id)
        target_success = self.selected_object_id == self.target_object_id
        if self.require_final_hold and final_hold is not None:
            target_success = target_success and final_hold
        result = {
            "success": bool(target_success),
            "target_object_id": self.target_object_id,
            "selected_object_id": self.selected_object_id,
            "asked_clarification": self.asked_clarification,
            "action_count": self.action_count,
            "acceptable_actions": sorted(self.acceptable_actions or []),
            "require_final_hold": self.require_final_hold,
            "errors": self.errors or [],
        }
        if final_observation is not None:
            result.update(
                {
                    "final_target_held_by_T": final_hold,
                    "num_final_fpv_images": len(final_observation.fpv_images),
                    "final_debug_image_channels": {
                        channel: len(images) for channel, images in final_observation.debug_images.items()
                    },
                }
            )
        return result


def _target_held_by_target_agent(observation: Observation | None, target_id: int | None) -> bool | None:
    if observation is None or observation.scene_graph is None or target_id is None:
        return None

    character_ids = [
        int(node["id"])
        for node in observation.scene_graph.get("nodes", [])
        if node.get("class_name") == "character" and node.get("id") is not None
    ]
    if not character_ids:
        return None
    target_agent_id = min(character_ids)
    return any(
        edge.get("from_id") == target_agent_id
        and edge.get("to_id") == target_id
        and edge.get("relation_type") in {"HOLDS_RH", "HOLDS_LH"}
        for edge in observation.scene_graph.get("edges", [])
    )
