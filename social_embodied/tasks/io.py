"""Load task specifications from benchmark instance files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from social_embodied.core.types import TaskSpec


def load_task_spec(path: Path) -> TaskSpec:
    """Load a `TaskSpec` from a JSON task instance file."""

    data = json.loads(path.read_text(encoding="utf-8"))
    return task_spec_from_dict(data, source_path=path)


def task_spec_from_dict(data: dict[str, Any], *, source_path: Path | None = None) -> TaskSpec:
    """Normalize an instance dict into the current `TaskSpec` dataclass."""

    required = ["task_id", "task_family", "scene_id", "prompt"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Task instance missing required fields: {missing}")

    e_behavior = _normalize_e_behavior(data.get("e_behavior", {}))
    metadata = dict(data.get("metadata", {}))
    for key in ("agents", "object_selection", "layout"):
        if key in data:
            metadata[key] = data[key]
    if source_path is not None:
        metadata["source_path"] = str(source_path)

    return TaskSpec(
        task_id=str(data["task_id"]),
        task_family=str(data["task_family"]),
        scene_id=int(data["scene_id"]),
        prompt=str(data["prompt"]),
        objects=dict(data.get("objects", {})),
        e_behavior=e_behavior,
        success=dict(data.get("success", {})),
        metadata=metadata,
    )


def _normalize_e_behavior(raw: dict[str, Any]) -> dict[str, Any]:
    e_behavior = dict(raw)
    gaze = raw.get("gaze")
    if isinstance(gaze, dict):
        e_behavior["gaze_target_id"] = gaze.get("target")
        e_behavior["gaze_duration"] = gaze.get("duration")
        e_behavior["gaze_visible"] = bool(gaze.get("visible", False))
    gesture = raw.get("gesture")
    if isinstance(gesture, dict):
        e_behavior["gesture"] = gesture.get("type")
        e_behavior["gesture_target_id"] = gesture.get("target")
        e_behavior["gesture_duration"] = gesture.get("duration")
        e_behavior["gesture_intensity"] = gesture.get("intensity")
        e_behavior["gesture_visible"] = bool(gesture.get("visible", False))
    return e_behavior
