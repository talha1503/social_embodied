"""Task registry."""

from __future__ import annotations

from pathlib import Path

from social_embodied.core.types import TaskSpec
from social_embodied.tasks.ambiguous_reference import build_task_0_spec
from social_embodied.tasks.io import load_task_spec


def get_task_spec(task_id: str) -> TaskSpec:
    """Resolve a task id to a concrete task spec."""

    if task_id in {"task_0", "ambiguous_reference_0000"}:
        return build_task_0_spec()
    if task_id.endswith(".json"):
        return load_task_spec(Path(task_id))
    raise KeyError(f"Unknown task id: {task_id}")
