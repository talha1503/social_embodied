"""Build benchmark observations from simulator state."""

from __future__ import annotations

from typing import Any

from social_embodied.core.types import Action, Event, ObjectRef, Observation, TaskSpec


def visible_objects_from_graph(scene_graph: dict[str, Any] | None) -> list[ObjectRef]:
    """Return coarse object refs from a VirtualHome environment graph.

    This is not true visibility yet. Actual FPV visibility should later use
    segmentation/camera projection. For now it gives oracle/debug agents a
    consistent structured view when a graph is available.
    """

    if not scene_graph:
        return []

    objects: list[ObjectRef] = []
    for node in scene_graph.get("nodes", []):
        object_id = node.get("id")
        class_name = node.get("class_name")
        if object_id is None or class_name is None:
            continue
        objects.append(
            ObjectRef(
                object_id=object_id,
                class_name=class_name,
                display_name=node.get("prefab_name") or class_name,
                room=node.get("room"),
                metadata={k: v for k, v in node.items() if k not in {"id", "class_name", "prefab_name", "room"}},
            )
        )
    return objects


def build_observation(
    *,
    step_id: int,
    task_spec: TaskSpec,
    events: list[Event],
    last_action: Action | None = None,
    fpv_images: list[Any] | None = None,
    debug_images: dict[str, list[Any]] | None = None,
    scene_graph: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Observation:
    """Create the public observation object for T."""

    return Observation(
        step_id=step_id,
        fpv_images=fpv_images or [],
        debug_images=debug_images or {},
        events=events,
        last_action=last_action,
        visible_objects=visible_objects_from_graph(scene_graph),
        scene_graph=scene_graph,
        task_prompt=task_spec.prompt,
        metadata=metadata or {},
    )
