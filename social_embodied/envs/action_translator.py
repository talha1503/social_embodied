"""Translate benchmark actions into VirtualHome script fragments."""

from __future__ import annotations

from social_embodied.core.types import Action


class ActionTranslationError(ValueError):
    """Raised when a symbolic action cannot be converted for VirtualHome."""


def to_virtualhome_script(action: Action) -> list[str]:
    """Convert one symbolic T action into VirtualHome script lines.

    The benchmark action vocabulary is intentionally higher-level than
    VirtualHome's script syntax. This translator is the only place that should
    know about VirtualHome command strings.
    """

    char = "<char0>" if action.actor == "T" else "<char1>"

    if action.action_type == "wait":
        return []

    if action.action_type == "say" or action.action_type == "ask":
        # Speech is tracked as a benchmark event for now. Unity speech/audio can
        # be added later without changing the agent API.
        return []

    if action.target_object_id is None and action.action_type not in {"move_to_room"}:
        raise ActionTranslationError(f"{action.action_type} requires target_object_id")

    object_name = action.params.get("object_class")
    object_id = action.target_object_id

    if action.action_type == "look_at":
        return [f"{char} [lookat] <{object_name}> ({object_id})"]
    if action.action_type == "move_to":
        return [f"{char} [walk] <{object_name}> ({object_id})"]
    if action.action_type == "pick_up":
        return [f"{char} [grab] <{object_name}> ({object_id})"]
    if action.action_type == "open":
        return [f"{char} [open] <{object_name}> ({object_id})"]
    if action.action_type == "close":
        return [f"{char} [close] <{object_name}> ({object_id})"]
    if action.action_type == "give_to":
        # VirtualHome does not have a generic "give" primitive in the current
        # backend. For now, move near E; task-specific code can add handoff.
        return [f"{char} [walk] <{object_name}> ({object_id})"]
    if action.action_type == "put_on":
        surface_id = action.params.get("surface_id")
        surface_class = action.params.get("surface_class")
        if surface_id is None or surface_class is None:
            raise ActionTranslationError("put_on requires surface_id and surface_class")
        return [f"{char} [putback] <{object_name}> ({object_id}) <{surface_class}> ({surface_id})"]

    if action.action_type == "move_to_room":
        if action.target_room is None:
            raise ActionTranslationError("move_to_room requires target_room")
        room_id = action.params.get("room_id", 1)
        return [f"{char} [walk] <{action.target_room}> ({room_id})"]

    raise ActionTranslationError(f"Unsupported action type: {action.action_type}")

