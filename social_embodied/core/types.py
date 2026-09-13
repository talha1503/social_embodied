"""Shared datatypes for agents, environments, tasks, and scorers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

AgentId = Literal["T", "E"]


@dataclass(frozen=True)
class ObjectRef:
    """A stable reference to an object in a task instance."""

    object_id: int | None
    class_name: str
    display_name: str | None = None
    room: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    """A speech, social cue, object interaction, or environment event."""

    event_type: str
    actor: AgentId | Literal["system"]
    content: str | None = None
    target_object_id: int | None = None
    target_agent: AgentId | None = None
    start_time: float | None = None
    duration: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Everything an agent is allowed to observe at one decision point."""

    step_id: int
    timestamp: float | None = None
    fpv_images: list[Any] = field(default_factory=list)
    debug_images: dict[str, list[Any]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    last_action: "Action | None" = None
    visible_objects: list[ObjectRef] = field(default_factory=list)
    agent_states: dict[str, Any] = field(default_factory=dict)
    scene_graph: dict[str, Any] | None = None
    task_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """A symbolic benchmark action chosen by an agent."""

    action_type: str
    actor: AgentId = "T"
    target_object_id: int | None = None
    target_room: str | None = None
    utterance: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    """Result returned by an environment after applying an action."""

    observation: Observation
    done: bool
    reward: float | None = None
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """Serializable task-instance description."""

    task_id: str
    task_family: str
    scene_id: int
    prompt: str
    target_agent: AgentId = "T"
    environment_agent: AgentId = "E"
    objects: dict[str, Any] = field(default_factory=dict)
    e_behavior: dict[str, Any] = field(default_factory=dict)
    success: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
