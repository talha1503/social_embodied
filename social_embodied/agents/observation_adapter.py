"""Agent-facing observation and action schemas.

This module keeps benchmark observations separate from model inputs. The raw
``Observation`` object contains simulator/debug fields; model-backed agents
should receive a controlled payload and choose from explicit action choices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from social_embodied.core.types import Action, Event, Observation, TaskSpec


@dataclass(frozen=True)
class CandidateObject:
    """Object candidate exposed to an agent."""

    label: str
    object_id: int
    class_name: str


@dataclass(frozen=True)
class ActionChoice:
    """One valid high-level action an agent may choose."""

    label: str
    action_type: str
    target_label: str | None = None
    target_object_id: int | None = None
    target_class: str | None = None
    utterance: str | None = None

    def to_action(self) -> Action:
        params: dict[str, Any] = {}
        if self.target_class is not None:
            params["object_class"] = self.target_class
        return Action(
            action_type=self.action_type,
            target_object_id=self.target_object_id,
            utterance=self.utterance,
            params=params,
        )


@dataclass(frozen=True)
class AgentInput:
    """Structured payload intended for an LLM/VLM policy."""

    role_text: str
    task_prompt: str | None
    conversation_history: list[dict[str, Any]]
    social_events: list[dict[str, Any]]
    candidates: list[CandidateObject]
    available_actions: list[ActionChoice]
    memory: list[dict[str, Any]]
    observation_summary: dict[str, Any]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "role_text": self.role_text,
            "task_prompt": self.task_prompt,
            "conversation_history": self.conversation_history,
            "social_events": self.social_events,
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "available_actions": [choice.__dict__ for choice in self.available_actions],
            "memory": self.memory,
            "observation_summary": self.observation_summary,
            "output_schema": {
                "action": "one of the available action labels",
                "target": "candidate label when the chosen action needs a target, otherwise null",
                "utterance": "short text only for ask/say actions",
            },
        }


class Task1ObservationAdapter:
    """Build controlled agent inputs for Task 1 reference ambiguity."""

    role_text = "You are target agent T. Help environment agent E complete their request."

    def build(self, task_spec: TaskSpec, observation: Observation) -> AgentInput:
        candidates = _candidate_objects(task_spec)
        return AgentInput(
            role_text=self.role_text,
            task_prompt=observation.task_prompt,
            conversation_history=_conversation_history(observation.events),
            social_events=_social_events(observation.events, task_spec),
            candidates=candidates,
            available_actions=_task1_action_choices(candidates),
            memory=_memory_from_task_spec(task_spec),
            observation_summary=_observation_summary(observation),
        )


def action_from_model_output(model_output: dict[str, Any], agent_input: AgentInput) -> Action:
    """Parse a constrained model output into a benchmark Action."""

    action_label = str(model_output.get("action", "")).strip()
    target_label = model_output.get("target")
    if target_label is not None:
        target_label = str(target_label).strip()

    choices = {choice.label: choice for choice in agent_input.available_actions}
    if action_label not in choices:
        valid = ", ".join(sorted(choices))
        raise ValueError(f"Unknown action label {action_label!r}; valid labels: {valid}")

    choice = choices[action_label]
    if choice.target_label is not None and target_label not in {None, "", choice.target_label}:
        raise ValueError(
            f"Action {action_label!r} targets {choice.target_label!r}, "
            f"but model returned target {target_label!r}"
        )

    if choice.action_type in {"ask", "say"}:
        utterance = model_output.get("utterance") or choice.utterance
        return Action(action_type=choice.action_type, utterance=str(utterance))

    return choice.to_action()


def _candidate_objects(task_spec: TaskSpec) -> list[CandidateObject]:
    candidates = []
    for idx, item in enumerate(task_spec.objects.get("candidates", [])):
        object_id = item.get("id")
        class_name = item.get("class_name")
        if object_id is None or class_name is None:
            continue
        suffix = chr(ord("A") + idx)
        candidates.append(
            CandidateObject(
                label=f"{class_name}_{suffix}",
                object_id=int(object_id),
                class_name=str(class_name),
            )
        )
    return candidates


def _task1_action_choices(candidates: list[CandidateObject]) -> list[ActionChoice]:
    choices = [
        ActionChoice(label="ask_clarification", action_type="ask", utterance="Which mug do you mean?"),
        ActionChoice(label="wait", action_type="wait"),
    ]
    for candidate in candidates:
        choices.append(
            ActionChoice(
                label=f"move_to_{candidate.label}",
                action_type="move_to",
                target_label=candidate.label,
                target_object_id=candidate.object_id,
                target_class=candidate.class_name,
            )
        )
        choices.append(
            ActionChoice(
                label=f"pick_up_{candidate.label}",
                action_type="pick_up",
                target_label=candidate.label,
                target_object_id=candidate.object_id,
                target_class=candidate.class_name,
            )
        )
    return choices


def _conversation_history(events: list[Event]) -> list[dict[str, Any]]:
    return [
        {
            "speaker": event.actor,
            "text": event.content,
            "time": event.start_time,
        }
        for event in events
        if event.event_type == "speech" and event.content
    ]


def _social_events(events: list[Event], task_spec: TaskSpec) -> list[dict[str, Any]]:
    result = []
    for event in events:
        if event.event_type == "speech":
            continue
        record = {
            "type": event.event_type,
            "actor": event.actor,
            "target_object_id": event.target_object_id,
            "target_label": _label_for_object_id(task_spec, event.target_object_id),
            "time": event.start_time,
            "duration": event.duration,
        }
        result.append(record)
    return result


def _label_for_object_id(task_spec: TaskSpec, object_id: int | None) -> str | None:
    if object_id is None:
        return None
    for candidate in _candidate_objects(task_spec):
        if candidate.object_id == int(object_id):
            return candidate.label
    target = task_spec.objects.get("target", {})
    if target.get("id") is not None and int(target["id"]) == int(object_id):
        return str(target.get("label") or f"{target.get('class_name', 'object')}_{object_id}")
    distractors = task_spec.objects.get("distractors", {})
    if isinstance(distractors, dict):
        for name, value in distractors.items():
            if isinstance(value, dict) and value.get("id") is not None and int(value["id"]) == int(object_id):
                return str(value.get("label") or name)
    return None


def _memory_from_task_spec(task_spec: TaskSpec) -> list[dict[str, Any]]:
    memory = task_spec.metadata.get("memory", [])
    return memory if isinstance(memory, list) else []


def _observation_summary(observation: Observation) -> dict[str, Any]:
    visibility = observation.metadata.get("visibility", {})
    return {
        "step_id": observation.step_id,
        "has_fpv_image": bool(observation.fpv_images),
        "num_fpv_images": len(observation.fpv_images),
        "has_scene_graph": observation.scene_graph is not None,
        "num_visible_objects": len(observation.visible_objects),
        "last_action": None if observation.last_action is None else observation.last_action.action_type,
        "visibility": visibility,
    }
