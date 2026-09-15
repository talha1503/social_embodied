"""Constrained JSON policy agent scaffolding.

The agent in this module is model-provider agnostic. A client receives a
structured ``AgentInput`` and returns a small JSON action object. This lets us
plug in GPT/Gemini/Claude/etc. later without changing the environment loop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Protocol

from social_embodied.agents.observation_adapter import (
    AgentInput,
    Task1ObservationAdapter,
    action_from_model_output,
)
from social_embodied.core.types import Action, Observation, TaskSpec


class ModelPolicyClient(Protocol):
    """Client interface for a model-backed policy."""

    def choose_action(self, agent_input: AgentInput) -> dict[str, Any]:
        """Return a constrained JSON action object."""


class Task1HeuristicPolicyClient:
    """Local deterministic client for exercising the model-agent interface.

    This is not a benchmark baseline. It exists so we can test the same
    observation/action path that a future LLM/VLM client will use.
    """

    def __init__(self, *, target_relation: str = "farthest_from_environment_agent") -> None:
        self.target_relation = target_relation
        self._phase = "move"
        self._target_label: str | None = None

    def reset(self) -> None:
        self._phase = "move"
        self._target_label = None

    def choose_action(self, agent_input: AgentInput) -> dict[str, Any]:
        if not agent_input.candidates:
            return {
                "action": "ask_clarification",
                "target": None,
                "utterance": "Which mug do you mean?",
            }

        if self._target_label is None:
            self._target_label = self._select_target(agent_input)

        if self._phase == "move":
            self._phase = "pick_up"
            return {
                "action": f"move_to_{self._target_label}",
                "target": self._target_label,
                "utterance": None,
            }

        return {
            "action": f"pick_up_{self._target_label}",
            "target": self._target_label,
            "utterance": None,
        }

    def _select_target(self, agent_input: AgentInput) -> str:
        # Task 1.1 uses candidate order after materialization where the target
        # relation determines the target object in the TaskSpec. The local
        # heuristic mirrors the current target-relation setup for smoke tests.
        if self.target_relation in {"closest_to_environment_agent", "closest_to_e"}:
            return agent_input.candidates[0].label
        return agent_input.candidates[-1].label


class ConstrainedJsonAgent:
    """Agent that converts observations to model input and parses JSON actions."""

    name = "constrained_json_agent"

    def __init__(
        self,
        client: ModelPolicyClient | None = None,
        *,
        adapter: Task1ObservationAdapter | None = None,
        name: str | None = None,
    ) -> None:
        self.client = client or Task1HeuristicPolicyClient()
        self.adapter = adapter or Task1ObservationAdapter()
        if name is not None:
            self.name = name

    def reset(self, task_spec: TaskSpec) -> None:
        self.task_spec = task_spec
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()

    def act(self, observation: Observation) -> Action:
        agent_input = self.adapter.build(self.task_spec, observation)
        model_output = self.client.choose_action(agent_input)
        action = action_from_model_output(model_output, agent_input)
        params = dict(action.params)
        params["_model_output"] = model_output
        params["_candidate_labels"] = [
            {"label": candidate.label, "object_id": candidate.object_id, "class_name": candidate.class_name}
            for candidate in agent_input.candidates
        ]
        return replace(action, params=params)
