"""Generic benchmark episode runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from social_embodied.core.agent import Agent
from social_embodied.core.scorer import Scorer
from social_embodied.core.types import Action, Observation, TaskSpec
from social_embodied.envs.virtualhome_env import SocialEmbodiedEnv


@dataclass(frozen=True)
class EpisodeResult:
    """Recorded output of one benchmark episode."""

    task_id: str
    agent_name: str
    actions: list[Action]
    observations: list[Observation]
    metrics: dict[str, Any]


def run_episode(
    *,
    env: SocialEmbodiedEnv,
    agent: Agent,
    task_spec: TaskSpec,
    scorer: Scorer,
    max_steps: int = 10,
) -> EpisodeResult:
    """Run one task episode using the common environment-agent-scorer loop."""

    observation = env.reset(task_spec)
    active_task_spec = env.task_spec or task_spec
    agent.reset(active_task_spec)
    scorer.reset(active_task_spec)
    observations = [observation]
    actions: list[Action] = []

    for _ in range(max_steps):
        action = agent.act(observation)
        actions.append(action)

        result = env.step(action)
        scorer.update(observation, action)
        observation = result.observation
        observations.append(observation)

        if result.done:
            break

        if action.action_type in {"pick_up", "give_to"}:
            break

    metrics = scorer.final_score(observation)
    return EpisodeResult(
        task_id=active_task_spec.task_id,
        agent_name=agent.name,
        actions=actions,
        observations=observations,
        metrics=metrics,
    )
