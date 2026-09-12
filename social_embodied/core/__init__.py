"""Core benchmark interfaces and datatypes."""

from social_embodied.core.agent import Agent
from social_embodied.core.scorer import Scorer
from social_embodied.core.task import Task
from social_embodied.core.types import (
    Action,
    AgentId,
    Event,
    Observation,
    ObjectRef,
    StepResult,
    TaskSpec,
)

__all__ = [
    "Action",
    "Agent",
    "AgentId",
    "Event",
    "ObjectRef",
    "Observation",
    "Scorer",
    "StepResult",
    "Task",
    "TaskSpec",
]

