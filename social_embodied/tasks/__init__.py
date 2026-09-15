"""Benchmark task definitions."""

from social_embodied.tasks.ambiguous_reference import (
    AmbiguousReferenceScorer,
    build_task_0_spec,
    build_task_1_1_spec,
)
from social_embodied.tasks.io import load_task_spec, task_spec_from_dict

__all__ = [
    "AmbiguousReferenceScorer",
    "build_task_0_spec",
    "build_task_1_1_spec",
    "load_task_spec",
    "task_spec_from_dict",
]
