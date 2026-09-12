"""VirtualHome-backed social embodied environment loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from social_embodied.core.types import Action, Event, Observation, StepResult, TaskSpec
from social_embodied.envs.action_translator import ActionTranslationError, to_virtualhome_script
from social_embodied.envs.observation_builder import build_observation


@dataclass(frozen=True)
class VirtualHomeConfig:
    """Runtime configuration for the VirtualHome adapter."""

    repo_root: Path = Path("virtualhome")
    port: str = "8080"
    unity_executable: Path | None = None
    connect: bool = False
    use_scene_graph: bool = True
    camera_mode: str = "FIRST_PERSON"
    image_width: int = 320
    image_height: int = 180


class SocialEmbodiedEnv:
    """Environment loop for social embodied benchmark tasks.

    When ``connect=False`` this runs in dry-run mode and records translated
    actions without requiring Unity. That makes task/agent development testable
    before simulator-specific details are complete.
    """

    def __init__(self, config: VirtualHomeConfig | None = None) -> None:
        self.config = config or VirtualHomeConfig()
        self.comm: Any | None = None
        self.task_spec: TaskSpec | None = None
        self.step_id = 0
        self.events: list[Event] = []
        self.last_action: Action | None = None
        self.scene_graph: dict[str, Any] | None = None
        self.executed_scripts: list[list[str]] = []

    def connect(self) -> None:
        """Create a UnityCommunication client if configured to use Unity."""

        if self.comm is not None:
            return
        if not self.config.connect:
            return

        import sys

        sim_path = self.config.repo_root / "virtualhome" / "simulation"
        sys.path.insert(0, str(sim_path))
        from unity_simulator.comm_unity import UnityCommunication

        kwargs: dict[str, Any] = {"port": self.config.port}
        if self.config.unity_executable is not None:
            kwargs["file_name"] = str(self.config.unity_executable)
        self.comm = UnityCommunication(**kwargs)

    def reset(self, task_spec: TaskSpec) -> Observation:
        """Reset simulator/task state and return the initial observation."""

        self.connect()
        self.task_spec = task_spec
        self.step_id = 0
        self.last_action = None
        self.executed_scripts = []
        self.events = self._initial_events(task_spec)
        self.scene_graph = None

        if self.comm is not None:
            reset_result = self.comm.reset(task_spec.scene_id)
            if isinstance(reset_result, tuple) and not reset_result[0]:
                raise RuntimeError(f"VirtualHome reset failed: {reset_result[1]}")
            if reset_result is False:
                raise RuntimeError(f"VirtualHome reset({task_spec.scene_id}) failed")

            if self.config.use_scene_graph:
                success, graph = self.comm.environment_graph()
                if success:
                    self.scene_graph = graph

        return self._observe()

    def step(self, action: Action) -> StepResult:
        """Apply a symbolic action and return the next observation."""

        if self.task_spec is None:
            raise RuntimeError("Call reset(task_spec) before step(action).")

        self.last_action = action
        self.step_id += 1

        try:
            script = to_virtualhome_script(action)
        except ActionTranslationError as exc:
            obs = self._observe(metadata={"error": str(exc)})
            return StepResult(observation=obs, done=True, reward=0.0, info={"error": str(exc)})

        self.executed_scripts.append(script)
        if script and self.comm is not None:
            success, message = self.comm.render_script(script, recording=False, skip_animation=True)
            if not success:
                obs = self._observe(metadata={"error": message, "script": script})
                return StepResult(observation=obs, done=True, reward=0.0, info={"error": message, "script": script})

            if self.config.use_scene_graph:
                graph_success, graph = self.comm.environment_graph()
                if graph_success:
                    self.scene_graph = graph

        obs = self._observe(metadata={"script": script})
        return StepResult(observation=obs, done=False, reward=None, info={"script": script})

    def close(self) -> None:
        """Close environment resources."""

        self.comm = None

    def _observe(self, metadata: dict[str, Any] | None = None) -> Observation:
        assert self.task_spec is not None
        return build_observation(
            step_id=self.step_id,
            task_spec=self.task_spec,
            events=self.events,
            last_action=self.last_action,
            fpv_images=[],
            scene_graph=self.scene_graph,
            metadata=metadata,
        )

    @staticmethod
    def _initial_events(task_spec: TaskSpec) -> list[Event]:
        events: list[Event] = []
        speech = task_spec.e_behavior.get("speech")
        if speech:
            events.append(Event(event_type="speech", actor="E", content=speech, start_time=0.0))

        gaze_target = task_spec.e_behavior.get("gaze_target_id")
        if gaze_target is not None:
            events.append(
                Event(
                    event_type="gaze",
                    actor="E",
                    target_object_id=int(gaze_target),
                    start_time=0.0,
                    duration=task_spec.e_behavior.get("gaze_duration"),
                )
            )

        gesture = task_spec.e_behavior.get("gesture")
        gesture_target = task_spec.e_behavior.get("gesture_target_id")
        if gesture:
            events.append(
                Event(
                    event_type="gesture",
                    actor="E",
                    content=gesture,
                    target_object_id=int(gesture_target) if gesture_target is not None else None,
                    start_time=task_spec.e_behavior.get("gesture_start_time", 0.0),
                    duration=task_spec.e_behavior.get("gesture_duration"),
                    metadata={"intensity": task_spec.e_behavior.get("gesture_intensity")},
                )
            )

        return events

