"""VirtualHome-backed social embodied environment loop."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
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
    target_character: str = "Chars/Male1"
    environment_character: str = "Chars/Female2"
    target_initial_room: str = "kitchen"
    environment_initial_room: str = "livingroom"
    capture_fpv: bool = True
    strict_reset: bool = False


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
        self.static_camera_count: int | None = None
        self.character_camera_names: list[str] = []
        self.t_fpv_camera_index: int | None = None
        self.reset_warning: str | None = None

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
        self.events = []
        self.scene_graph = None
        self.static_camera_count = None
        self.character_camera_names = []
        self.t_fpv_camera_index = None
        self.reset_warning = None

        if self.comm is not None:
            reset_result = self.comm.reset(task_spec.scene_id)
            if isinstance(reset_result, tuple) and not reset_result[0]:
                self.reset_warning = f"VirtualHome reset returned False: {reset_result[1]}"
                if self.config.strict_reset:
                    raise RuntimeError(self.reset_warning)
            if reset_result is False:
                self.reset_warning = f"VirtualHome reset({task_spec.scene_id}) returned False"
                if self.config.strict_reset:
                    raise RuntimeError(self.reset_warning)

            self._setup_characters()

            if self.config.use_scene_graph:
                success, graph = self.comm.environment_graph()
                if success:
                    self.scene_graph = graph

            if task_spec.task_family == "ambiguous_reference":
                self.task_spec = self._materialize_ambiguous_reference(task_spec)

        self.events = self._initial_events(self.task_spec)

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
        fpv_images = self._capture_fpv_images()
        return build_observation(
            step_id=self.step_id,
            task_spec=self.task_spec,
            events=self.events,
            last_action=self.last_action,
            fpv_images=fpv_images,
            scene_graph=self.scene_graph,
            metadata=self._observation_metadata(metadata),
        )

    def _setup_characters(self) -> None:
        assert self.comm is not None

        success, static_count = self.comm.camera_count()
        if success:
            self.static_camera_count = int(static_count)

        names_success, names_payload = self.comm.character_cameras()
        if names_success:
            self.character_camera_names = self._parse_character_camera_names(names_payload)

        # Character order matters for scripts: T is char0, E is char1.
        self.comm.add_character(
            self.config.target_character,
            initial_room=self.config.target_initial_room,
        )
        self.comm.add_character(
            self.config.environment_character,
            initial_room=self.config.environment_initial_room,
        )

        self.t_fpv_camera_index = self._character_camera_index(character_index=0, camera_name=self.config.camera_mode)

    @staticmethod
    def _parse_character_camera_names(payload: Any) -> list[str]:
        if isinstance(payload, list):
            return [str(item) for item in payload]
        if isinstance(payload, str):
            import json

            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return []

    def _character_camera_index(self, *, character_index: int, camera_name: str) -> int | None:
        if self.static_camera_count is None or not self.character_camera_names:
            return None
        try:
            offset = self.character_camera_names.index(camera_name)
        except ValueError:
            return None
        return self.static_camera_count + character_index * len(self.character_camera_names) + offset

    def _capture_fpv_images(self) -> list[Any]:
        if not self.config.capture_fpv or self.comm is None or self.t_fpv_camera_index is None:
            return []
        success, images = self.comm.camera_image(
            [self.t_fpv_camera_index],
            mode="normal",
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        if not success:
            return []
        return images

    def _observation_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        result = dict(metadata or {})
        result.update(
            {
                "static_camera_count": self.static_camera_count,
                "character_camera_names": self.character_camera_names,
                "t_fpv_camera_index": self.t_fpv_camera_index,
                "executed_scripts": self.executed_scripts,
                "reset_warning": self.reset_warning,
            }
        )
        return result

    def _materialize_ambiguous_reference(self, task_spec: TaskSpec) -> TaskSpec:
        candidates = self._find_candidate_objects(["mug", "cup", "waterglass", "wineglass", "juiceglass"])
        if len(candidates) < 2:
            candidates = self._find_candidate_objects(["book", "remotecontrol", "cellphone", "apple"])
        if len(candidates) < 2:
            return task_spec

        selected = candidates[:2]
        target = selected[-1]
        target_id = int(target["id"])

        objects = {
            **task_spec.objects,
            "candidates": [
                {"id": int(obj["id"]), "class_name": str(obj["class_name"]), "label": f"{obj['class_name']}_{obj['id']}"}
                for obj in selected
            ],
            "target": {
                "id": target_id,
                "class_name": str(target["class_name"]),
                "label": f"{target['class_name']}_{target_id}",
            },
        }
        e_behavior = {
            **task_spec.e_behavior,
            "gaze_target_id": target_id,
            "gesture_target_id": target_id,
        }
        success = {
            **task_spec.success,
            "target_object_id": target_id,
        }
        return replace(task_spec, objects=objects, e_behavior=e_behavior, success=success)

    def _find_candidate_objects(self, class_names: list[str]) -> list[dict[str, Any]]:
        if not self.scene_graph:
            return []
        wanted = set(class_names)
        candidates = []
        for node in self.scene_graph.get("nodes", []):
            if node.get("class_name") in wanted and node.get("id") is not None:
                candidates.append(node)
        candidates.sort(key=lambda node: (str(node.get("class_name")), int(node.get("id"))))
        return candidates

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
