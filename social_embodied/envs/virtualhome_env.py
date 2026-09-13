"""VirtualHome-backed social embodied environment loop."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import math
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
    controlled_layout: bool = True
    capture_debug_cameras: bool = True
    use_controlled_fpv_camera: bool = True


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
        self.controlled_fpv_camera_index: int | None = None
        self.overview_camera_index: int | None = None
        self.reset_warning: str | None = None
        self.layout_metadata: dict[str, Any] = {}

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
        self.controlled_fpv_camera_index = None
        self.overview_camera_index = None
        self.reset_warning = None
        self.layout_metadata = {}

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
                if self.config.controlled_layout:
                    self._apply_controlled_layout()

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
        debug_images = self._capture_debug_images()
        return build_observation(
            step_id=self.step_id,
            task_spec=self.task_spec,
            events=self.events,
            last_action=self.last_action,
            fpv_images=fpv_images,
            debug_images=debug_images,
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
        if not self.config.capture_fpv or self.comm is None:
            return []
        camera_index = self.controlled_fpv_camera_index or self.t_fpv_camera_index
        if camera_index is None:
            return []
        success, images = self.comm.camera_image(
            [camera_index],
            mode="normal",
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        if not success:
            return []
        return images

    def _capture_debug_images(self) -> dict[str, list[Any]]:
        if not self.config.capture_debug_cameras or self.comm is None or self.overview_camera_index is None:
            return {}
        success, images = self.comm.camera_image(
            [self.overview_camera_index],
            mode="normal",
            image_width=self.config.image_width,
            image_height=self.config.image_height,
        )
        if not success:
            return {}
        return {"overview": images}

    def _observation_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        result = dict(metadata or {})
        result.update(
            {
                "static_camera_count": self.static_camera_count,
                "character_camera_names": self.character_camera_names,
                "t_fpv_camera_index": self.t_fpv_camera_index,
                "controlled_fpv_camera_index": self.controlled_fpv_camera_index,
                "overview_camera_index": self.overview_camera_index,
                "executed_scripts": self.executed_scripts,
                "reset_warning": self.reset_warning,
                "layout": self.layout_metadata,
            }
        )
        return result

    def _materialize_ambiguous_reference(self, task_spec: TaskSpec) -> TaskSpec:
        candidates = self._find_candidate_objects(["mug", "cup", "waterglass", "wineglass", "juiceglass"])
        if len(candidates) < 2:
            candidates = self._find_candidate_objects(["book", "remotecontrol", "cellphone", "apple"])
        if len(candidates) < 2:
            return task_spec

        selected = self._select_nearby_pair(candidates)
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

    def _select_nearby_pair(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        same_class_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        any_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for idx, left in enumerate(candidates):
            for right in candidates[idx + 1 :]:
                dist = self._distance_xz(self._node_center(left), self._node_center(right))
                pair = (dist, left, right)
                any_pairs.append(pair)
                if left.get("class_name") == right.get("class_name"):
                    same_class_pairs.append(pair)

        pairs = same_class_pairs or any_pairs
        if not pairs:
            return candidates[:2]
        _, left, right = min(pairs, key=lambda item: item[0])
        return [left, right]

    def _apply_controlled_layout(self) -> None:
        if self.comm is None or self.task_spec is None:
            return
        target = self.task_spec.objects.get("target", {})
        candidates = self.task_spec.objects.get("candidates", [])
        target_id = target.get("id")
        target_class = target.get("class_name")
        if target_id is None or target_class is None or len(candidates) < 2:
            return

        candidate_nodes = {
            int(node["id"]): node
            for node in self._find_candidate_objects([str(item["class_name"]) for item in candidates])
            if node.get("id") is not None
        }
        selected_nodes = [candidate_nodes.get(int(item["id"])) for item in candidates]
        selected_nodes = [node for node in selected_nodes if node is not None]
        target_node = candidate_nodes.get(int(target_id))
        if not selected_nodes or target_node is None:
            return

        centers = [self._node_center(node) for node in selected_nodes]
        center = [
            sum(pos[0] for pos in centers) / len(centers),
            0.0,
            sum(pos[2] for pos in centers) / len(centers),
        ]

        t_pos = [center[0], 0.0, center[2] + 1.8]
        e_pos = [center[0] + 1.4, 0.0, center[2] + 0.7]

        t_move_success = self.comm.move_character(0, t_pos)
        e_move_success = self.comm.move_character(1, e_pos)

        orient_script = [
            f"<char0> [lookat] <{target_class}> ({target_id}) | <char1> [lookat] <{target_class}> ({target_id})"
        ]
        orient_success, orient_message = self.comm.render_script(
            orient_script,
            recording=False,
            skip_animation=True,
            image_synthesis=[],
            processing_time_limit=20,
        )

        self.layout_metadata = {
            "candidate_ids": [item.get("id") for item in candidates],
            "target_id": target_id,
            "target_class": target_class,
            "candidate_centers": {int(node["id"]): self._node_center(node) for node in selected_nodes},
            "layout_center": center,
            "target_agent_position": t_pos,
            "environment_agent_position": e_pos,
            "move_character_success": {"T": t_move_success, "E": e_move_success},
            "orient_script": orient_script,
            "orient_success": orient_success,
            "orient_message": orient_message,
        }

        if self.config.capture_debug_cameras:
            self._add_overview_camera(center)
        if self.config.use_controlled_fpv_camera:
            self._add_controlled_fpv_camera(t_pos, self._node_center(target_node))

        if self.config.use_scene_graph:
            graph_success, graph = self.comm.environment_graph()
            if graph_success:
                self.scene_graph = graph

    def _add_controlled_fpv_camera(self, t_position: list[float], target_center: list[float]) -> None:
        if self.comm is None:
            return
        success, camera_count = self.comm.camera_count()
        if not success:
            return
        self.controlled_fpv_camera_index = int(camera_count)
        camera_pos = [t_position[0], 1.45, t_position[2]]
        rotation = self._look_at_euler(camera_pos, target_center)
        add_success, add_message = self.comm.add_camera(
            position=camera_pos,
            rotation=rotation,
            field_view=70,
        )
        self.layout_metadata["controlled_fpv_camera"] = {
            "index": self.controlled_fpv_camera_index,
            "position": camera_pos,
            "rotation": rotation,
            "success": add_success,
            "message": add_message,
        }

    def _add_overview_camera(self, center: list[float]) -> None:
        if self.comm is None:
            return
        success, camera_count = self.comm.camera_count()
        if not success:
            return
        self.overview_camera_index = int(camera_count)
        camera_pos = [center[0], 4.0, center[2] + 3.2]
        rotation = self._look_at_euler(camera_pos, [center[0], 1.0, center[2]])
        add_success, add_message = self.comm.add_camera(
            position=camera_pos,
            rotation=rotation,
            field_view=65,
        )
        self.layout_metadata["overview_camera"] = {
            "index": self.overview_camera_index,
            "position": camera_pos,
            "rotation": rotation,
            "success": add_success,
            "message": add_message,
        }

    @staticmethod
    def _node_center(node: dict[str, Any]) -> list[float]:
        bbox = node.get("bounding_box") or {}
        center = bbox.get("center")
        if center:
            return [float(center[0]), float(center[1]), float(center[2])]
        transform = node.get("obj_transform") or {}
        position = transform.get("position") or [0.0, 0.0, 0.0]
        return [float(position[0]), float(position[1]), float(position[2])]

    @staticmethod
    def _distance_xz(left: list[float], right: list[float]) -> float:
        return math.sqrt((left[0] - right[0]) ** 2 + (left[2] - right[2]) ** 2)

    @staticmethod
    def _look_at_euler(position: list[float], target: list[float]) -> list[float]:
        dx = target[0] - position[0]
        dy = target[1] - position[1]
        dz = target[2] - position[2]
        horizontal = math.sqrt(dx * dx + dz * dz)
        pitch = -math.degrees(math.atan2(dy, horizontal))
        yaw = math.degrees(math.atan2(dx, dz))
        return [pitch, yaw, 0.0]

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
