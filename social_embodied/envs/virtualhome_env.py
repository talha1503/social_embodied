"""VirtualHome-backed social embodied environment loop."""

from __future__ import annotations

import copy
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
    capture_segmentation: bool = True
    segmentation_color_tolerance: int = 3


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
        self.instance_color_map: dict[str, Any] = {}
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
        self.instance_color_map = {}
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
            self.task_spec = self._resolve_symbolic_task_references(self.task_spec)
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
            try:
                success, message = self.comm.render_script(script, recording=False, skip_animation=True)
            except Exception as exc:
                message = f"Unity render_script failed: {exc}"
                obs = self._observe(metadata={"error": message, "script": script})
                return StepResult(observation=obs, done=True, reward=0.0, info={"error": message, "script": script})
            if not success:
                obs = self._observe(metadata={"error": message, "script": script})
                return StepResult(observation=obs, done=True, reward=0.0, info={"error": message, "script": script})

            if self.config.use_scene_graph:
                try:
                    graph_success, graph = self.comm.environment_graph()
                except Exception as exc:
                    message = f"Unity environment_graph failed after script: {exc}"
                    obs = self._observe(metadata={"error": message, "script": script})
                    return StepResult(observation=obs, done=True, reward=0.0, info={"error": message, "script": script})
                else:
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
        visibility = self._visibility_metadata(debug_images)
        return build_observation(
            step_id=self.step_id,
            task_spec=self.task_spec,
            events=self.events,
            last_action=self.last_action,
            fpv_images=fpv_images,
            debug_images=debug_images,
            scene_graph=self.scene_graph,
            metadata=self._observation_metadata(metadata, visibility=visibility),
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
            self._agent_config("target", "character", self.config.target_character),
            initial_room=self._agent_config("target", "initial_room", self.config.target_initial_room),
        )
        self.comm.add_character(
            self._agent_config("environment", "character", self.config.environment_character),
            initial_room=self._agent_config("environment", "initial_room", self.config.environment_initial_room),
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
        camera_index = self._fpv_camera_index()
        if camera_index is None:
            return []
        try:
            success, images = self.comm.camera_image(
                [camera_index],
                mode="normal",
                image_width=self.config.image_width,
                image_height=self.config.image_height,
            )
        except Exception:
            return []
        if not success:
            return []
        return images

    def _capture_debug_images(self) -> dict[str, list[Any]]:
        if self.comm is None:
            return {}
        debug_images: dict[str, list[Any]] = {}
        if self.config.capture_debug_cameras and self.overview_camera_index is not None:
            try:
                success, images = self.comm.camera_image(
                    [self.overview_camera_index],
                    mode="normal",
                    image_width=self.config.image_width,
                    image_height=self.config.image_height,
                )
            except Exception:
                success, images = False, []
            if success:
                debug_images["overview"] = images

        if self.config.capture_segmentation:
            fpv_camera_index = self._fpv_camera_index()
            if fpv_camera_index is not None:
                try:
                    success, images = self.comm.camera_image(
                        [fpv_camera_index],
                        mode="seg_inst",
                        image_width=self.config.image_width,
                        image_height=self.config.image_height,
                    )
                except Exception:
                    success, images = False, []
                if success:
                    debug_images["fpv_seg_inst"] = images
            if self.overview_camera_index is not None:
                try:
                    success, images = self.comm.camera_image(
                        [self.overview_camera_index],
                        mode="seg_inst",
                        image_width=self.config.image_width,
                        image_height=self.config.image_height,
                    )
                except Exception:
                    success, images = False, []
                if success:
                    debug_images["overview_seg_inst"] = images
        return debug_images

    def _fpv_camera_index(self) -> int | None:
        return self.controlled_fpv_camera_index or self.t_fpv_camera_index

    def _observation_metadata(
        self,
        metadata: dict[str, Any] | None,
        *,
        visibility: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = dict(metadata or {})
        result.update(
            {
                "static_camera_count": self.static_camera_count,
                "character_camera_names": self.character_camera_names,
                "t_fpv_camera_index": self.t_fpv_camera_index,
                "controlled_fpv_camera_index": self.controlled_fpv_camera_index,
                "overview_camera_index": self.overview_camera_index,
                "executed_scripts": copy.deepcopy(self.executed_scripts),
                "reset_warning": self.reset_warning,
                "layout": copy.deepcopy(self.layout_metadata),
                "visibility": visibility or {},
            }
        )
        return result

    def _visibility_metadata(self, debug_images: dict[str, list[Any]]) -> dict[str, Any]:
        if self.comm is None or self.task_spec is None:
            return {}
        seg_images = {
            "fpv": debug_images.get("fpv_seg_inst", []),
            "overview": debug_images.get("overview_seg_inst", []),
        }
        if not any(seg_images.values()):
            return {}

        instance_colors = self._instance_colors()
        if not instance_colors:
            return {}

        target = self.task_spec.objects.get("target", {})
        candidates = self.task_spec.objects.get("candidates", [])
        objects = []
        seen_ids = set()
        for item in [target, *candidates]:
            object_id = item.get("id")
            if object_id is None or int(object_id) in seen_ids:
                continue
            seen_ids.add(int(object_id))
            objects.append(
                {
                    "id": int(object_id),
                    "class_name": item.get("class_name"),
                    "role": "target" if object_id == target.get("id") else "candidate",
                }
            )

        per_object = []
        for item in objects:
            color = instance_colors.get(str(item["id"]))
            pixels = {
                channel: self._count_instance_pixels(images, color)
                for channel, images in seg_images.items()
            }
            per_object.append(
                {
                    **item,
                    "pixels": pixels,
                    "visible_in_fpv": pixels.get("fpv", 0) > 0,
                    "visible_in_overview": pixels.get("overview", 0) > 0,
                }
            )

        target_record = next((item for item in per_object if item["role"] == "target"), None)
        candidate_records = [item for item in per_object if item["role"] == "candidate"]
        return {
            "per_object": per_object,
            "target_visible_in_fpv": bool(target_record and target_record["visible_in_fpv"]),
            "target_visible_in_overview": bool(target_record and target_record["visible_in_overview"]),
            "num_candidates_visible_in_fpv": sum(1 for item in candidate_records if item["visible_in_fpv"]),
            "num_candidates_visible_in_overview": sum(1 for item in candidate_records if item["visible_in_overview"]),
        }

    def _instance_colors(self) -> dict[str, Any]:
        if self.instance_color_map or self.comm is None:
            return self.instance_color_map
        try:
            success, colors = self.comm.instance_colors()
        except Exception:
            return {}
        if success:
            self.instance_color_map = colors
        return self.instance_color_map

    def _count_instance_pixels(self, images: list[Any], color: Any) -> int:
        if not images or not color:
            return 0
        try:
            import numpy as np
        except ImportError:
            return 0

        rgb = np.asarray([float(color[0]), float(color[1]), float(color[2])]) * 255.0
        bgr = rgb[::-1]
        total = 0
        tolerance = self.config.segmentation_color_tolerance
        for image in images:
            arr = np.asarray(image)
            if arr.ndim != 3 or arr.shape[-1] < 3:
                continue
            arr = arr[:, :, :3].astype(float)
            rgb_match = np.all(np.abs(arr - rgb) <= tolerance, axis=-1)
            bgr_match = np.all(np.abs(arr - bgr) <= tolerance, axis=-1)
            total += int(np.count_nonzero(rgb_match | bgr_match))
        return total

    def _materialize_ambiguous_reference(self, task_spec: TaskSpec) -> TaskSpec:
        selection = task_spec.metadata.get("object_selection", {})
        selected = self._select_fixed_candidates(selection)
        if selected:
            return self._task_with_selected_candidates(task_spec, selected, selection)

        candidate_classes = selection.get("candidate_classes", ["mug", "cup", "waterglass", "wineglass", "juiceglass"])
        fallback_classes = selection.get("fallback_classes", ["book", "remotecontrol", "cellphone", "apple"])
        candidates = self._find_candidate_objects([str(item) for item in candidate_classes])
        if len(candidates) < 2:
            candidates = self._find_candidate_objects([str(item) for item in fallback_classes])
        if len(candidates) < 2:
            return task_spec

        selected = self._select_candidate_pair(candidates, selection)
        return self._task_with_selected_candidates(task_spec, selected, selection)

    def _task_with_selected_candidates(
        self,
        task_spec: TaskSpec,
        selected: list[dict[str, Any]],
        selection: dict[str, Any],
    ) -> TaskSpec:
        target_index = int(selection.get("target_index", len(selected) - 1))
        target_id_config = selection.get("target_id")
        if target_id_config is not None:
            for idx, node in enumerate(selected):
                if int(node["id"]) == int(target_id_config):
                    target_index = idx
                    break
        target_index = max(0, min(target_index, len(selected) - 1))
        target = selected[target_index]
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

    def _select_fixed_candidates(self, selection: dict[str, Any]) -> list[dict[str, Any]]:
        if selection.get("mode") != "fixed_ids" or not self.scene_graph:
            return []
        candidate_ids = [int(item) for item in selection.get("candidate_ids", [])]
        if len(candidate_ids) < 2:
            return []
        nodes_by_id = {
            int(node["id"]): node
            for node in self.scene_graph.get("nodes", [])
            if node.get("id") is not None
        }
        selected = [nodes_by_id[item] for item in candidate_ids if item in nodes_by_id]
        return selected if len(selected) >= 2 else []

    def _resolve_symbolic_task_references(self, task_spec: TaskSpec) -> TaskSpec:
        target = task_spec.objects.get("target", {})
        target_id = target.get("id")
        if target_id is None:
            return task_spec

        e_behavior = dict(task_spec.e_behavior)
        for key in ("gaze_target_id", "gesture_target_id"):
            if e_behavior.get(key) == "target":
                e_behavior[key] = int(target_id)

        success = dict(task_spec.success)
        if success.get("target_object_id") is None:
            success["target_object_id"] = int(target_id)

        return replace(task_spec, e_behavior=e_behavior, success=success)

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

    def _select_candidate_pair(
        self,
        candidates: list[dict[str, Any]],
        selection: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        selection = selection or {}
        mode = selection.get("mode", "auto_nearby_pair")
        if mode not in {"auto_nearby_pair", "fixed_ids"}:
            raise ValueError(f"Unsupported object_selection mode: {mode}")

        same_class_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        any_pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for idx, left in enumerate(candidates):
            for right in candidates[idx + 1 :]:
                dist = self._distance_xz(self._node_center(left), self._node_center(right))
                pair = (dist, left, right)
                any_pairs.append(pair)
                if left.get("class_name") == right.get("class_name"):
                    same_class_pairs.append(pair)

        pairs = same_class_pairs if selection.get("require_same_class", True) else same_class_pairs or any_pairs
        if not pairs and selection.get("require_same_class", True):
            pairs = any_pairs
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

        layout = self.task_spec.metadata.get("layout", {})
        t_offset = self._vector_config(layout, "target_agent_offset", [0.0, 0.0, 1.8])
        e_offset = self._vector_config(layout, "environment_agent_offset", [1.4, 0.0, 0.7])

        t_pos = [center[0] + t_offset[0], t_offset[1], center[2] + t_offset[2]]
        e_pos = [center[0] + e_offset[0], e_offset[1], center[2] + e_offset[2]]

        t_move_success = self.comm.move_character(0, t_pos)
        e_move_success = self.comm.move_character(1, e_pos)

        orient_script = []
        orient_success = None
        orient_message = None
        if layout.get("orient_agents_to_target", True):
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

        self._apply_visible_social_cues(target_class=str(target_class), target_id=int(target_id))

        overview_config = layout.get("overview_camera", {})
        if self.config.capture_debug_cameras and overview_config.get("enabled", True):
            self._add_overview_camera(center)
        fpv_config = layout.get("controlled_fpv_camera", {})
        if self.config.use_controlled_fpv_camera and fpv_config.get("enabled", True):
            self._add_controlled_fpv_camera(t_pos, self._node_center(target_node), center)

        if self.config.use_scene_graph:
            graph_success, graph = self.comm.environment_graph()
            if graph_success:
                self.scene_graph = graph

    def _add_controlled_fpv_camera(
        self,
        t_position: list[float],
        target_center: list[float],
        candidate_center: list[float],
    ) -> None:
        if self.comm is None:
            return
        success, camera_count = self.comm.camera_count()
        if not success:
            return
        self.controlled_fpv_camera_index = int(camera_count)
        fpv_config = self.task_spec.metadata.get("layout", {}).get("controlled_fpv_camera", {}) if self.task_spec else {}
        camera_height = float(fpv_config.get("height", 1.45))
        field_view = float(fpv_config.get("field_view", 70))
        camera_pos = [t_position[0], camera_height, t_position[2]]
        if fpv_config.get("look_at") == "candidate_center":
            look_at_height = float(fpv_config.get("look_at_height", 0.85))
            look_at = [candidate_center[0], look_at_height, candidate_center[2]]
        else:
            look_at = target_center
        rotation = self._look_at_euler(camera_pos, look_at)
        add_success, add_message = self.comm.add_camera(
            position=camera_pos,
            rotation=rotation,
            field_view=field_view,
        )
        self.layout_metadata["controlled_fpv_camera"] = {
            "index": self.controlled_fpv_camera_index,
            "position": camera_pos,
            "look_at": look_at,
            "rotation": rotation,
            "success": add_success,
            "message": add_message,
        }

    def _add_overview_camera(self, center: list[float]) -> None:
        if self.comm is None:
            return
        layout = self.task_spec.metadata.get("layout", {}) if self.task_spec else {}
        overview_config = layout.get("overview_camera", {})
        success, camera_count = self.comm.camera_count()
        if not success:
            return
        self.overview_camera_index = int(camera_count)
        offset = self._vector_config(overview_config, "offset", [0.0, 4.0, 3.2])
        look_at_offset = self._vector_config(overview_config, "look_at_offset", [0.0, 1.0, 0.0])
        field_view = float(overview_config.get("field_view", 65))
        camera_pos = [center[0] + offset[0], center[1] + offset[1], center[2] + offset[2]]
        look_at = [center[0] + look_at_offset[0], center[1] + look_at_offset[1], center[2] + look_at_offset[2]]
        rotation = self._look_at_euler(camera_pos, look_at)
        add_success, add_message = self.comm.add_camera(
            position=camera_pos,
            rotation=rotation,
            field_view=field_view,
        )
        self.layout_metadata["overview_camera"] = {
            "index": self.overview_camera_index,
            "position": camera_pos,
            "rotation": rotation,
            "success": add_success,
            "message": add_message,
        }

    def _apply_visible_social_cues(self, *, target_class: str, target_id: int) -> None:
        if self.comm is None or self.task_spec is None:
            return

        cue_scripts: list[dict[str, str]] = []
        if self.task_spec.e_behavior.get("gaze_visible"):
            cue_scripts.append({"cue": "gaze", "script": f"<char1> [lookat] <{target_class}> ({target_id})"})

        gesture = str(self.task_spec.e_behavior.get("gesture") or "").lower()
        if self.task_spec.e_behavior.get("gesture_visible") and gesture in {"point", "pointat", "point_at"}:
            cue_scripts.append({"cue": "gesture", "script": f"<char1> [pointat] <{target_class}> ({target_id})"})

        if not cue_scripts:
            self.layout_metadata["visible_social_cues"] = {"records": [], "success": None}
            return

        records = []
        for cue_script in cue_scripts:
            success, message = self.comm.render_script(
                [cue_script["script"]],
                recording=False,
                skip_animation=True,
                image_synthesis=[],
                processing_time_limit=20,
            )
            records.append({**cue_script, "success": success, "message": message})

        self.layout_metadata["visible_social_cues"] = {
            "records": records,
            "success": all(record["success"] for record in records),
        }

    def _agent_config(self, agent_role: str, key: str, default: str) -> str:
        if self.task_spec is None:
            return default
        agents = self.task_spec.metadata.get("agents", {})
        agent = agents.get(agent_role, {}) if isinstance(agents, dict) else {}
        return str(agent.get(key, default))

    @staticmethod
    def _vector_config(config: dict[str, Any], key: str, default: list[float]) -> list[float]:
        value = config.get(key, default) if isinstance(config, dict) else default
        if not isinstance(value, (list, tuple)) or len(value) != 3:
            return default
        return [float(value[0]), float(value[1]), float(value[2])]

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
