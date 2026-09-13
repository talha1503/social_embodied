"""Write local debug traces for benchmark episodes."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


def write_episode_trace(result: Any, output_root: Path, *, run_name: str | None = None) -> dict[str, Any]:
    """Persist a complete local trace for one episode.

    The trace layout is:

    - ``trace.json``: task, actions, metrics, observation metadata, visible objects, and image paths.
    - ``images/``: FPV and debug camera frames, grouped by observation index and channel.
    """

    run_dir = _unique_run_dir(output_root, run_name or _default_run_name(result))
    images_dir = run_dir / "images"
    env_dir = run_dir / "env"
    images_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    observations = []
    saved_images: list[str] = []
    saved_env: list[str] = []
    for obs_idx, observation in enumerate(result.observations):
        image_records = _save_observation_images(observation, images_dir, obs_idx)
        env_record = _save_observation_env(observation, env_dir, obs_idx)
        saved_images.extend(record["path"] for record in image_records)
        if env_record is not None:
            saved_env.append(env_record["path"])
        observations.append(
            {
                "index": obs_idx,
                "step_id": observation.step_id,
                "task_prompt": observation.task_prompt,
                "events": _jsonable(observation.events),
                "last_action": _jsonable(observation.last_action),
                "visible_objects_summary": _visible_objects_summary(observation.visible_objects),
                "scene_graph_summary": _scene_graph_summary(observation.scene_graph),
                "scene_graph_path": env_record["path"] if env_record else None,
                "metadata": _jsonable(observation.metadata),
                "images": image_records,
            }
        )

    trace = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "task_id": result.task_id,
        "agent_name": result.agent_name,
        "run_dir": str(run_dir),
        "task": _task_from_observations(result),
        "metrics": _jsonable(result.metrics),
        "actions": _jsonable(result.actions),
        "observations": observations,
        "saved_images": saved_images,
        "saved_env": saved_env,
    }

    trace_path = run_dir / "trace.json"
    trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "trace_path": str(trace_path),
        "saved_images": saved_images,
        "saved_env": saved_env,
    }


def _save_observation_env(observation: Any, env_dir: Path, obs_idx: int) -> dict[str, str] | None:
    if not observation.scene_graph:
        return None
    path = env_dir / f"obs_{obs_idx:03d}_scene_graph.json"
    path.write_text(json.dumps(_jsonable(observation.scene_graph), indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path)}


def _save_observation_images(observation: Any, images_dir: Path, obs_idx: int) -> list[dict[str, Any]]:
    frames = [
        ("fpv", image_idx, image)
        for image_idx, image in enumerate(observation.fpv_images)
    ]
    for channel, images in observation.debug_images.items():
        for image_idx, image in enumerate(images):
            frames.append((channel, image_idx, image))

    records = []
    for channel, image_idx, image in frames:
        filename = f"obs_{obs_idx:03d}_{channel}_{image_idx:02d}.png"
        path = images_dir / filename
        _save_image(image, path)
        records.append(
            {
                "channel": channel,
                "observation_index": obs_idx,
                "image_index": image_idx,
                "path": str(path),
            }
        )
    return records


def _save_image(image: Any, path: Path) -> None:
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Writing debug traces with images requires pillow and numpy.") from exc

    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        # VirtualHome decodes PNGs through OpenCV, so RGB frames arrive as BGR.
        arr = arr[:, :, ::-1]
    Image.fromarray(arr).save(path)


def _default_run_name(result: Any) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{result.task_id}_{timestamp}"


def _unique_run_dir(output_root: Path, run_name: str) -> Path:
    run_dir = output_root / run_name
    if not run_dir.exists():
        return run_dir

    suffix = 1
    while True:
        candidate = output_root / f"{run_name}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def _task_from_observations(result: Any) -> dict[str, Any]:
    if not result.observations:
        return {}
    final_metadata = result.observations[-1].metadata
    return {
        "prompt": result.observations[0].task_prompt,
        "layout": _jsonable(final_metadata.get("layout", {})),
        "reset_warning": final_metadata.get("reset_warning"),
    }


def _scene_graph_summary(scene_graph: dict[str, Any] | None) -> dict[str, Any]:
    if not scene_graph:
        return {"num_nodes": 0, "num_edges": 0}
    return {
        "num_nodes": len(scene_graph.get("nodes", [])),
        "num_edges": len(scene_graph.get("edges", [])),
    }


def _visible_objects_summary(visible_objects: list[Any]) -> dict[str, Any]:
    class_counts = Counter(obj.class_name for obj in visible_objects)
    return {
        "num_objects": len(visible_objects),
        "class_counts": dict(sorted(class_counts.items())),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value
