"""Run the Task 0 dry-run benchmark loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from social_embodied.agents import SocialCueOracleAgent
from social_embodied.envs import SocialEmbodiedEnv, VirtualHomeConfig
from social_embodied.evaluation import run_episode
from social_embodied.tasks import AmbiguousReferenceScorer, build_task_0_spec


def save_fpv_images(result: Any, output_dir: Path) -> list[str]:
    """Save every FPV image captured during an episode."""

    frames = [
        (obs_idx, image_idx, image)
        for obs_idx, observation in enumerate(result.observations)
        for image_idx, image in enumerate(observation.fpv_images)
    ]
    if not frames:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Saving FPV images requires pillow and numpy.") from exc

    for obs_idx, image_idx, image in frames:
        arr = np.asarray(image)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            # VirtualHome decodes PNGs through OpenCV, so RGB frames arrive as BGR.
            arr = arr[:, :, ::-1]
        path = output_dir / f"obs_{obs_idx:03d}_fpv_{image_idx:02d}.png"
        Image.fromarray(arr).save(path)
        saved.append(str(path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 0 ambiguous reference baseline.")
    parser.add_argument("--connect", action="store_true", help="Connect to a running VirtualHome Unity simulator.")
    parser.add_argument("--port", default="8080", help="Unity simulator HTTP port.")
    parser.add_argument("--no-fpv", action="store_true", help="Disable FPV image capture.")
    parser.add_argument("--save-fpv-dir", type=Path, help="Directory for captured FPV debug images.")
    parser.add_argument("--max-steps", type=int, default=10)
    args = parser.parse_args()

    task_spec = build_task_0_spec()
    agent = SocialCueOracleAgent()
    env = SocialEmbodiedEnv(
        VirtualHomeConfig(
            connect=args.connect,
            port=args.port,
            capture_fpv=not args.no_fpv,
        )
    )
    scorer = AmbiguousReferenceScorer()

    result = run_episode(env=env, agent=agent, task_spec=task_spec, scorer=scorer, max_steps=args.max_steps)
    saved_fpv_paths = save_fpv_images(result, args.save_fpv_dir) if args.save_fpv_dir else []
    payload = {
        "metrics": result.metrics,
        "saved_fpv_paths": saved_fpv_paths,
        "actions": [
            {
                "type": action.action_type,
                "target_object_id": action.target_object_id,
                "utterance": action.utterance,
                "params": action.params,
            }
            for action in result.actions
        ],
        "final_observation": {
            "num_fpv_images": len(result.observations[-1].fpv_images),
            "num_visible_objects": len(result.observations[-1].visible_objects),
            "metadata": result.observations[-1].metadata,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
