"""Run Task 1 social reference ambiguity scenarios."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from social_embodied.agents import SocialCueOracleAgent, TargetObjectOracleAgent
from social_embodied.debugging import write_episode_trace
from social_embodied.envs import SocialEmbodiedEnv, VirtualHomeConfig
from social_embodied.evaluation import run_episode
from social_embodied.tasks import AmbiguousReferenceScorer, build_task_1_1_spec, load_task_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 1 Scenario 1.1.")
    parser.add_argument("--connect", action="store_true", help="Connect to a running VirtualHome Unity simulator.")
    parser.add_argument("--port", default="8080", help="Unity simulator HTTP port.")
    parser.add_argument(
        "--unity-executable",
        type=Path,
        help="Optional Unity executable/app path for UnityCommunication to launch.",
    )
    parser.add_argument(
        "--task-instance",
        type=Path,
        help="Path to a JSON task instance. Defaults to Task 1.1 Scenario 1.",
    )
    parser.add_argument(
        "--agent",
        choices=["target-oracle", "social-cue-oracle"],
        default="target-oracle",
        help="Baseline agent to run.",
    )
    parser.add_argument("--no-fpv", action="store_true", help="Disable FPV image capture.")
    parser.add_argument("--debug-dir", type=Path, default=Path("debug"), help="Root directory for local debug traces.")
    parser.add_argument("--debug-run-name", help="Optional debug run folder name.")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()

    task_spec = load_task_spec(args.task_instance) if args.task_instance else build_task_1_1_spec()
    agent = TargetObjectOracleAgent() if args.agent == "target-oracle" else SocialCueOracleAgent()
    env = SocialEmbodiedEnv(
        VirtualHomeConfig(
            connect=args.connect,
            port=args.port,
            unity_executable=args.unity_executable,
            capture_fpv=not args.no_fpv,
        )
    )
    scorer = AmbiguousReferenceScorer()

    result = run_episode(env=env, agent=agent, task_spec=task_spec, scorer=scorer, max_steps=args.max_steps)
    debug_trace = write_episode_trace(result, args.debug_dir, run_name=args.debug_run_name)
    payload = {
        "task_id": result.task_id,
        "agent": result.agent_name,
        "metrics": result.metrics,
        "debug_trace": debug_trace,
        "actions": [
            {
                "type": action.action_type,
                "target_object_id": action.target_object_id,
                "utterance": action.utterance,
                "params": action.params,
            }
            for action in result.actions
        ],
        "initial_observation": {
            "events": [
                {
                    "type": event.event_type,
                    "actor": event.actor,
                    "target_object_id": event.target_object_id,
                    "content": event.content,
                    "duration": event.duration,
                }
                for event in result.observations[0].events
            ],
            "metadata": result.observations[0].metadata,
        },
        "final_observation": {
            "num_fpv_images": len(result.observations[-1].fpv_images),
            "debug_image_channels": {
                channel: len(images) for channel, images in result.observations[-1].debug_images.items()
            },
            "num_visible_objects": len(result.observations[-1].visible_objects),
            "metadata": result.observations[-1].metadata,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
