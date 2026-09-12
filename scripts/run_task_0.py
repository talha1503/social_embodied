"""Run the Task 0 dry-run benchmark loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
import argparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from social_embodied.agents import SocialCueOracleAgent
from social_embodied.envs import SocialEmbodiedEnv, VirtualHomeConfig
from social_embodied.evaluation import run_episode
from social_embodied.tasks import AmbiguousReferenceScorer, build_task_0_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 0 ambiguous reference baseline.")
    parser.add_argument("--connect", action="store_true", help="Connect to a running VirtualHome Unity simulator.")
    parser.add_argument("--port", default="8080", help="Unity simulator HTTP port.")
    parser.add_argument("--no-fpv", action="store_true", help="Disable FPV image capture.")
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
    payload = {
        "metrics": result.metrics,
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
