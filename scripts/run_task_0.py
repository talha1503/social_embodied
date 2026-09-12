"""Run the Task 0 dry-run benchmark loop."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from social_embodied.agents import ScriptedAgent
from social_embodied.core.types import Action
from social_embodied.envs import SocialEmbodiedEnv, VirtualHomeConfig
from social_embodied.evaluation import run_episode
from social_embodied.tasks import AmbiguousReferenceScorer, build_task_0_spec


def main() -> None:
    task_spec = build_task_0_spec()
    target = task_spec.objects["target"]

    agent = ScriptedAgent(
        [
            Action(
                action_type="pick_up",
                target_object_id=target["id"],
                params={"object_class": target["class_name"]},
            )
        ]
    )
    env = SocialEmbodiedEnv(VirtualHomeConfig(connect=False))
    scorer = AmbiguousReferenceScorer()

    result = run_episode(env=env, agent=agent, task_spec=task_spec, scorer=scorer)
    print(json.dumps(result.metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
