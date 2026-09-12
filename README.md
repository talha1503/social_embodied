# Social Embodied

Research workspace for social embodied reasoning experiments built around VirtualHome and VirtualHome Unity.

## Repository Layout

- `docs/`: project notes, benchmark design, and timeline.
- `virtualhome/`: Git submodule pointing to the local VirtualHome fork/branch.
- `virtualhome_unity/`: Git submodule pointing to the local VirtualHome Unity fork/branch.

## Clone

```bash
git clone --recurse-submodules https://github.com/talha1503/social_embodied.git
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Submodule Branches

Both submodules are currently pinned to the `social-embodied-setup` branch in the `talha1503` forks.

## Benchmark Loop

The top-level package defines the benchmark-facing agent/environment contract:

- `Observation`: FPV images, social events, visible objects, optional scene graph, and task prompt.
- `Action`: symbolic actions such as `look_at`, `move_to`, `pick_up`, `ask`, and `wait`.
- `SocialEmbodiedEnv`: VirtualHome adapter with a dry-run mode for early task development.
- `run_episode`: common reset/observe/act/step/score loop.

Run the first dry-run task:

```bash
python scripts/run_task_0.py
```

Expected output is a JSON metrics object with `success: true` for the scripted baseline.

To run the same loop against a running VirtualHome Unity app:

```bash
/usr/bin/open -n virtualhome/virtualhome/simulation/unity_simulator/macos_exec.2.2.4.app --args -screen-fullscreen 0 -screen-quality 4
social_env/bin/python scripts/run_task_0.py --connect --max-steps 2
```

The current macOS v2.2.4 executable can return `False` from `reset(...)` even while the scene/API continue working. The adapter records this as `reset_warning` instead of failing by default. Repeated connected runs may accumulate character cameras in that build; restart the Unity app for clean manual experiments.
