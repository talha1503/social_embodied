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

The default Task 0 instance lives at:

```text
benchmark/tasks/task_0/instances/ambiguous_reference_0000.json
```

Run a specific instance:

```bash
python3 scripts/run_task_0.py --task-instance benchmark/tasks/task_0/instances/ambiguous_reference_0000.json
```

To run the same loop against a running VirtualHome Unity app:

```bash
social_env/bin/python scripts/launch_virtualhome_macos.py --fresh
social_env/bin/python scripts/run_task_0.py --connect --max-steps 2 --debug-run-name task0_manual
```

Run Task 1.1, the no-history "Bring me the mug" scenario where E watches TV
and the target is the mug farther from E:

```bash
social_env/bin/python scripts/run_task_1.py --connect --port 8080 --max-steps 2 --debug-run-name task1_1_manual
```

The naive cue-following baseline can be run as a negative control:

```bash
social_env/bin/python scripts/run_task_1.py --connect --port 8080 --agent social-cue-oracle --max-steps 2 --debug-run-name task1_1_social_cue_baseline
```

Connected runs write local debug traces by default:

```text
debug/<run_name>/
  trace.json
  images/
    obs_000_fpv_00.png
    obs_000_overview_00.png
    obs_000_fpv_seg_inst_00.png
    obs_000_overview_seg_inst_00.png
  env/
    obs_000_scene_graph.json
```

`trace.json` includes the action timeline, simulator/camera metadata, visible cue scripts, final graph checks, and segmentation-based target/candidate visibility summaries. `debug/` is ignored by git.

The current macOS v2.2.4 executable can return `False` from `reset(...)` even while the scene/API continue working. The adapter records this as `reset_warning` instead of failing by default. Repeated connected runs may accumulate character cameras in that build; restart the Unity app for clean manual experiments.
