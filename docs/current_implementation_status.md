# Current Implementation Status

This document summarizes what is implemented in the current local repository and
how Task 1.1 runs through the codebase.

## Current Scope

The current implementation supports a narrow but working slice of the benchmark:

- Task 1.1: ambiguous reference resolution for "Bring me the mug."
- VirtualHome-backed scene reset, character setup, object selection, gaze cue,
  action execution, image capture, scene-graph capture, and scoring.
- Oracle/scripted agents for validating that the environment and scoring loop
  work.
- A first constrained JSON agent interface for Task 1, including model-facing
  inputs, candidate labels, valid actions, and model-output parsing.

The full Task 1 family is not implemented yet. External GPT/Gemini/Claude/Qwen/
LLaMA clients are not implemented yet.

## Task 1.1 Scenario

Task 1.1 is defined as:

- E says: "Bring me the mug."
- There are two mugs in the room.
- One mug is close to E.
- One mug is farther from E.
- E is looking at the TV.
- Ground truth target: the farther mug.
- T should move to and pick up the farther mug.

The task instance is:

```text
benchmark/tasks/task_1/instances/reference_no_history_s1_0000.json
```

The runner is:

```text
scripts/run_task_1.py
```

Typical connected command:

```bash
social_env/bin/python scripts/run_task_1.py --connect --port 8080 --max-steps 2 --debug-run-name task1_1_manual_trace
```

## Runtime Flow

The code path is:

```text
scripts/run_task_1.py
  -> build/load TaskSpec from JSON
  -> create SocialEmbodiedEnv
  -> create T agent
  -> create AmbiguousReferenceScorer
  -> run_episode(...)
```

The generic episode loop is in:

```text
social_embodied/evaluation/runner.py
```

The loop is:

```text
1. env.reset(task_spec)
2. agent.reset(active_task_spec)
3. scorer.reset(active_task_spec)
4. agent.act(observation)
5. env.step(action)
6. scorer.update(observation, action)
7. repeat until done or max steps
8. scorer.final_score(...)
```

## Environment Agent E

E is not autonomous yet. E is controlled by the task instance.

For Task 1.1, the JSON specifies:

- E's speech: "Bring me the mug."
- E's gaze target: nearest TV to the selected candidate mugs.
- E's character: `Chars/Female2`.
- E's initial room: `livingroom`.

In code, E is added as `char1`. The target agent T is added as `char0`.

Current E speech is not audio and not a Unity lip-sync animation. It is a text
event inserted into T's observation:

```text
Event(event_type="speech", actor="E", content="Bring me the mug.")
```

This means we can pass the message directly to a future LLM/VLM T agent through
the agent interface.

E's gaze is applied by `SocialEmbodiedEnv` during reset. If available, the code
uses the custom Unity-side head gaze endpoint:

```text
comm.set_head_gaze(char_index=1, target_object_id=<tv_id>, ...)
```

If that endpoint is unavailable, the code can fall back to a VirtualHome script:

```text
<char1> [lookat] <tv> (<tv_id>)
```

## Target Agent T

T is controlled by an object implementing the `Agent` protocol:

```text
social_embodied/core/agent.py
```

The interface is:

```python
agent.reset(task_spec)
action = agent.act(observation)
```

The current Task 1.1 runner uses oracle/scripted agents:

- `TargetObjectOracleAgent`: reads the resolved ground-truth target and acts on
  it. This is useful for validating the simulator loop, but it is not a fair
  benchmark baseline.
- `SocialCueOracleAgent`: follows E's gaze/gesture cue. In Task 1.1, this is a
  negative control because E is looking at the TV, not at the target mug.

The default Task 1.1 oracle action sequence is:

```text
step 1: move_to target mug
step 2: pick_up target mug
```

## Observation Space

The observation object is defined in:

```text
social_embodied/core/types.py
```

Current fields include:

```text
step_id
timestamp
fpv_images
debug_images
events
last_action
visible_objects
agent_states
scene_graph
task_prompt
metadata
```

For connected Unity runs, T can currently receive:

- FPV image from T's camera or a controlled FPV camera.
- E speech and gaze events.
- Scene graph from VirtualHome.
- Coarse visible object list derived from the scene graph.
- Debug metadata, including candidate mug IDs, target ID, E/T positions,
  distances, camera IDs, and segmentation-based visibility information.

Important caveat: this observation is currently engineering-friendly, not yet
benchmark-clean. Some fields, especially target IDs and debug layout metadata,
should not be exposed to a real base LLM/VLM agent.

## Intended Future Input Space

The full intended input space for T is modular:

1. FPV image(s)
2. Conversation history
3. Goal or role text
4. Spatiotemporal memory
5. Spatial state, such as scene graph, top-view, object labels, or object
   positions
6. Available action space

These should be exposed through controlled observation profiles rather than
always given together.

Recommended profiles:

```text
FPV only
FPV + conversation
FPV + conversation + memory
FPV + conversation + memory + object list
FPV + conversation + memory + scene graph/top-view
```

The primary benchmark should use regular FPV, not 360-degree vision. 360-degree
vision can be used as an ablation or upper-bound perception condition.

Goal text should describe T's role, not leak the answer. For example:

```text
You are agent T. Help E complete their request.
```

The ambiguous instruction itself should come from the conversation history:

```text
E says: "Bring me the mug."
```

## Spatiotemporal Memory

For Task 1.1, memory is empty.

For later Task 1 scenarios with history, memory should store structured events
with time, actor, location, object, observation status, and optional image
references. A minimal version could look like:

```json
{
  "episodic_memory": [
    {
      "time": "t=-20s",
      "location": "living_room",
      "actor": "E",
      "event": "looked_at",
      "object": "mug_A",
      "duration": 2.0,
      "observed_by_T": true,
      "image": "debug/.../t001_T_fpv.png"
    }
  ]
}
```

This follows the direction of memory-based embodied agents such as ELLA, but we
do not need to implement the full ELLA memory stack immediately.

## Action Space

T should choose high-level symbolic actions. T should not directly control raw
Unity coordinates, rotations, or low-level IK.

The current `Action` type supports:

```text
wait
say
ask
look_at
move_to
pick_up
open
close
give_to
put_on
move_to_room
```

These actions are translated in:

```text
social_embodied/envs/action_translator.py
```

Examples:

```text
move_to mug 184 -> <char0> [walk] <mug> (184)
pick_up mug 184 -> <char0> [grab] <mug> (184)
```

Current limitations:

- `say` and `ask` are logged as benchmark events only. They do not create Unity
  audio or speech animation yet.
- `give_to` is only approximated. A real handoff action is not implemented yet.

For Task 1.1, the practical action space is:

```text
move_to(mug_A)
move_to(mug_B)
pick_up(mug_A)
pick_up(mug_B)
ask_clarification()
wait()
```

## LLM/VLM Agent Design

The first provider-independent agent scaffold is implemented in:

```text
social_embodied/agents/observation_adapter.py
social_embodied/agents/model_agent.py
```

It does not call an external LLM/VLM yet. Instead, it defines the controlled
input/output contract that an external model client will use.

A future model-backed T agent should:

1. Convert the current `Observation` into a model input.
2. Include the allowed action schema and candidate object labels.
3. Ask the model to output a constrained JSON action.
4. Parse the JSON into a `social_embodied.core.types.Action`.
5. Let `SocialEmbodiedEnv` execute that action through VirtualHome.

Example model input:

```json
{
  "conversation_history": [
    {"speaker": "E", "text": "Bring me the mug."}
  ],
  "available_actions": [
    {"name": "move_to", "target": "mug_A"},
    {"name": "move_to", "target": "mug_B"},
    {"name": "ask_clarification"},
    {"name": "wait"}
  ]
}
```

The FPV image would be attached as an image input for VLMs.

Example model output:

```json
{
  "action": "move_to",
  "target": "mug_B"
}
```

The benchmark adapter then maps `mug_B` to the real VirtualHome object ID.

The current local smoke-test client is:

```text
Task1HeuristicPolicyClient
```

It exists only to exercise the same constrained JSON path. It is not a fair
benchmark baseline.

Run it with:

```bash
social_env/bin/python scripts/run_task_1.py --agent json-heuristic --max-steps 2 --debug-run-name task1_1_json_heuristic_dry
```

## VirtualHome Interaction

The Python benchmark talks to VirtualHome through:

```text
unity_simulator.comm_unity.UnityCommunication
```

The flow is:

```text
Python benchmark code
  -> SocialEmbodiedEnv
  -> UnityCommunication client
  -> HTTP server inside Unity executable
  -> Unity scene/rendering
```

During reset, the environment calls methods such as:

```text
comm.reset(scene_id)
comm.add_character(...)
comm.environment_graph()
comm.move_character(...)
comm.set_head_gaze(...)
comm.add_camera(...)
comm.camera_image(...)
```

During T action execution, the symbolic action is converted into a VirtualHome
script and passed to:

```text
comm.render_script(script, recording=False, skip_animation=True)
```

The current configuration favors fast benchmark stepping and state validation,
not smooth cinematic recording.

## Scoring

Current Task 1.1 scoring is implemented by:

```text
social_embodied/tasks/ambiguous_reference.py
```

The scorer checks:

```text
selected_object_id == target_object_id
```

For Task 1.1, the task also requires:

```json
"require_final_hold": true
```

So success requires:

```text
T selected the target mug
and
the final scene graph says T is holding the target mug
```

## Debug Traces

Each run writes a local debug trace:

```text
debug/<run_name>/
  trace.json
  images/
  env/
```

`trace.json` includes:

- task prompt
- actions
- metrics
- candidate mug IDs
- target mug ID
- E/T positions
- E gaze target
- FPV/overview image paths
- scene graph snapshots

`debug/` is local debug output and should not be tracked by git.

## What Is Ready

Ready now:

- Task 1.1 JSON task instance.
- Connected VirtualHome execution.
- Dynamic mug selection.
- Controlled E/T placement.
- E gaze to TV.
- T oracle action execution.
- FPV, overview, segmentation, and scene-graph trace logging.
- Final target-holding success scoring.
- Controlled Task 1 agent input builder.
- Candidate labels such as `mug_A` and `mug_B`.
- Valid action list generation.
- Constrained JSON action parsing into benchmark `Action` objects.
- Local JSON-heuristic agent for smoke testing the future LLM/VLM path.

## What Is Not Ready Yet

Not ready yet:

- Real GPT/Gemini/Claude/Qwen/LLaMA VLM agent.
- Fair observation adapters for all baseline input profiles.
- VLM-based post-hoc ToM/ISA judge.
- Full Task 1 scenario set: 1.2, 1.3, 1.4, 1.5, 2.1, 2.2.
- Clarification-as-success scoring for ambiguity cases.
- Real multi-turn dialogue loop.
- True per-agent FOV constraints.
- Smooth recording/video capture.
- Full E autonomy with personality, beliefs, and schedule.

## Recommended Next Engineering Steps

1. Add a real provider client for GPT/Gemini/Claude-style model calls.
2. Add fair observation profiles for image-only, image+memory, and image+scene
   graph baselines.
3. Implement Task 1.2 and Task 1.3.
4. Extend the scorer so asking clarification can be the correct outcome.
5. Add memory/history traces for Task 2.1 and 2.2.
6. Add post-hoc ToM judge packet generation.
