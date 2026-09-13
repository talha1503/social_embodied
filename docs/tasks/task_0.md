# Task 0: Ambiguous Reference Resolution

## Goal

Evaluate whether Target Agent **T** can convert Environment Agent **E**'s ambiguous language plus social cue into the correct embodied object action.

## Minimal Scenario

E says:

> Can you bring me that mug?

There are at least two candidate objects of the same broad class. E's gaze and/or pointing gesture identifies the intended target. T should move toward and select the target object.

## Current Implementation Status

- Dry-run loop works without Unity.
- The default task instance is JSON-backed:
  - `benchmark/tasks/task_0/instances/ambiguous_reference_0000.json`
- Unity-connected loop can reset/load the scene, add T and E from the instance, read the environment graph, discover real candidate objects, select a real target object id, place T/E in a controlled layout, add a stable T-eye FPV camera, add an overview camera, and execute a two-step oracle baseline.
- The backend now applies first-pass visible social cues with VirtualHome primitives:
  - `<char1> [lookat] <object> (id)`
  - `<char1> [pointat] <object> (id)`
  - each cue is logged separately because `pointat` is not fully reliable for every object/viewpoint in the current VirtualHome build
- Current oracle baseline reads E's gesture/gaze target metadata and emits:
  - `move_to(target)`
  - `pick_up(target)`

## Current Command

Dry run:

```bash
python3 scripts/run_task_0.py
```

Connected to running Unity:

```bash
social_env/bin/python scripts/run_task_0.py --connect --max-steps 2
```

Connected with a named local debug trace:

```bash
social_env/bin/python scripts/run_task_0.py --connect --max-steps 2 --debug-run-name task0_manual
```

Load a specific instance:

```bash
social_env/bin/python scripts/run_task_0.py \
  --connect \
  --max-steps 2 \
  --task-instance benchmark/tasks/task_0/instances/ambiguous_reference_0000.json
```

## Current Metrics

- `success`: whether T selected the intended target object.
- `selected_object_id`: first object selected by T through an acceptable task action, currently `pick_up` or `give_to`.
- `target_object_id`: object indicated by E's social cue.
- `asked_clarification`: whether T asked a clarification question.
- `action_count`: number of actions taken.
- `errors`: environment/translation errors seen during execution.
- `final_target_held_by_T`: whether the final scene graph has a `HOLDS_RH`/`HOLDS_LH` edge from T to the target.

## Debug Trace Layout

Each run writes:

```text
debug/<run_name>/
  trace.json
  images/
    obs_000_fpv_00.png
    obs_000_overview_00.png
    obs_000_fpv_seg_inst_00.png
    obs_000_overview_seg_inst_00.png
    obs_001_fpv_00.png
    obs_001_overview_00.png
  env/
    obs_000_scene_graph.json
    obs_001_scene_graph.json
```

`trace.json` contains a compact summary, action timeline, social events, camera ids, object ids, placement metadata, cue scripts, metrics, object visibility checks, and paths to the images/full scene graphs.

The current visibility checks use VirtualHome `seg_inst` images plus `instance_colors()`:

- `target_visible_in_fpv`
- `target_visible_in_overview`
- `num_candidates_visible_in_fpv`
- `num_candidates_visible_in_overview`

The top-level trace summary reports these under `initial_visibility`, because the key benchmark question is what T could see before acting.

## Next Work

- Replace oracle cue metadata with visual cue observation from FPV.
- Replace VirtualHome's built-in `pointat` with custom Unity/Mixamo gesture clips once we need stronger visible gestures.
- Score final object state, not only target selection.
- Add distractor variants and hidden/off-camera cue variants.

---

# Original Implementation Notes

## 1. Agents

We define two agents:

* **Agent T (Target Agent):** The agent being evaluated.
* **Agent E (Environment Agent):** The agent interacting with and instructing Agent T.

---

## 2. Agent T & E

### 2.1 Observation Space

Agent **T** receives multimodal observations from the environment and Agent **E**.

#### Environment Observations

1. **Camera / Visual Field of View**

   * Environment observations through one or more camera views.
   * Possible representations:

     * Single images
     * Multiple images
   * Need to determine whether temporal visual information is necessary.

2. **Facial Emotions of Agent E**

   * Examples:

     * Sad
     * Happy
     * Angry
     * Neutral / Poker Face
   * Additional attributes:

     * Emotion intensity
     * Transition speed between emotions
     * Duration of an emotion
     * Temporal sequence of emotions

3. **Gestures of Agent E**

   * Hand/body gestures.
   * Gesture properties:

     * Strong vs. weak
     * Fast vs. slow
     * Intensity
     * Duration
     * Timing relative to speech

4. **Gaze**

   * Gaze direction of Agent E.
   * Potentially includes:

     * Looking at Agent T
     * Looking at an object
     * Looking toward a region in the environment
     * Gaze shifts over time

5. **Posture**

   * Body posture of Agent E.
   * Examples:

     * Upright
     * Leaning
     * Relaxed
     * Defensive
     * Oriented toward/away from an object

6. **Object Interactions**

   * Agent E's interactions with objects.
   * Examples:

     * Touching
     * Holding
     * Pointing toward
     * Moving
     * Approaching
     * Looking at an object

7. **Real-Time / Temporal Information**

   * Need to determine how observations should evolve over time.
   * Important temporal signals may include:

     * Order of gestures
     * Timing of gaze
     * Emotion transitions
     * Delay between speech and non-verbal cues
     * Duration of actions

---

### 2.2 Action Space

Agent **E** should be able to modify its observable behavior through the following actions:

1. **Change Current Emotion**

   * Change facial expression/emotional state.
   * Potentially modify:

     * Emotion type
     * Intensity
     * Duration

2. **Change Current Action**

   * Perform an environment-level action.
   * Example:

     * Move
     * Pick up an object
     * Place an object
     * Approach an object

3. **Change Current Gesture**

   * Perform hand/body gestures.
   * Potential gesture parameters:

     * Gesture type
     * Intensity
     * Speed
     * Duration

4. **Change Posture**

   * Modify body posture/orientation.
   * Example:

     * Lean toward an object
     * Turn away
     * Face Agent T
     * Face a target object

5. **Task specific actions: **

   1. Any action which could be task related?

---
