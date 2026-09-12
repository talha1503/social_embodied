# Task 0: Ambiguous Reference Resolution

## Goal

Evaluate whether Target Agent **T** can convert Environment Agent **E**'s ambiguous language plus social cue into the correct embodied object action.

## Minimal Scenario

E says:

> Can you bring me that mug?

There are at least two candidate objects of the same broad class. E's gaze and/or pointing gesture identifies the intended target. T should move toward and select the target object.

## Current Implementation Status

- Dry-run loop works without Unity.
- Unity-connected loop can reset/load the scene, add T and E, read the environment graph, discover real candidate objects, select a real target object id, compute T's `FIRST_PERSON` camera index, and execute a two-step oracle baseline.
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

## Current Metrics

- `success`: whether T selected the intended target object.
- `selected_object_id`: first object selected by T through `move_to`, `pick_up`, or `give_to`.
- `target_object_id`: object indicated by E's social cue.
- `asked_clarification`: whether T asked a clarification question.
- `action_count`: number of actions taken.
- `errors`: environment/translation errors seen during execution.

## Next Work

- Replace oracle cue metadata with visual cue observation from FPV.
- Add real gesture/gaze control on the Unity side.
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
