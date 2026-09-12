# Implementation

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

