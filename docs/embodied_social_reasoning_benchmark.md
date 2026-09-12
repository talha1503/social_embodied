# Embodied Social Reasoning Benchmark

## Environment

- **Simulation Environment:** [VirtualHome](http://virtual-home.org/)
- **Papers:** Papers

---

## Objective

To create a benchmark that assesses **social reasoning skills of agents in embodied environments**.

The overall objective is to understand **latent mental states** from not just language, but also:

- Facial emotions
- Gaze
- Timing
- Hidden or contradictory emotions
- Gestures
- Actions
- Other multimodal social cues

---

## Main Contributions

1. **Taxonomy of social embodiment tasks, skills, and agent capabilities**
   - Goes beyond dialogue.
   - Includes gestures, actions, emotions, gaze, timing, and other non-verbal signals.

2. **A simulated benchmark for embodied social reasoning**
   - Tests agents on social reasoning skills involved in agent-agent interaction scenarios.

3. **Evaluation of Theory of Mind and common ground**
   - 1st-order ToM
   - 2nd-order ToM
   - 3rd-order ToM
   - Common-ground reasoning in social scenarios

4. **Baseline methods**
   - Present baseline methods for interaction in the simulated environment.
   - Study how embodied agents can deal with social scenarios.

---

# Simulation Environment

## Indoor Scenes

- Bedroom
- Living room
- Kitchen

## Agents

Two agents:

- **T — Target Agent**: humanoid agent being evaluated
- **E — Environment Agent**: simulated human/environment agent interacting with T

Both **T** and **E** can have different baseline personalities.

---

# Environment Agent E

## Properties

- Personality type
- Daily schedule
- Goal(s) for the day based on its schedule
- Own beliefs based on personality
- Ability to perform **2nd-order Theory of Mind** with respect to what agent T knows

### 2nd-Order ToM Idea

A simple implementation may be to give agent E access to the current memory/belief state representing what T knows.

This could also allow extension toward deeper **n-th-order Theory of Mind**.

## Memory Mechanism

Reference: [ELLA](https://umass-embodied-agi.github.io/Ella/)

- Short-term multimodal memory
- Long-term multimodal memory

---

# Target Agent T

## Properties

- Personality type
- Goals for the day
  - Goals should consider agent E and the environment.
- Own beliefs
- 1st-order Theory of Mind
- Short-term beliefs
- Long-term beliefs
- Ability to perform **3rd-order Theory of Mind**
  - Reason about E's 2nd-order beliefs about T.

## Memory Mechanism

- Short-term multimodal memory
- Long-term multimodal memory

---

# Observation Space for Target Agent T

Potential observations include:

## Vision

- Camera field-of-view of the environment
- Multiple images or video
- Potentially **360-degree vision**

## Facial Emotions of Agent E

Examples:

- Sad
- Happy
- Angry
- Poker / neutral expression

Additional dimensions:

- Transition speed between emotions
- Duration of facial expressions
- Hidden or contradictory emotions

## Gestures

Gestures may vary according to:

- Strength / intensity
- Fast vs. slow motion
- Duration
- Hand-based gestures

## Gaze

- Direction of gaze
- Duration of gaze
- Gaze shifts
- Joint-attention signals

## Posture

- Posture state
- Changes in posture

**Open question:** Confirm how much posture variation VirtualHome can support.

## Object Interactions

Potential social information from how E interacts with objects:

- Force
- Intensity
- Speed
- Hesitation
- Repeated attempts
- Carefulness / aggressiveness

## Timing

Timing-related social signals may include:

- Delay before actions
- Delay before responses
- Duration of gestures
- Temporal coordination between verbal and non-verbal cues

---

# Action Space for E and T

Possible actions include:

- Change current emotion
- Change current action
- Change current gesture
- Change posture
- Interact with objects
- Move through the environment
- Speak / respond through language

---

# Ground Truth

Potential ground-truth annotations include:

- 1st-order ToM belief states for agent E
- 2nd-order ToM belief states for agent E
- Task-specific latent goals
- Intentions
- Preferences
- Priorities
- Knowledge states
- Common-ground states

**Open question:** Determine what ground truth should be task-specific versus globally maintained as part of the agent state.

---

# Benchmark Task Classes

## 1. Single-Turn Interaction Tasks

### 1.1 Ambiguity with Respect to Reference

#### Definition

Agent E gives agent T an instruction using language that is ambiguous.

T should resolve the ambiguity not only through language, but also using:

- Non-verbal cues
- Beliefs
- Gaze
- Environment context
- Common ground

#### Theory of Mind Degree

- **1st-order ToM:** T should understand E's intended reference.

#### Evaluation

- **Task Success Rate**
  - Whether the intended task was completed successfully.

**Open question:** Should partial success or sub-goal completion also be evaluated?

#### Example Tasks

- **Object ambiguity:**  
  “Bring me the mug.”

- **Location ambiguity:**  
  “Go there.”  
  “Put it here.”

- **Frame-of-reference ambiguity:**  
  “It is behind me.”  
  “Come to the left.”

- **Gaze ambiguity:**  
  E looks at an object while speaking; T must infer the intended object.

- **Object-reference ambiguity:**  
  “The bottle next to the book.”

---

### 1.2 Agency-Based Tasks from Non-Verbal Cues

#### Definition

Agent E performs a gesture, facial expression, or other non-verbal behavior **without explicitly instructing T**.

T should infer what action, if any, is socially appropriate.

#### Theory of Mind Degree

- **1st-order ToM:** T should understand E's needs, preferences, or intentions.

#### Evaluation

- **Action correctness**
  - Each gesture or multimodal cue combination may correspond to a set of acceptable actions.

- **Urgency latency**
  - How quickly does T react?

> For fair evaluation, perception, decoding, and generation latency of individual models may need to be normalized or excluded.

#### Example Tasks

##### Non-Consent

E:

- Raises a hand
- Shakes their head
- Shows disapproval

T should understand that E does not approve of the intended action.

##### Help

E struggles with a task.

T should infer whether assistance is needed.

However, T should also reason about its current commitments.

Example:

- T is cooking something on very high heat.
- E appears to need help.
- T should determine whether immediately abandoning the cooking task creates a safety risk.

##### Correction

E displays non-verbal disapproval of how T is performing a task.

T should infer that its current approach is undesirable and modify its behavior.

---

### 1.3 Priority Between Different Tasks

#### Definition

Agent E has multiple requests for T, but T must infer which request should be handled first.

#### Theory of Mind Degree

- **1st-order ToM:** T should understand E's priorities.

#### Evaluation

- Order of task completion
- Task completion rate
- Priority accuracy

#### Example Tasks

##### Hidden Priorities from Modality-Specific Cues

E requests objects A and B but:

- Looks at A
- Gestures toward A
- Shows urgency toward A

T should infer that A is higher priority.

##### Safety

E requests an object, but a fire alarm starts beeping.

T should override or postpone the request in favor of the higher-priority safety event.

##### Persona-Dependent Personalized Requests

Different E personas may have different priority preferences.

Possible implementations:

1. Hard-code preferences for personality types.
2. Procedurally generate preferences.
3. Learn preferences over repeated interactions.

---

### 1.4 False-Belief Correction

#### Definition

T recognizes that E holds an incorrect belief about the world and must decide:

- Whether correction is necessary
- How to correct E
- When to correct E
- Whether E's false belief matters for the current task

#### Theory of Mind Degree

Potentially:

- 1st-order ToM
- 2nd-order ToM

#### Evaluation

To be defined.

#### Example Tasks

To be defined.

---

# 2. Multi-Turn Interaction Tasks

## 2.1 Ambiguity Clarification Through Action-Dialogue Conversation

### Definition

E and T engage in a multimodal interaction involving:

- Language
- Action
- Gesture
- Gaze
- Environment interaction

The intent is not fully clear from a single observation.

The goal is for clarification to happen with **minimal explicit language** and greater use of multimodal cues.

### Theory of Mind Degrees

- **1st-order ToM:**  
  T reasons about E's beliefs, intentions, and priorities.

- **2nd-order ToM:**  
  E responds to T based on E's beliefs about what T knows or believes.

- **3rd-order ToM:**  
  T reasons about E's beliefs regarding T's beliefs.

### Evaluation

To be defined.

Potential metrics:

- Task completion
- Number of clarification turns
- Amount of explicit language required
- Belief-state accuracy
- Efficiency of multimodal grounding
- Recovery from incorrect assumptions

---

## 2.2 Instruction Following Through Demonstration

E demonstrates an action or sequence instead of explicitly describing it.

T must infer:

- The task
- Relevant objects
- Important steps
- Whether exact imitation or goal-level imitation is intended

---

## 2.3 Recovery from Failed Tasks

T attempts a task and fails.

The benchmark should test whether T can:

- Recognize failure
- Infer why it failed
- Interpret E's reaction
- Update its beliefs
- Retry appropriately

**Open question:** Clearly distinguish this class from ambiguity clarification.

---

## 2.4 Negotiation / Persuasion

T and E may have:

- Different goals
- Different preferences
- Conflicting priorities
- Incomplete knowledge about each other

The interaction may test:

- Negotiation
- Persuasion
- Compromise
- Preference inference
- Social adaptation
- Multi-turn belief updates

Tasks from the **single-turn interaction classes** may also be extended into multi-turn variants.

---

# 3. Multi-Day Temporal Tasks

## 3.1 Personalization

T should learn persistent preferences of E across repeated interactions.

Examples:

- Preferred morning routine
- Preferred object placement
- Preferred sequence of household tasks
- Interaction style preferences

---

## 3.2 Routine Learning

T should infer repeated behavioral patterns from E.

Example:

> E typically prepares coffee immediately after waking up.

T may eventually anticipate this routine.

---

## 3.3 Knowledge Drift

E's:

- Preferences
- Beliefs
- Goals
- Routines
- Environment knowledge

may change over time.

T must detect and adapt to these changes.

### Evaluation

Single-turn and multi-turn tasks can be evaluated repeatedly over temporal horizons to measure:

- Adaptation
- Forgetting
- Belief updating
- Personalization
- Robustness to preference changes

---

# Agent Architecture

To be developed.

Potential components:

1. Multimodal perception
2. Episodic memory
3. Semantic / long-term memory
4. Belief-state tracking
5. BDI-style reasoning
6. Theory-of-Mind module
7. Planning module
8. VLA policy / action model
9. Social reward or preference model

---

# Baseline Methods

To be developed.

Potential categories:

- Language-only baseline
- Vision-language baseline
- VLA baseline
- Explicit belief-state baseline
- Memory-augmented baseline
- BDI-based agent
- ToM-augmented agent
- Full multimodal social reasoning agent

---

# Ablations

Potential ablations:

- No gaze
- No facial emotion
- No gesture
- No timing information
- No personality
- No memory
- No long-term memory
- No 2nd-order ToM
- No 3rd-order ToM
- Language only
- Vision only
- No belief hierarchy
- No procedural persona variation

---

# Notes and Research Questions

## 1. 360-Degree Vision

Consider whether the target agent should receive a full 360-degree visual observation.

Questions:

- Is this realistic?
- Does it remove interesting embodied perception challenges?
- Should T instead need to actively look around?

---

## 2. Procedurally Generated Human Beliefs

Different humans have vastly different beliefs.

For the evaluation benchmark, environment agents could be procedurally created with randomly sampled:

- Beliefs
- Desires
- Intentions
- Preferences
- Personality traits
- Routines

One possibility is to prompt LLMs to generate these latent states.

A target agent would then need to interact with many different environment agents and infer their:

- Beliefs
- Desires
- Intentions
- Preferences

If there are:

- `x` tasks
- `y` environment agents

then the benchmark can potentially create approximately:

\[
x \times y
\]

task-agent combinations.

This allows the benchmark to scale without manually writing every interaction.

---

## 3. Higher-Order Theory of Mind

The target method could include **3rd-order reasoning**.

Example:

> T predicts what E predicts about T.

However, this requires E to possess a meaningful model of T's beliefs.

A benchmark with environment agents capable of **2nd-order reasoning** would therefore enable evaluation of **3rd-order reasoning** in T.

### Open Question

How deep should this hierarchy go?

Possible choices:

- 1st-order only
- Up to 2nd-order
- Up to 3rd-order
- Dynamically selected depth
- Arbitrary n-th order in synthetic settings

A key concern is whether deeper reasoning corresponds to meaningful social behavior or simply increases benchmark complexity.

---

## 4. Hierarchical Beliefs

Beliefs may be represented hierarchically by timescale and abstraction.

### Immediate Sensory Observations

These are **not necessarily beliefs**.

Examples:

- “E is looking at the cup.”
- “The stove is on.”
- “E has raised their hand.”

### Short-Term / Medium-Term Beliefs

Derived from recent observations over a temporal window.

Examples:

- “Guests are arriving today, so the living room needs to be cleaned first.”
- “E appears unusually rushed this morning.”

### Long-Term Beliefs

Higher-level persistent knowledge or principles.

Examples:

- “I should be helpful.”
- “E dislikes it when I perform X before Y in the morning.”
- “E typically prefers tea over coffee.”

### Belief Updates

Medium-term patterns may update long-term beliefs.

Example:

> Repeated observation: E consistently becomes annoyed when T performs X before Y in the morning.

This may eventually become:

> Long-term belief: E prefers Y to be done before X.

---

# Future Directions

## Multi-Agent Extension

Extend beyond a single T-E pair to environments containing:

- Multiple humans
- Multiple target agents
- Groups with conflicting beliefs
- Shared and private knowledge
- Coalitions
- Group norms

---

## Novel Social Reasoning Agent

Train a target agent with a novel method that explicitly leverages advanced Theory-of-Mind principles.

Potential direction:

- Improved **BDI (Belief-Desire-Intention)** representation
- Coupled with a **Vision-Language-Action (VLA)** model
- Explicit multimodal belief tracking
- Hierarchical memory
- Higher-order ToM

---

## Transfer to Existing Benchmarks

A later stage could test whether training on the simulated benchmark improves performance on commonly used external social-reasoning benchmarks.

Goal:

> Show that simulation-based training improves general social reasoning rather than only benchmark-specific task performance.

---

## Latent Reasoning Instead of Natural-Language Thought

Most current approaches represent explicit reasoning or intermediate mental states using natural language.

This raises an important question:

> Why must natural language be the representation used for reasoning?

Potential research direction:

- Represent beliefs, intentions, and ToM states directly in a model's latent space.
- Perform social reasoning without explicit natural-language chain-of-thought.
- Compare:
  - Natural-language reasoning
  - Structured symbolic belief representations
  - Continuous latent belief representations
  - Hybrid approaches

Potential advantages:

- More compact representations
- Faster reasoning
- Less dependence on verbalizability
- Better representation of ambiguous or multimodal social states

---

# Open Questions

- How should false-belief correction tasks be formalized?
- What should count as ground truth for each task class?
- How deep should ToM reasoning go?
- How should belief states be represented?
- Should personalities and preferences be hard-coded or procedurally generated?
- How much variation is needed across environment agents?
- How should timing and reaction latency be evaluated fairly across different model architectures?
- How should partial task success be scored?
- How much language should be allowed in multimodal clarification tasks?
- What should distinguish immediate observations from beliefs?
- Should 360-degree perception be given directly or require active exploration?
- Can higher-order ToM be evaluated without introducing artificial reasoning complexity?
- Can social reasoning be represented directly in latent space rather than natural language?
