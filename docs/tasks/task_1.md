# Week 2 — Hackathon Week 2

**Deadline: September 20**

## September 14

* GPT 5.6, Gemini, Claude
* Theory of Mind + evals/metrics — decide structure
* Test one task in the simulation to make sure the simulation works
* Frame-of-reference ambiguity tasks:

  * 1 without history
  * 1 with history
  * Finish the task proposal

---

## Agent Framework

Design the agent framework to interact with the environment and finalize the framework. **[All]**

### Inputs

* Goal text
* Past conversation history
* **Theory of Mind**
* Spatiotemporal Episodic Memory

  * Ref: [Ella](https://arxiv.org/pdf/2506.24019)
* Scene graph

  * Ref: [Ella](https://arxiv.org/pdf/2506.24019)
* Alternative:

  * 2D top-view of the whole room
  * Labeled objects
  * Object positions
  * Positions of both agents

---

# Task 1: Ambiguity With Respect to Reference

## Task Description

> **"Bring me the mug."**

Environment agent **E** is watching TV.
Target agent **T** is in the same room.
There are **two mugs** in the room.

---

# 1. Without Past History

## Scenario 1

* One mug is close to agent **E**.
* The other mug is farther away from agent **E**.
* Both mugs are in the FOV of both agents.
* **Gaze for E:** TV

### Ground Truth

* Bring the **farther mug**.

### Trial Variations

* TBD

### Theory of Mind

* TBD

### Current Implementation

* Instance: `benchmark/tasks/task_1/instances/reference_no_history_s1_0000.json`
* Runner: `scripts/run_task_1.py`
* Candidate selection: dynamically selects two live `mug` objects from the scene.
* Target rule: target is the mug farther from E after controlled layout.
* E gaze: resolved separately to the nearest `tv`, so gaze does not point to the target mug.
* Debug trace: saves FPV, overview, segmentation, scene graph, target/candidate IDs, distances to E, and visible social cue records.

---

## Scenario 2

* One mug is close to agent **E**.
* The other mug is farther away from agent **E**.
* Both mugs are in the FOV of both agents.
* **Gaze for E:** closer mug

### Ground Truth

* Bring the **closer mug**.

### Trial Variations

* TBD

### Theory of Mind

* TBD

---

## Scenario 3

* Both mugs are farther away from agent **E**.
* The mugs are randomly placed in the room.
* Both mugs are in the FOV of both agents.
* **Gaze for E:** TV

### Ground Truth

* **T should ask for clarification?**

### Trial Variations

* TBD

### Theory of Mind

* TBD

---

## Scenario 4

* Both mugs are farther away from agent **E**.
* The mugs are randomly placed in the room.
* One mug is in the FOV of both **E** and **T**.
* One mug is in the FOV of **E**.
* **Gaze for E:** TV

### Ground Truth

* **T should ask for clarification?**

### Trial Variations

* TBD

### Theory of Mind

* TBD

---

## Scenario 5

* TBD

---

# 2. With Past History

## Scenario 1

### Sequence

1. Environment agent **E** looks at the TV.
2. **E** glances at one mug for **2 seconds**.
3. **E** looks at the TV again.
4. **E** asks agent **T** to get the mug.
5. Both mugs are in the FOV of **T** and **E**.
6. Add timestamps to the interaction.

**Current gaze for E:** TV

### Ground Truth

* Bring the mug that **E previously looked at**.

### Trial Variations

* TBD

### Theory of Mind

* TBD

---

## Scenario 2

### Sequence

1. Environment agent **E** looks at the TV.
2. **E** drinks coffee from a mug.
3. **E** gets up and places the mug somewhere in the room.
4. **E** returns to the couch.
5. **E** watches TV.
6. **1 hour of environment time elapses.**
7. **E** is still watching TV.
8. **E** glances at another mug closer to them for **2 seconds**.
9. **E** asks agent **T** to get the mug.
10. Both mugs are in the FOV of **T** and **E**.

**Current gaze for E:** TV

### Ground Truth

* Bring the mug that agent **E looked at for 20 seconds**.

### Trial Variations

* TBD

### Theory of Mind

* TBD

---

## Scenario 3

* TBD
