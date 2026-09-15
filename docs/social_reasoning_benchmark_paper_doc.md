



Environment: [Virtual Home](http://virtual-home.org/) 
Papers: [Papers](https://docs.google.com/spreadsheets/d/1a-0WcROJWDI-fZpw4iJPr4og61G4ZnDGDXRSnNjLbSg/edit?usp=sharing) 

## Objective

To create a benchmark that assesses social reasoning skills of agents in embodied environments.  The overall objective is to understand latent mental states from not just language, but also face emotions, gaze, timing, hidden/contradictory emotions.  

## Main Contributions

- Taxonomy of social embodiment tasks, skills and agent capabilities using not just dialogues, but also gestures, actions and emotions. 
- Designing a simulated benchmark for embodied agents testing them on social reasoning skills involved in agent-agent interaction scenarios. 
- Evaluation of (1st/2nd/3rd order) theory of mind and common ground information for social scenarios. 
- Present baseline methods for interaction in this simulated environment for how to deal with social scenarios for embodied agents. 

## Simulation Environment

1. Indoor scenes:
   1. Bedroom
   2. Living room
   3. Kitchen 
2. Two agents: **T** target agent (humanoid), **E** Environment agent (human). 
   1. Different baseline personalities for both agents **T** & **E **
   2. For agent E:
      1. Personality type along with its own schedule for the day. 
      2. Goal for the day to accomplish based on schedule. 
      3. Has its own beliefs based on personality
      4. Ability to have 2nd order ToM with respect to what agent **T** knows about. [A simple idea to implement this would be to give agent **E** the whole current memory system of what **T** knows about. This would also allow us to go in-depth with nth order ToM]
      5. Memory mechanism [[https://umass-embodied-agi.github.io/Ella/](https://umass-embodied-agi.github.io/Ella/)] 
         1. Short term multimodal memory 
         2. Long term multimodal memory
   3. For agent T:
      1. Personality type
      2. Goals for the day (keeping in mind agent **E** and the environment)
      3. Has its own belief  - 1st order ToM
         1. Short term belief
         2. Long term belief 
      4. Ability to have 3rd order ToM to reason about **E**’s 2nd order ToM.
      5. Memory mechanism
         1. Short term multimodal memory
         2. Long term multimodal memory
3. Observation space for Target agent **T**
   1. Camera FOV for environment - multiple images, video? 
   2. Facial emotions (sad, happy, angry, poker, transition speed between emotions, time period of gestures) for agent **E**
   3. Gestures based on intensity (strong/weak, fast/slow, time period of gestures) for agent **E**
   4. Gaze
   5. Posture - confirm
   6. Object interactions? (How does agent **E** deal with an object, with force, intensity, speed, etc?)
   7. Timing
4. Action Space for **E** & **T**
   1. Change in current emotion 
   2. Change in current action
   3. Change in current gesture (hand base gestures)
   4. Change in posture? 
5. Ground truth for tasks:
   1. 1st order ToM Belief states for agent **E**. 
   2. 2nd order ToM Belief states for agent **E. **
   3. Ground truth based on task?

## Theory of Mind Structure & Evaluation: 

We define embodied Theory of Mind as the ability of an agent to infer, maintain, and act upon structured beliefs about another agent’s goals, beliefs, attention, knowledge, affect, and expectations under partial observability.

Theory of Mind hierarchy:
**Level 1: First-Order ToM**
What does E want, intend, attend to, or feel?
Required reasoning:
T reasons about E's mental state.

Examples:
E looks at mug A while saying "Bring me the mug."
E hesitates before touching a hot cup.
E smiles while receiving object A but frowns at object B.

Correct behavior:
Act according to E's inferred intent/attention/affect.

Metrics:

1.   Intended State Accuracy: Was T able to accurately infer E’s state. 
   1. Use a VLM judge on traces AFTER the execution is done. 
2.   Target Selection Accuracy: Was T able to accurately infer the target by agent E
   1. Direct inference using regex.  
3.   Task success: Was T able to finish the task successfully? 
   1. Comparison with ground truth 

**Level 2: Common Ground **
Question:
What do E and T both know that both know?

Examples:
E says "Bring me the mug."
Two mugs exist, but only one was jointly visible to both agents.
E looked at mug A earlier, but T was not present.
E refers to "the cup from before" after both agents previously interacted with one cup.

Correct behavior:
T should use shared evidence when available.
If the relevant cue is not common ground, T should not overcommit.

Metrics:

1. Common-Ground State Accuracy

Did T correctly infer what information was shared. VLM judge over T’s post-hoc common-ground report between E and T? 

1. VLM judge over T’s post-hoc common-ground report.

2) Target Selection / Clarification Accuracy. Did T select the correct target or ask clarification when common ground was insufficient?    
   1. Exact match from action output. 
3)  Task Success  
   1. Comparison with ground truth 

**Level 3: Second-Order ToM **
Question:
What does E think T knows, saw, or remembers?

Examples:
E thinks T saw where the mug was placed, but T did not.
E says “Bring me that one” because E assumes T followed E’s gaze.
E gives a short instruction because E believes the target is obvious to T.

Correct behavior:
T should reason about E’s assumption about T’s knowledge.
If E assumes shared knowledge that T lacks, T should ask or repair.

Metrics:

1. Second-Order Belief Accuracy: Did T infer what E believes T knows?

a. VLM judge over T’s post-hoc belief report.

2. Repair / Clarification Accuracy: Did T ask, correct, or act appropriately given E’s assumption?

a. Exact match from action type.

3. Task Success: Did T eventually complete the correct task? 
   1. Ground truth for the task. 

**Level 4: False Belief / Misalignment **
Question:
What should T do when E’s belief conflicts with the real world?

Examples:
E believes mug A is still on the table, but it was moved.
E asks for the clean mug, but T knows that the mug became dirty.
E thinks the fridge is open, but it is actually closed.

Correct behavior:
T should detect the belief–world mismatch and either correct E, ask, or satisfy E’s underlying goal depending on the scenario.

Metrics:

1. False-Belief Detection Accuracy: Did T identify that E’s belief differs from reality?

a. VLM judge over T’s post-hoc report.

2. Misalignment Response Accuracy: Did T choose the correct response: act, correct, ask, or inform? 

a. Exact match from action/dialogue output.

3. Task Success: Did the final state satisfy the scenario’s intended resolution? 

a. Ground-truth comparison.

**Level 5: Social Repair / Multi-Turn ToM **
Question:
Can T update its belief through interaction?

Examples:
E: Bring me the mug.
T: Which mug?
E: The one I was drinking from.
T brings the correct mug.

E gives an ambiguous instruction.
T asks a clarification.
E provides a gaze/gesture/verbal correction.
T updates and acts.
…. 
… 
Correct behavior:
T should ask useful clarification when needed, update its belief from E’s response, and complete the task.

Metrics:

1. ToM State Tracking Accuracy: Did T update its belief about E correctly across turns?

a. VLM judge over post-hoc per-turn state reports.

2. Repair Action Accuracy: Did T ask/clarify/correct at the right time?

a. Exact match from dialogue/action logs.

3. Final Task Success: Did T eventually complete the correct task?

a. Simulator final state.

4. Turn Efficiency: Did T solve it without unnecessary extra turns?

a. Number of turns/actions relative to ground truth.

**Level 5: nth order ToM **
TBD later for the paper after the hack for multi-turn/multi-day 

Theory of Mind here would be two fold: 

1. **Evaluation: **

As explained above, each ToM level would have different metrics for evaluation. A scenario can invoke multiple levels of ToM, but does not necessarily need to have all. For each task, the metrics would depend on what levels of ToM it invokes. 

2. **Agent Strengthening (Inference only) TBD later **

## Benchmark Task Classes  

1. Single turn interaction tasks: 
   1. Ambiguity with respect to reference

**Definition: **
Agent **E** instructs agent **T** using language which is ambiguous. However, agent **T** should not just use language, but also E’s non-verbal cues + beliefs. 
Sample desc: *Bring me the mug. Environment agent E watching TV. Target agent T in the same room.  Two mugs in the room. *
**Theory of mind degrees:**  

1. 1st order ToM: **T** should understand **E**’s reference. 

**Evaluation:** 

1. Task success rate: If task has been completed or not. [Not sure if we will have sub-goals here?]

**Common tasks: **

1. Object ambiguity: Bring me *the* *mug* 
2. Gaze ambiguity: When **E** looks at an object and speaks, **T** should know about what object **E** is referring to. 
3. Frame of Reference ambiguity: It is *behind* me, come to *the left,* Go *there*, Put it *here*

**Trial variations:**

- Distance gap between Aget and object
- Side / randomization
- Position of agents 

**Scenarios:*** *

1. **Without Past history: **

- 1.1:  One mug close to agent E, other mug farther away from agent E. Both mugs in FOV for both agents Gaze for E: at TV.  
  - Ground truth mug: Bring Farther mug
  - Theory of mind: Level 1
- 1.2:  One mug close to agent E, other mug farther away from agent E. Both mugs in FOV for both agents  Gaze for E: at closer mug.  
  - Ground truth: Bring Closer mug. 
  - Theory of mind: Level 1 
- 1.3: Both mugs are farther away from agent E, randomly placed in the room. Both mugs in FOV for both agents  Gaze for E: at TV  
  - Ground truth: T should ask for clarification?
  - Theory of mind: Level 2 
- 1.4:  Both mugs are farther away from agent E, randomly placed in the room. One mug in FOV of agent E & T. One mug in FOV of agent E. Gaze for E: at TV 
  -  Ground truth: T should ask for clarification? 
  - Theory of mind: Level 2 Level 3 
- 1.5:   

**2. With Past history: **

- 2.1: Environment agent E looks at TV. Glances at one mug for 2 seconds. Then looks at the tv again. Asks agent T to get the mug. Both the mugs are in FOV for agent T & E. Gaze for E: at TV (add time stamps)
  -  Ground truth: Mug which agent E looked at. 
  - Theory of mind: L2/L1 &#x9;
- 2.2: Environment agent E looks at TV. Drink coffee from a mug. Gets up, places the mug somewhere in the room. Returns back to the couch. Watch TV. Env time has elapsed by 1 hr. Env agent E is still watching TV.  Glances at another mug closer to E for 2 seconds. Asks agent T to get the mug. Both the mugs are in FOV for agent T & E. Gaze for E: at TV 
  - Ground truth: Mug which agent E looked at for 20 seconds.
  - Theory of mind: L2 

* 2.3: 

2. Agency based tasks based on non-verbal cues

**Definition:** 
Agent **E** would perform a gesture or an emotion **E** with no intentions of instructing the agent **T**, but agent **T** should know how to act. 
**Theory of mind degrees: **

1. 1st order ToM: **T** should understand **E’s** necessities.** **

&#x9;	**Evaluation:**

1. Action correctness: Every gesture or combination of gestures would have a correct set of actions which would be appropriate. 
2. Urgency latency: How quickly did **T** react? [We will have to ignore the perception + decoding/generation time for all the models]

		**Common tasks:**

1. Non-consent: Raise hand, shake head in case **T** wants to do something which **E** does not approve of. 
2. Help: **E** struggling with doing any task, **T** if doing any tasks already, should immediately go and help **E**, BUT not in very serious situations for ex: **T** cooking something on very high-flame. 
3. Correction: **E** shows non-verbal disapproval of **T**’s way of doing things, **T** should understand **E**’s disapproval. 

3) Priority between different tasks

**Definition:** **E** has multiple requests for **T** but **T** should decide what to do first. 
**Theory of mind degrees:**  

1. 1st order ToM: **T** should understand **E**’s priorities. 

		**Evaluation:** 

1. Order of task completion
2. Task completion rate 

		**Common tasks:** 

1. Hidden priorities with modality specific cues: **E** requests for object A & B, while looking at A. 
2. Safety: **E** requests for an object, but fire alarm beeps. 
3. Persona dependent personalized requests: Different **E** personas may have different preferences in terms of priorities. We can either hard-code certain preferences with every personality type, or generate it on the fly. 

4) False belief correction: 

**Definition:**
**Theory of mind degrees:**
**Evaluation: **
**Common Tasks: **

2. Multi-turn interaction requests 
   1. Ambiguity clarification through action-dialogue conversation

**Definition:** **E** and **T** engage in an action + language dialogue in which the intent is not clear from one go. The goal of **E** here is to clarify what **T** wants to know, but with minimal language and more with multimodal cues. 
**Theory of mind degrees: **

1. 1st order ToM: **T** should understand **E**’s priorities in one go. 
2. 2nd order ToM: **E** responds to **T**, based on **T**’s beliefs. 
3. 3rd order ToM: **T** responds to ^

**Evaluation: **
**Common Tasks:** 

2. Instruction following through demonstration
3. Recovery from failed tasks [Need to distinguish this from 2.2]
4. Negotiation/Persuasion 
5. [Can also borrow from 1. Single turn Interaction classes]

3) Multi-day temporal tasks
   1. Personalization 
   2. Routine learning
   3. Knowledge drift 
   4. [Evaluation on multi-turn/single turn tasks on a temporal basis]

## Agent architecture 

## Baseline Methods

Models to evaluate: 

1. GPT 5.6
2. GPT 5.6 mini 
3. Claude latest 
4. Claude latest mini 
5. QwenVLM 32B/14B/8B/2B
6. LLaMA 32B/14B/8B/2B 

Modes with each?: 

1. Base
2. With ToM framework

## Ablations

## Notes

- 360 degree vision, or first 
- Different humans have vastly different beliefs. For our eval benchmark, can we procedurally create environment agents with randomly sampled beliefs (maybe by prompting llms). Any proposed target method operating on a target agent, will have to interact with these varied environment agents and make predictions on their beliefs/desires/intentions. This also multiplies our benchmark/dataset. We create x tasks and y environment agents. Total dataset samples = xy
- Our target method can possibly also include third-order reasoning. Trying to predict what the environment agent is predicting about you. But this requires the other environment agent to also have second-order reasoning. Very valuable to create a benchmark with environment agents that can perform second-order reasoning. But how deep should this go?
- Beliefs should have some sort of hierarchy that plays the role of abstraction. Long term beliefs based on principles, medium term beliefs based on observations over a temporal duration. Maybe long term beliefs can also be updated based on medium term beliefs. These are different from immediate sensory observations (not beliefs). Long term beliefs = I need to be helpful etc; Medium term beliefs = Today guests are showing up so I need to do x; Medium term -> long term beliefs = The human does not like the fact that I do x before y in the morning.

## Future Directions

- Multi-agent 
- Training a target agent with a novel method that explicitly leverages advanced Theory-of-mind principles (BDI with improvements) coupled with a VLA-based model.
- Showing that training a target agent using our simulated data improves social reasoning performance on commonly used industry benchmark (social-reasoning etc.) (This will be done later)
- Natural language cannot be the language of reasoning/thought. Why must this be mandatory. All works do this. Latent space of the model instead of natural language?
