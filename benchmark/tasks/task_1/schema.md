# Task 1 Instance Schema

Task 1 extends the Task 0 ambiguous-reference loop with explicit social
reference rules.

## Scenario 1.1

`reference_no_history_s1_0000.json` encodes the first no-history scenario:

- E says "Bring me the mug."
- E is watching TV.
- Two candidate mugs are selected from the live scene graph.
- E is placed near candidate mug 0.
- The target mug is computed as the candidate farthest from E.
- E's visible gaze target is resolved separately to the nearest TV, not to the target mug.

The current implementation reuses `task_family: "ambiguous_reference"` so it
can run through the same environment, action translator, scorer, and debug trace
writer as Task 0.

Relevant instance fields:

- `object_selection.mode: "auto_spread_pair"` picks a pair of same-class mugs.
- `object_selection.target_relation: "farthest_from_environment_agent"` makes
  the target relation-driven instead of hard-coded.
- `layout.environment_agent_anchor` places E relative to one candidate object.
- `layout.controlled_fpv_camera.position_offset` can move the camera in front
  of T's body so the character mesh does not occlude the first-person image.
- `e_behavior.gaze.target` can be a dynamic object selector, such as the nearest
  `tv` to the selected candidates.
