# Task 0 Instance Schema

Task 0 instances are JSON files loaded into `TaskSpec`.

Required top-level fields:

- `task_id`: stable instance id.
- `task_family`: currently `ambiguous_reference`.
- `scene_id`: VirtualHome scene id.
- `prompt`: instruction visible to the target agent.
- `agents`: T/E character prefabs and initial rooms.
- `object_selection`: how candidate objects are resolved from the scene graph.
- `layout`: controlled placement and camera settings.
- `e_behavior`: speech, gaze, and gesture cue plan.
- `success`: expected target/action conditions.

Implemented object selection modes:

- `fixed_ids`: resolves `candidate_ids` from the live scene graph and uses `target_id` or `target_index` as the target.
- `auto_nearby_pair`: finds two nearby objects from `candidate_classes`, optionally requiring the same class, and falls back to `fallback_classes` if needed. The selected object at `target_index` becomes the target.

The first implemented layout mode is `controlled_pair`. It places T and E at offsets from the candidate-pair center, adds a controlled FPV camera for T, and adds an overview camera for debugging.

Debug traces capture both normal RGB frames and `seg_inst` frames. The backend uses `instance_colors()` to estimate whether the target/candidate objects are visible in FPV and overview cameras.

Visible social cues currently use VirtualHome script primitives:

- gaze: `<char1> [lookat] <object> (id)`
- gesture: `<char1> [pointat] <object> (id)`

Each cue is executed and logged separately under `layout.visible_social_cues.records`, because the built-in `pointat` primitive can be less reliable than `lookat` for small objects. This is the backend-facing hook. Later Unity-side work can replace the `pointat` primitive with a custom Mixamo gesture controller while keeping the JSON task format stable.
