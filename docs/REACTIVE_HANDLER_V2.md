# Reactive Page Handler v2

`reactive_handler.v2` is the only supported page-handler schema. A state that
contains an older or unversioned handler is rejected; create a fresh state
workspace instead of migrating it.

The page handler is only one execution backend. The state bootstrap response
first chooses exactly one top-level route:

```text
execution
  reactive_2d    -> reactive page handler
  invoke_tool    -> registered tool, dispatched before the page handler
  cannot_handle  -> explicit stop
```

`scene_mode` describes the observed surface; it does not name an executor.
`reactive_2d` is valid only for `scene_mode=ui_2d`. Registered tools declare
their own supported scene modes and may target either 2D or 3D surfaces. Tool
calls own their internal loops and budgets and never consume reactive action
or repair quotas. A registered tool returns one of `progressed`, `completed`,
`no_progress`, `failed`, or `aborted`; control then returns to the outer visual
loop with a fresh screenshot.

## Model

An operation owns a flat provider library. There are no strategies, stateful
step runners, generic exploration profiles, continuation leases, or dynamic
capacity extensions.

```text
operation
  providers[]
    locators[]
    hints[]
    deferred_hints[]
    effect_hints[]
    successors[]
```

- A provider describes one semantic action.
- Locators are ordered fallback methods for finding that action target.
- Hints are soft visual ranking evidence. `pass > unknown/no hint > fail`.
- Effect hints are observable post-action hypotheses. They are optional.
- Providers not referenced as successors receive a chain bonus on a fresh
  visit. After an action runs, its successors receive the same bonus.
- Deferred hints are materialized from the stable frame after their named
  predecessor. They remain provisional until the target provider produces a
  confirmed effect.

The core schema contains no domain-specific page or object rules. Such rules
belong in separately registered presets or future plugins.

## Coordinates

Every persisted point and rectangle uses the 1000x1000 logical coordinate
space and must explicitly carry `coordinate_space: logical`. Template and OCR
results carry real coordinates internally. The action executor converts a
logical point exactly once and never reconverts a real point.

Supported locators:

- `point`
- `region_template`
- `text_target`

Supported visual probes:

- `template`
- `line_count` (detection only)
- `text` (detection and recognition; expensive)

OCR probes require a bounded region. The runner first evaluates cheap probes
for every provider and only evaluates OCR hints for a bounded number of leading
candidates.

## Selection

Provider ranking combines:

1. soft visual evidence;
2. base priority;
3. a chain bonus for fresh-visit roots or the previous action's successors;
4. activation status and small historical-success bonus;
5. visit-local attempt suppression.

A failure suppresses the provider only for the current visit. It does not
globally reduce guard reliability or disable a provider that may still be valid
for another page-family instance.

Pixel difference is diagnostic data only. It is not proof of progress and does
not reopen providers.

## Effects

An effect hint compares its probe before and after the action:

- `pass`: the post-action probe passes;
- `becomes_pass`: it did not pass before and passes afterward;
- `becomes_fail`: it did not fail before and fails afterward.

The result is `confirmed`, `contradicted`, or `unverified`. Missing effect hints
produce `unverified`, not failure. A transition to another known state is also
confirmed progress. Losing the current state match without another known match
is deferred to outer state resolution and is not counted as provider success.

## Repair and generalization

Repair first verifies that the current frame still belongs to the believed
state, then repairs the operation when it does. It receives:

- the current stable full-resolution frame;
- a contact sheet containing a stored family sample and recent action frames;
- a manifest connecting cells to providers and outcomes;
- provider definitions, failures, budgets, and deferred/provisional hints.

The model may request at most two contact-sheet cells at full resolution in one
additional query round. It then returns:

- a local patch for the current instance;
- zero or more separate family generalization candidates.

A high-confidence `state_misidentified` result returns control to outer state
resolution with the believed state temporarily excluded. The outer runtime,
not the repair model, creates a new state when necessary and strengthens the
excluded state's matcher with conditions verified against its positive samples
and the current screen. The newly selected state is left unchanged.

Equivalent repair patches stop the loop. Family candidates cannot be point-only
and must resolve using their dynamic locators on both the current frame and a
stored, visually distinct family sample. Accepted candidates start as `canary`.
They become `active` only after confirmed success on two distinct visits.

## Budgets

Defaults are intentionally fixed and small:

- repair after 8 actions without a confirmed effect;
- at most 16 actions per visit;
- at most 2 repairs per visit while stalled or uncertain; when the repair model
  confirms that a multi-step interaction is progressing, up to 8 repairs per
  visit are allowed within the existing action, time, and process hard limits;
- at most 180 seconds per visit;
- at most 120 reactive actions and 20 repairs per process run.

The soft limit triggers repair. Hard limits terminate the handler with a
structured log event and cannot be expanded by an LLM verdict.

Danger-page handling is outside this module. Runtime-owned, manually maintained
danger signatures should pause or terminate execution before provider ranking.
