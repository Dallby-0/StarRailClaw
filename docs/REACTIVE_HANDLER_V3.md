# Reactive Page Handler v3.1

`reactive_handler.v3.1` is the only supported page-handler schema. Existing
workspaces are intentionally not migrated; rebuild them after a schema change.

The handler is limited to `ui_2d` states. Registered tools remain a parallel
execution route and may handle either 2D or 3D scenes.

## Provider Contract

Each provider represents one semantic UI operation and contains:

- a stable `operation_key` used for conservative semantic merging;
- ordered `locators[]` used to resolve the action;
- soft `guards[]` used only for ranking;
- bounded `watches[]` from which runtime may learn guards;
- observable post-action `effects[]`;
- `successors[]` used as a short-chain prior.

All persisted coordinates use the 1000x1000 logical coordinate space.
Locators fall back in declaration order. A guard pass or failure changes rank;
it never hard-filters a provider. Active guards have strong weight and
provisional guards have weak weight.

Each operation declares `entry_providers[]`. On a fresh visit only an available
entry provider is considered; an isolated provider added by repair cannot
silently become the first step. If the declaration is absent internally,
runtime only infers roots that have successors.

`line_count` guards use a target and tolerance. Exact, near, and merely present
text produce progressively weaker positive evidence. Guards sharing a `group`
are equivalent variants and contribute the strongest result; different groups
accumulate evidence subject to a bounded total score.

## Appearance Watches

A watch identifies a region whose visual state distinguishes adjacent steps:

```json
{
  "id": "confirm_state",
  "rect": [650, 700, 980, 950],
  "modalities": ["appearance"],
  "after_provider": "confirm",
  "coordinate_space": "logical"
}
```

After a strongly confirmed action, runtime extracts the changed subregion
inside the watch. The before template becomes a provisional guard for the
current provider and the after template becomes one for `after_provider`. When
`after_provider` is empty and there is exactly one successor, that successor is
used. The two frames are mutual negative checks.

Runtime owns pixel differencing, template bounds, thresholds, evidence, and
promotion. The LLM only selects the semantic region, modality, and provider
relationship. If no discriminative subregion exists, no guard is created.

A learned guard becomes active only after it matches positive frames from two
different visits, has at least two mutual-negative checks, and has no false
positive. Evidence is bounded and stored with the guard.

## Effects And Progress

Effects are observable hypotheses (`pass`, `becomes_pass`, or `becomes_fail`).
They may confirm an action but do not alone define page identity. Pixel change
does not prove progress. Known state transitions and stable local text changes
remain separate progress evidence.

Only `confirmed`, `transitioned`, or successor-backfilled actions contribute
guard observations. Contradicted, unresolved, and action-error attempts never
train guards.

An unverified action that visibly changes a watched region is retained as a
visit-local pending transition. If one of its declared successors subsequently
succeeds, runtime backfills the predecessor as `confirmed_by_successor` and
learns the predecessor before/after guard pair. Raw frames never enter the
JSON runtime state.

## Semantic Regions And Merge

`click_region` describes a bounded area with a preferred point and at most four
historical candidate points. Runtime never clicks arbitrary random positions.
It may try at most two candidates per visit, and only retries when the previous
click caused no visible change. A changed screen stops point fallback.

Providers merge only when their `operation_key` values match, their click
regions overlap by at least 80% of the smaller region, and non-empty effects
and successors do not conflict. Candidate points and guard variants are
retained. The incoming definition is also placed in the operation's bounded
`provider_archive`, so a bad merge remains diagnosable and recoverable.

Every ranking decision logs the total score and its base, status, history,
successor, entry, and guard components.

## Runtime Limits And Repair

Per-visit soft and hard action limits and run-global hard limits prevent
unbounded execution. Repair is invoked only when no provider is available or
the soft action limit is reached. Repair may correct locators, watches,
successors, or guards, and may propose a family-scoped provider when multiple
instances show a real shared visual invariant.

Family candidates must retain dynamic locator validation. Learned guard
evidence is local to each candidate and is never globally degraded by one
failed visit.
