# FRS-v015 One-Shot HRL Engineering Plan

Status: active, volatile engineering plan. Updated: 2026-07-22.

This plan applies the planning-compression rules from `one-shot-execution`.
Lifecycle stages, source owners, evidence tiers, tests, documentation refresh,
and an internal repair cycle are embedded checks inside an execution unit. They
are not separate user-visible steps.

## Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Contract registry: `../contracts/README.md`
- Active contracts: Method v015, Training v007, Gain v003, PPO v003, Eval v003
- Acceptance checklist: `../checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- Evidence ledger: `../../testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`

No method or Concept Figure change is pending. If implementation reveals a
real semantic contradiction, execution stops for a human decision instead of
silently changing these authorities.

## Terminal Outcome

Engineering is complete when the official Stage3-v015 path is connected,
obvious runtime bugs are removed, one bounded official training smoke reaches a
real grouped update and committed checkpoint, and its decisive diagnostics are
finite and non-degenerate. After that, remaining work is experiment execution
and paper evidence, not another engineering-audit chain.

## Completed Foundation

`G0--G4` and the supporting G5 implementation are retained as completed
evidence through `E-FI-61`. In particular, the repository already contains:

- immutable local scenarios and deployment-provenance q29 intent;
- `928D` combined observation, FEMR `158D` authority, and frozen-GMT `770D`
  authority;
- Repair/Noisy two-role reset and one FEMR action followed by K-step GMT
  evidence;
- v003 Gain, one policy row per attempt, grouped reduction, and exact-one
  transaction update;
- strict HSL-v1 actor-only initialization and Stage3-v015 checkpoint identity;
- committed save/fresh-reload and bounded diagnostic/evaluation connectors.

HSL is frozen as a validated auxiliary initializer. It is reopened only if
fresh official-run evidence identifies HSL as the first broken owner. The old
Q1--Q6 quality chain is preserved in the ledger as reusable evidence, but is no
longer a mandatory prerequisite to training.

## G5-E0: One-Shot HRL Engineering Closure

This is one authorized implementation-and-verification unit.

### Scope

1. Batch-inspect the official route:

   ```text
   Stage3 config
   -> explicit HSL-v1 actor initializer
   -> q29 928/158/770 observation authority
   -> sealed multi-Segment x M transaction
   -> one-action K evidence
   -> v003 Gain and return
   -> grouped PPO
   -> optimizer exact-one update
   -> committed Stage3-v015 checkpoint
   -> bounded diagnostics
   ```

2. Repair every obvious in-scope route, shape, identity, lifecycle, or logging
   defect found before or during the bounded smoke. One internal rerun after
   repair remains part of this unit.
3. Run only the focused compile/contracts needed to catch local regressions.
4. Run one bounded official Stage3 smoke and inspect its complete log.
5. Refresh the volatile plan/checklist/canvas and append final evidence.

### Embedded acceptance checks

- official v015 branch is reached without legacy resume/evaluator fallback;
- action has shape `[B,6]`, is finite, and is not collapsed by interface error;
- scenario/noisy hash, transaction identity, valid policy rows, and group mass
  remain present;
- `intent_gain`, `physics_gain`, `repair_cost`, `gain_total`, return, and
  advantage are finite and expose both sign and scale;
- gradients and optimizer counters prove exactly one update after the complete
  transaction, not during attempt collection;
- no Traceback, NaN/Inf, partial transaction, later FEMR action, or Clean actor
  input appears;
- a committed Stage3-v015 checkpoint path is emitted.

These are assertions inside G5-E0. Failure first triggers local diagnosis and
repair, not a new numbered planning step.

### Stop conditions

Pause only when one of these true boundaries is reached:

- a method/contract decision has multiple materially different answers;
- a destructive, costly, remote, or long-running action needs new authority;
- the official route remains contradictory after one complete diagnosis and
  repair cycle;
- the bounded run completes but learned metrics are no-op, regressing, or
  mutually contradictory, activating the conditional policy-quality branch.

### Execution result

Status: stopped at the declared policy-quality boundary, `E-FI-64`.

The official 8-env, 1-iteration, 1-update route reached strict HSL-v1
initialization, 928/158/770 authority, two sealed Segments x two attempts,
one-action K=8 evidence, grouped v003 PPO, exact-one Adam update, and a
committed Stage3-v015 checkpoint. The bounded telemetry connector was repaired
inside this unit to expose return, advantage, action scale, and pre/post-clip
gradient without changing Gain/PPO/HSL semantics.

Engineering connectivity passed, but closure did not: three of four attempts
had negative Gain, `harm_fraction=0.75`, every advantage was negative, and all
four `physics_gain` values were exactly zero. The gradient path was active
(`18/18` parameter tensors, pre-clip norm `611.1876`, post-clip `0.5`), so this
is not an optimizer/no-op wiring failure. It activates the smallest conditional
policy-quality audit before X1 or any additional training.

The conditional audit identified and repaired the first invalid evidence owner
at `E-FI-65`. Formal one-action-K collection now seals paired survival,
ZMP/support, height-contact consistency, and their common valid-step mask into
the immutable v015 carrier and consumes them through the unchanged v003 Gain.
Diagnostics now expose raw paired Physics components, policy value, return,
raw advantage, and row-aligned grouped-scaled advantage. Deterministic S1/S2
contracts pass; real simulator Physics values remain unconfirmed until one
bounded live sentinel is separately authorized.

Crossing files, owners, offline/live evidence tiers, or test types is not a
stop condition by itself.

## X1: Formal Experiments And Composition

This is the next true high-cost boundary after G5-E0 passes. It combines former
G6/G7 setup checks with the experiment that consumes them:

- choose the authorized training budget and seeds;
- run formal Stage3-v015 training and checkpoint trajectory collection;
- run paired frozen-GMT versus FEMR+GMT deployment composition;
- collect experiment tables, plots, and paper artifacts.

Offline connectivity and report checks are embedded in X1 setup. Long GPU
runs and externally consumed results require explicit authorization, so X1 is
not silently entered from G5-E0.

## Conditional Diagnostic Branches

- Use `formal-runtime-audit` only when the official route exposes an unresolved
  owner, shape, checkpoint, or runtime-connectivity contradiction.
- Use `policy-quality-audit` only after a runnable policy exhibits no-op,
  regression, harmful Gain, contradictory physics evidence, or unexpected
  checkpoint trajectories.
- Open only the smallest diagnostic owner implicated by the first abnormal
  fact. Do not restore Q1--Q6 as a mandatory sequence.

## Planning Deletion Test

The former implementation, integration, persistence, offline/live, checklist,
and evidence-refresh micro-steps do not require independent user decisions and
are merged into G5-E0. Former G6/G7 are merged into X1 because only the costly
experiment boundary requires new authorization. Removing any remaining unit
would either mix engineering with a costly experiment or conceal a genuine
method decision.

## Cursor

Current cursor: `G5-E0 ready for explicit execution authorization`.

This planning update does not execute G5-E0, tests, simulator, training, or a
live run.
