# FRS-v015 Physics-Constrained Intent Migration Engineering Plan

Status: active, volatile engineering plan. Updated: 2026-07-23.

## Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Active contracts: FRS-METHOD-v016 / FRS-GAIN-v005 / FRS-PPO-v004 /
 FRS-TRAIN-v010
- Current incompatible source route: FRS-METHOD-v015 / FRS-GAIN-v004 /
 FRS-PPO-v003 / FRS-TRAIN-v009
- P0 decision record:
  `FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Checklist: `../checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- Evidence: `../../testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`

## Terminal Outcome

Preserve the Noisy zero-action counterfactual and all existing v015 replay
semantics, while replacing v004 scalar Physics/Intent utility with:

```text
scalar paired Intent objective - repair cost -> one scalar Critic
Contact / phase-ZMP / survival K-step evidence -> actor constraints
```

The final engineering closure must prove that Physics determines admissible
actor-update directions, Intent improves only inside that space, and one sealed
multi-Segment x M transaction still produces exactly one optimizer update and
one compatible checkpoint.

## Preserved Foundation

E-FI-0--E-FI-71 remain valid evidence for local scenarios, deployment q29 H,
928/158/770 observation authority, two roles, one action, frozen-GMT K evidence,
multi-Segment x M atomicity, grouped equal mass, proposal-only HSL, K-stage
Critic curriculum, exact-one update, and v004 runtime connectivity. They do not
prove that the v004 scalar target preserves Physics direction information.

The current source remains the last runnable v004/v003/v009 route. It is now
an explicit contract-mismatch for further training because `clamp`/`amax` and
unsafe-tier scalarization erased distinctions visible in raw ZMP evidence.

## Source Of Truth And Migration Surface

| Semantic object | Proposed owner | Current incompatible owner | Required isolation |
| --- | --- | --- | --- |
| Noisy baseline | existing sealed scenario / two-role collector | same owner | retain exactly one shared zero-action baseline; never score it as a PPO row |
| K-step Physics evidence | command + live probe + storage | evidence exists but is reduced for v004 | retain ordered Contact/phase-ZMP/survival and semantic masks until loss |
| scalar objective | `frontres_gain.py` v005 owner | v004 tiered `gain_total` | only paired Intent improvement minus repair cost reaches scalar Critic |
| constrained actor update | `frontres_segment_ppo.py` v004 owner | scalar-advantage PPO v003 | separate Physics constraint surrogates; no scalar Physics reward fallback |
| exact-one transaction | formal update owner | v009/v003 path | same sealed rows and grouped mass; one projected actor + scalar-Critic step |
| replay priority | existing sampler evidence owner | v004 scalar Gain projection | selection-only evidence; cannot multiply actor loss or recreate scalar Physics reward |
| persistence | `frontres_checkpointing.py` | checkpoint-v4 binds v004/v003/v009 | new identity rejects old objective/constraint state before mutation |
| diagnostics/evaluation | existing diagnostic owners | v004 deficit/utility summaries | raw per-channel/time constraints, projection facts, objective/value/advantage |

## Step Map

### P0 (Preparatory): Design Rebase And Owner Audit

Objective: record the confirmed Noisy/Physics/Intent authority split and expose
the remaining constrained-update decision.

Scope: v005 proposal, Concept Figure Q-PAIR/Q-01 wording, registry mismatch,
current Architecture, plan, checklist, canvas, evidence ledger, and read-only
CodeGraph owner/consumer audit.

Non-scope: active-contract supersession, source code, tests, checkpoint I/O,
simulator, training, or live run.

Evidence: E-FI-72 plus CodeGraph-confirmed route
`frontres_gain -> storage return -> grouped PPO -> formal update -> checkpoint`.

Stop: the user has not confirmed the exact constrained-update mechanism.

Why separate: selecting projection/recovery semantics changes the optimization
method and checkpoint identity; this is a real human scientific decision.

Status: completed in this document-only rebase.

### P1 / 4: Contract Activation And Constraint Mathematics

Objective: freeze one executable constrained-update rule before code changes.

Scope: define physical-unit Contact/ZMP/survival residuals, time aggregation
without saturation, grouped constraint surrogates, the constraint-gradient
projection/recovery rule, scalar Intent-Critic target, infeasible-case behavior,
selection-only replay-priority authority, and coordinated METHOD-v016 /
GAIN-v005 / PPO-v004 / TRAIN-v010 identities.
Activate contracts, registry, Concept Figure mapping, and current Architecture.

Non-scope: source code, test execution, checkpoint I/O, simulator, training,
live run, new actor input/output, second actor/Critic/optimizer, or HSL changes.

Expected evidence: formula fixtures showing that different severe ZMP/contact
states remain distinguishable; safe-state collisions are semantically
irrelevant; Noisy cannot waive an absolute Repair constraint; the scalar Critic
target contains no Physics mixture.

Stop: projection requires a second learned network, a hand-weighted scalar
Physics score, a hard adverse-row mask, an undefined infeasible fallback, or a
new actor-visible privileged signal.
Status: completed document-only. E-FI-73 activates the four coordinated
contracts, archives their superseded predecessors, and leaves source/runtime
deliberately blocked for P2.

### P2 / 4: One-Shot Offline Engineering Closure

Status: completed offline at E-FI-74. The active source route now implements
the v005 scalar/vector authority split, PPO-v004 grouped joint projection,
TRAIN-v010 gradient phases, and checkpoint-v5 strict identity. This is not
simulator or policy-quality evidence.

Objective: implement the complete new contract through the formal update and
persistence boundary in one local engineering unit.

Scope: existing evidence/result schemas, `frontres_gain.py`, storage/candidate
carrier, grouped PPO constrained-loss owner, formal exact-one update,
diagnostics/evaluation, checkpoint save/reload identity, legacy isolation, and
all focused deterministic S1/S2/S3 contracts. Reuse existing modules; do not
create a parallel training stack.
Unique owners: `frontres_gain.py` owns scalar target and physical residuals;
`frontres_segment_ppo.py` owns grouped actor/constraint gradients and their
joint projection; the existing formal transaction/update owner owns exact-one
commit; `frontres_checkpointing.py` owns checkpoint-v5 validation before any
state mutation; the existing training schedule owner implements v010 fresh
Critic entry and per-global-K recalibration.

Non-scope: simulator, real training, live run, long horizon experiment,
multi-seed, deployment composition, HSL, actor observation, GMT, H/K/M, or
group-mass changes.

Embedded checks:

- S1: raw K-step identity, phase/N-A semantics, no saturation collision,
  constraint projection/recovery, permutation and missing-evidence fail-closed;
- S2: one complete multi-Segment x M transaction carries the same Noisy
  baseline, scalar objective, vector constraints, equal group mass, and exactly
  one optimizer step; v004/v003 formal fallback rejects;
- S3: new checkpoint binds all contract/layout/curriculum/constraint identities,
 committed receipt and solver identity; no learned or persistent dual state is
 introduced; old v004/v009, tampered, or partial resume rejects before mutation;
- curriculum: first v010 entry fresh-initializes Critic/critic normalizer and
 optimizer while actor/std remain frozen; every global K increase preserves the
 same v010 Critic, re-enters critic-only recalibration, then actor ramp and joint;
- regression: 928/158/770, q29 provenance, one-action-K, K curriculum, HSL
  isolation, frozen GMT, and actor/Critic gradient authority remain unchanged.

Engineering acceptance: deterministic evidence proves the scalar Critic learns
only the Intent objective and actor gradients are transformed only by the named
Physics constraints, with no hidden scalar Physics utility.

Stop: any constraint evidence is silently filled, saturated, reduced before its
semantic owner, mixed across scenarios/roles/K, fed to the actor, or requires
more than one committed optimizer step.

### P3 / 4: Bounded Official Physics-Constraint Sentinel

Objective: prove the new route on one real 8-env sealed transaction.

Scope: one explicit Stage3-v015 engineering run at one K stage, one iteration,
one complete transaction, one optimizer update, one committed new-identity
checkpoint, log review, evidence/checklist/Architecture closeout.

Required telemetry: scenario/Noisy hashes, raw Contact/phase-ZMP/survival by K,
semantic masks, scalar Intent objective/return/value/advantage, each constraint
residual and gradient, projected actor direction, actor/Critic parameter deltas,
group mass, exact-one counts, and save identity.

Non-scope: long training, K transition, multiple seeds, deployment composition,
paper experiment, or method tuning.

Stop: missing raw evidence, v004/v003 fallback, constraint-gradient absence,
actor update that worsens every violated Physics channel, scalar-Critic Physics
contamination, nonfinite projection, or update count other than one.

Why separate: this is the simulator/material-cost and external-runtime boundary.

### P4 / 4: Policy-Quality Admission

Objective: decide whether the new target/update is informative enough to admit
actor-ramp training.

Scope: compare the bounded sentinel's Noisy/Repair causal evidence, constraint
movement, scalar objective/value calibration, action non-collapse, and demo-
quality hacking sentinels; then define the smallest justified training horizon.

Non-scope: automatic long training, multi-seed, deployment composition, or
paper claims.

Expected evidence: the new route distinguishes raw improvements hidden by v004,
does not reward sustained lean/unplanned stepping, and provides a nonzero
Physics correction direction when Repair is inadmissible.

Stop: no feasible/corrective direction, no-op collapse, systematic Physics
regression, Intent-only shortcut, or evidence that the chosen constrained
optimizer does not realize the confirmed concept. Such a result returns to P1,
not to coefficient tuning.

## Why Four Main Steps

The plan has four execution steps plus preparatory P0. The splits are not by
file or test tier:

- P1 is the completed scientific/optimization contract decision;
- P2 is the largest safe local implementation and verification closure;
- P3 requires simulator/material-cost authority;
- P4 is the policy-quality and longer-training authorization boundary.

Removing any one would either hide a human semantic decision, cross a costly
runtime boundary, or admit training without causal evidence. All owner edits,
focused tests, formal connectivity, persistence, and routine document refresh
are deliberately merged into P2.

## Cursor

Current cursor: `P2 / 4 complete at E-FI-74; P3 / 4 requires separate simulator/material-cost authorization`.
