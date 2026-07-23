# FRS-GAIN-v005 Vector Physics-Constrained Intent Proposal

Status: accepted and activated by P1; retained as the E-FI-72/P0 decision
record. Updated: 2026-07-23.

## Design Delta

Old v004 semantics:

```text
Contact / ZMP / survival
-> normalized per-role violations
-> max Physics deficit
-> admissible/unsafe scalar utility tier
-> paired scalar Gain
-> one scalar return / advantage
```

The bounded C4 and ZMP plateau evidence shows that distinct K-step physical
states can collide at the same saturated deficit and utility. A Repair can
improve raw ZMP while receiving exactly the same Physics utility as Noisy.
Longer Critic calibration cannot reconstruct distinctions already deleted by
the target.

Confirmed v005 concept:

```text
Noisy rollout = zero-action counterfactual baseline
Repair rollout = consequence of one policy-sampled Delta SE(3)
Contact / phase-ZMP / survival = separate K-step Physics constraints
Intent improvement - repair cost = scalar optimization objective
Physics constraints = allowed actor-update directions
```

FEMR therefore does not trade Intent against Physics inside one handcrafted
reward score. It seeks the smallest Intent-faithful repair inside the
executable space defined by Contact, ZMP, and survival.

## Confirmed Noisy-Rollout Authority

Noisy rollout remains mandatory during Stage-3 training and evaluation. It
answers the causal question: what happens under the same sealed scenario when
FEMR applies no repair?

Noisy and every Repair attempt share `x_t`, current root artifact, deployment
q29 intent, Clean GMT-only continuation, K, frozen GMT, scenario hash, and
valid-step clock. Noisy remains useful for:

- paired Intent improvement;
- no-op and repair-necessity evidence;
- scenario-difficulty control and diagnostics;
- detecting whether FEMR caused improvement or harm.

Noisy is not the Physics safety threshold. `Repair better than Noisy` does not
make Repair physically admissible when both induce unplanned contact, sustained
lean, invalid phase-ZMP, or insufficient survival. Deployment still executes
one FEMR action followed by frozen GMT; it does not run an online Noisy branch.

## Physics Evidence Contract

The evaluator retains detached, immutable, ordered evidence until the
optimization boundary:

```text
expected support mode [B,K,2]
actual left/right ContactSensor state [B,K,2]
phase-conditioned signed ZMP margin/recovery evidence [B,K,...]
survival/terminal evidence [B,K]
semantic validity and ZMP-N/A masks [B,K,...]
```

Expected support remains deterministically derived from the same sealed Clean
continuation used only by GMT execution and the Physics evaluator. None of this
evidence enters FEMR observations, HSL targets, Intent targets, or deployment
inputs.

The following v004 reductions are retired from the proposed training signal:

- `[0,1]` saturation of violation severity;
- temporal/channel `max` or `amax` as the policy target;
- unsafe/admissible disjoint scalar utility intervals;
- a weighted or tiered scalar sum in which Intent and Physics share one Gain.

A semantic N/A mask remains valid only where ZMP does not exist, such as flight.
It may not hide a difficult or adverse valid step. A later constraint statistic
may reduce time only after its physical sufficient-statistic claim is explicit;
it must not saturate severe states or erase different required repair directions.

## Objective And Critic Boundary

The proposed scalar objective is only:

```text
intent_objective = paired Intent improvement - full-6D repair cost
```

The existing single scalar Critic predicts this objective at the active global
K stage. It does not predict a v004 Physics utility or a moving combination of
Intent and Physics. Contact, ZMP, and survival reach the actor update as
separate constraint evidence. This proposal adds no second actor, rho, second
Critic, cost-predictor network, contact predictor, new actor input, or second
optimizer.

## Activated Optimization Realization

The smallest mechanism that directly matches the concept is a grouped
first-order constrained policy update:

1. keep the existing grouped scalar Intent PPO surrogate and scalar value loss;
2. construct separate grouped Contact, ZMP, and survival constraint surrogates
   from the same policy rows and immutable K-step evidence;
3. compute the actor direction closest to the Intent direction that satisfies
   the local Physics constraints in constraint-gradient space;
4. apply that actor direction together with the ordinary scalar-Critic gradient
   in the existing exact-one optimizer step.

This is recommended over a fixed weighted sum, another clipped utility, or a
dual/cost network because the operation itself expresses the concept: Physics
removes destructive update directions; Intent chooses among the remaining
directions. The exact projection/recovery rule, infeasible-constraint behavior,
and checkpoint identity are frozen by FRS-PPO-v004 and FRS-TRAIN-v010. Source
implementation remains a separate P2 boundary.

## Preserved Boundaries

- one full-6D `Delta SE(3)` actor and one action at `t`;
- one scalar Critic with the existing global K curriculum;
- H as deployment/Noisy q29 future Intent;
- K as executable-evidence horizon;
- sealed multi-Segment x M transaction and equal grouped mass;
- exactly one optimizer step after the transaction seals;
- proposal-only HSL frozen;
- Clean `x_t` as dynamics reset only;
- Clean continuation as GMT/Physics-evaluator evidence only.

## Forbidden Shortcuts

- removing Noisy rollout because Physics becomes absolute;
- treating relative Repair-vs-Noisy Physics improvement as admissibility;
- feeding Contact, ZMP, support phase, Clean continuation, or constraint labels
  to the actor;
- silently relabeling v004 `gain_total` as the new objective;
- using a hard sample mask to discard physically adverse policy rows;
- compressing all constraint channels back into one scalar Physics score;
- changing grouped mass, M, H, K, one-action-K, HSL, or frozen GMT;
- starting long training while v004 remains the formal return/PPO identity.

## Contract-Version Consequence

P1 activates coordinated new versions because the method objective,
optimization rule, Critic target, formal consumer identity, and persistence
boundary all change:

- `FRS-METHOD-v016`: Noisy counterfactual plus Physics-constrained Intent repair;
- `FRS-GAIN-v005`: scalar Intent objective plus vector Physics evidence;
- `FRS-PPO-v004`: grouped constrained actor update with unchanged equal mass;
- `FRS-TRAIN-v010`: one scalar Intent Critic, v009 K curriculum preserved, new
  objective/constraint/checkpoint identity.

P1 has activated these identities. v004/v003/v009 remain the last implemented
route but are method-incompatible with the active contracts. Training and live
execution remain blocked until P2 replaces the source path and proves strict
legacy rejection.

## P1 Activation Record

The accepted mechanism is grouped first-order constraint-gradient projection.
FRS-GAIN-v005 fixes physical-unit residuals and the scalar Intent target;
FRS-PPO-v004 fixes joint projection, corrective recovery, and fail-closed
infeasibility; FRS-TRAIN-v010 fixes Critic fresh-entry/per-K recalibration and
checkpoint-v5 identity. E-FI-73 records the document-only activation.
