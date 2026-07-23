---
contract_id: FRS-METHOD-v016
status: active
effective_date: 2026-07-23
updated_date: 2026-07-23
supersedes: FRS-METHOD-v015
scope: FrontRES Stage 3 local root-artifact repair with one scalar paired Intent objective, independent Contact/phase-ZMP/survival actor constraints, grouped first-order constraint-gradient projection, one-action K evidence, and sealed multi-attempt Segment Replay
---

# Physics-Constrained Intent Segment Replay

## Design Delta

FRS-METHOD-v015 correctly established the deployable q29 future-intent actor,
one action at `t`, frozen-GMT K-step evidence, and sealed multi-Segment x M
transaction. Its v004 reward subcontract then collapsed Contact, ZMP, survival,
Intent, and repair cost into one scalar target. E-FI-72 shows that this deletes
different corrective directions before the Critic and actor update.

FRS-METHOD-v016 preserves the experiment and changes only optimization
authority:

```text
paired Intent improvement - full-6D repair cost -> scalar objective / Critic
Repair Contact evidence                         -> actor constraint
Repair phase-ZMP evidence                       -> actor constraint
Repair survival evidence                        -> actor constraint
```

Physics is not a reward term. It defines the locally permitted actor-update
directions. Intent chooses the closest permitted direction.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `M-02` | Sealed Local Scenario |
| `FRS-DP-02` | Segment Replay | `SR-01` | Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | One-Action K Evidence |
| `FRS-DP-04` | FrontRES 6D Repair | `M-04` | Actor Authority |
| `FRS-DP-05` | Frozen GMT | `M-10` | Frozen Execution Boundary |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Noisy Counterfactual And Repair Evidence |
| `FRS-DP-07` | Repair Gain | `Q-01` | Physics-Constrained Intent Update |
| `FRS-DP-10` | Future Motion Context | `M-11` | Deployable Future Intent |

No Contact, constraint, or solver block is added to the Concept Figure. They
are the detailed semantics of `Q-PAIR/Q-01`.

## Preserved Local Scenario And Actor Interface

One scenario seals:

```text
x_t identity
current root artifact_t
deployment/Noisy q29 intent[t:t+H]
Clean GMT-only continuation[t+1:t+K]
expected support evidence derived from that continuation
K, valid-step clock, scenario_id, noisy_segment_hash
```

`x_t` restores dynamics only. Clean continuation remains available only to
frozen GMT and the Physics evaluator. The actor still reads exactly the v015
deployable 158D prefix and emits one full-6D `Delta SE(3)` action at `t`.
Contact, ZMP, survival, expected support, Clean continuation, noise labels,
perturbation time, and future root/global state never enter the actor input.

## Noisy Counterfactual And Repair Evidence

Each scenario has one shared Noisy zero-action rollout and `M >= 2` Repair
attempts sampled from one frozen `pi_old`. All share the same scenario and
dynamics identity. Noisy answers the causal questions “was repair necessary?”
and “did Repair improve Intent relative to doing nothing?”. It is not the
Physics threshold and is never a PPO row.

Every Repair attempt retains ordered, immutable K-step evidence:

```text
expected/actual Contact and alignment facts
phase-conditioned signed ZMP margin and recovery facts
survival/terminal facts
valid-step and semantic N/A masks
```

Missing required evidence invalidates the transaction. Flight-only ZMP is a
semantic `N/A`, not a numeric zero.

## Scalar Objective And Independent Constraints

The single scalar training target is:

```text
y_I = IntentQuality(Repair, deployment q29)
    - IntentQuality(Noisy, deployment q29)
    - full6D_repair_cost
```

The scalar Critic predicts only `y_I`. Absolute Repair Contact, phase-ZMP, and
survival residuals remain separate and reach only the actor constraint
surrogates. Intent cannot waive a Physics violation, and Noisy being worse
cannot make Repair admissible.

FRS-GAIN-v005 owns the objective and residuals. FRS-PPO-v004 owns their grouped
first-order use. No scalar Physics score, admissible/unsafe utility, dual reward,
or cost-Critic is active.

## Physics-Constrained Intent Update

For the complete grouped transaction, let `p_I` be the scalar-Intent actor
ascent direction and `g_C`, `g_Z`, `g_S` the gradients of the independent
constraint surrogates in the direction of increasing violation. The actor
direction is the Euclidean projection of `p_I` onto their joint first-order
non-worsening cone. The three constraints are solved together, never in an
order-dependent sequence and never by a weighted sum.

If the projected Intent direction collapses, FRS-PPO-v004 attempts a common
constraint-recovery direction. If no finite common first-order descent exists,
the transaction fails closed for actor/std: their gradients are exactly zero,
while the scalar Critic may still update in the same exact-one optimizer step.
Unprojected Intent gradients may never be restored as a fallback.

## Frozen-Policy Transaction And Equal Mass

The v015 transaction semantics remain:

1. freeze one `pi_old`;
2. select multiple scenarios;
3. collect one Noisy baseline and all M Repair attempts per scenario;
4. perform no optimizer step during collection;
5. keep one policy row per Repair attempt regardless of K;
6. reduce motion -> Segment -> attempt with equal mass;
7. solve one grouped constraint projection;
8. execute exactly one optimizer step after the transaction seals.

Search, Noisy, Clean, oracle, or failed evidence rows never become policy rows.
Replay priority is selection-only. It may retain named per-constraint frontier
facts, but may not multiply actor loss, change grouped mass, or reconstruct a
scalar Physics reward.

## Training And Persistence Boundary

FRS-TRAIN-v010 owns Critic initialization and K-stage scheduling. The first
entry into the v016/v005 target uses the strict HSL-v1 actor initializer and a
fresh scalar Critic/optimizer target identity. Old v004 Stage-3 checkpoints are
not migration sources. At every later global K increase, the same v010 Critic
continues but re-enters critic-only recalibration while actor/std freeze.

Full resume requires the coordinated identities:

```text
FRS-METHOD-v016
FRS-GAIN-v005
FRS-PPO-v004
FRS-TRAIN-v010
frontres-v015-checkpoint-v5
```

## Forbidden Behavior

- scalarizing Physics back into the Critic target or actor advantage;
- subtracting Noisy Physics to define Repair feasibility;
- hard-masking adverse valid policy rows;
- sequential/order-dependent projection of Contact, ZMP, and survival;
- a second actor, Critic, optimizer, rho, constraint predictor, or dual network;
- Contact/Clean/constraint leakage to actor input or HSL target;
- later FEMR actions inside K, mixed K inside a transaction, or unequal group mass;
- v004/v003/v009/checkpoint-v4 fallback or compatibility padding.

## Acceptance And Stop Conditions

P2 must prove raw evidence preservation, distinct constraint surrogates, grouped
projection/recovery, actor/Critic gradient separation, exact-one update, strict
v5 persistence, and legacy rejection through deterministic S1/S2/S3 evidence.

Stop before source implementation or training if a constraint requires hidden
labels or actor-visible Clean data, the joint projection cannot be defined
without scalar Physics weights or a second learned network, infeasible recovery
falls back to Intent, missing evidence is zero-filled, or one transaction needs
more than one optimizer step.

## Owned Subcontracts

- Objective and constraints: `../reward/FRS-GAIN-v005-vector-physics-constraints.md`.
- Grouped actor update: `../optimization/FRS-PPO-v004-grouped-constraint-gradient-projection.md`.
- Critic/K schedule: `../training/FRS-TRAIN-v010-intent-critic-k-curriculum.md`.
- Evaluation remains `../evaluation/FRS-EVAL-v003-local-repair-composition-evaluation.md` until separately versioned.
