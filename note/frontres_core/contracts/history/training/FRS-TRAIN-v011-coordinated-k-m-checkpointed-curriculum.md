---
contract_id: FRS-TRAIN-v011
status: superseded
effective_date: 2026-07-28
updated_date: 2026-08-01
superseded_by: FRS-TRAIN-v012
supersedes: FRS-TRAIN-v010
scope: immutable coordinated global K x exact-M curriculum, fresh scalar Intent-Critic initialization, per-stage critic-only recalibration, checkpointed quality blocks, and checkpoint-v6 persistence
---

# Coordinated K x M Checkpointed Curriculum

## Design Delta

FRS-TRAIN-v010 established the v006 scalar Intent-Critic target, homogeneous
global K stages, repeated Critic recalibration and strict checkpoint-v5
persistence. Its formal transaction width still inherited state-driven sampler
trial counts with only `minimum_policy_attempts=2`. E-FI-86 therefore completed
1999 K8 transactions with exactly two Segments and M=2 throughout. Increasing
environment count alone could select more Segments instead of increasing
attempts for the same Segment.

FRS-TRAIN-v011 preserves the actor, Critic, objective, Physics constraints,
grouped PPO and one-action-K semantics. It adds one training authority: the
complete global curriculum coordinates executable-evidence horizon K and exact
attempts per selected Segment M. The schedule, review boundaries and maximum
absolute iteration are frozen before the first transaction.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-02` | Segment Replay | `SR-01` | Exact-M Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Global Coordinated K x M Schedule |
| `FRS-DP-08` | HSL Warmup | `M-03` | First Entry Into The New Identity |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Per-Stage Recalibration And Actor Ramp |

## Preserved Method Authority

- actor input remains the deployable 158D prefix;
- actor output remains one full-6D `Delta SE(3)` action at t;
- one attempt remains one PPO policy row regardless of K;
- scalar Critic predicts only FRS-GAIN-v006 paired Intent improvement minus
  full-6D repair cost;
- expected/actual Contact, loaded-support phase-ZMP and survival remain
  independent FRS-PPO-v004 actor constraints;
- Clean continuation remains GMT/Physics-evaluator evidence only;
- one sealed multi-Segment x M transaction produces exactly one grouped
  optimizer update.

No rho, second actor/Critic/optimizer, K actor input, Clean actor future,
Noisy prefix, noise label or perturbation timing is introduced.

## First Entry Into The New Identity

A fresh v011 campaign may initialize actor, std and the 158D actor-prefix
normalizer only from strict `frontres-v015-hsl-proposal-v1`. It must then:

```text
fresh-initialize the scalar Critic and Critic normalizer/state
fresh-initialize Stage-3 optimizer and sampler state
resolve the complete K x M schedule and fingerprint
set phase=critic_only and actor_loss_weight=0
freeze actor and std exactly
```

Checkpoint-v5 and earlier, including the E-FI-86 K8/M2 pilot, reject before
mutating actor, Critic, normalizers, optimizer, sampler or curriculum. Old
Stage-3 actor-only migration remains forbidden; HSL-v1 is the only cold-start
actor source.

## Scalar Critic Authority

At active coordinated stage j:

```text
V^I_j(o_critic_t) ~= E[y_I,K_j | o_critic_t]
A^I_Kj = y_I,K_j - V^I_old_j(o_critic_t)
```

There is one Critic, no K input and no K-specific head. Contact, ZMP, survival,
admissibility, constraint residuals, projection status and replay priority may
not enter the Critic target, value loss or value normalization.

## Global Coordinated K x M Schedule

The immutable campaign schedule is:

```text
C_KM = [(K_j, M_j, N_c_j, N_a_j, N_joint-review_j)]

stage 0 = (8,  2, 200, 500, 1300)
stage 1 = (16, 3, 300, 300,  900)
stage 2 = (32, 4, 400, 300,  625)
```

K is positive and strictly increasing. M is an integer at least two and
non-decreasing. Every stage has positive critic-only and actor-ramp spans.
Every non-final stage has a positive joint span before transition. The final
stage remains joint after its declared first review span.

The same fingerprint binds:

```text
selected_segment_count = 2
maximum_absolute_iteration = 8000
checkpoint_review_boundaries = (2000, 3500, 4825, 6500, 8000)
policy_rows_per_transaction = selected_segment_count * M_j
role_rows_per_transaction = 2 * policy_rows_per_transaction
required_num_envs = role_rows_per_transaction
```

The formal layouts are therefore K8/M2/8-env, K16/M3/12-env and
K32/M4/16-env. Every selected Segment contributes exactly M Repair policy rows
and paired Noisy role rows. Replay priority selects Segment identities only;
it cannot change stage M, transaction width or grouped mass. Legacy
state-driven `rollout_trial_count` remains outside the v011 formal route.

## Per-Stage Recalibration And Actor Ramp

For stage-local committed iteration l_j:

```text
0 <= l_j < N_c_j:
    phase = critic_only
    actor_loss_weight = 0

N_c_j <= l_j < N_c_j + N_a_j:
    phase = actor_warmup
    actor_loss_weight = (l_j - N_c_j + 1) / N_a_j

l_j >= N_c_j + N_a_j:
    phase = joint
    actor_loss_weight = 1
```

At every coordinated K x M transition, the same Critic parameters and
compatible v011 optimizer state continue. The next committed transaction
re-enters critic-only; actor/std freeze but are not reinitialized. After
recalibration, FRS-PPO-v004 projects the actor direction before the declared
ramp weight is applied.

The first review spans give approximately equal actor evidence mass:

```text
K8/M2:  2*2*(1300 + (500+1)/2) = 6202 actor-equivalent rows
K16/M3: 2*3*( 900 + (300+1)/2) = 6303 actor-equivalent rows
K32/M4: 2*4*( 625 + (300+1)/2) = 6204 actor-equivalent rows
```

These are weighted policy-row equivalents, not optimizer-step counts and not
K-step physical frames.

## Transaction Atomicity

Curriculum, exact-M layout, target, constraints, solver and frozen-policy
identity resolve before transaction open and remain sealed. Collection performs
no optimizer step. A failed or partial transaction cannot advance phase, K, M,
absolute iteration, optimizer count or checkpoint receipt.

Stage advancement happens only after a committed receipt. The next transaction
observes the new K and M at stage iteration zero and critic-only phase.

## Checkpointed Quality Blocks

Each declared boundary is one normal decision unit:

```text
official formal training to the absolute boundary
-> committed checkpoint
-> existing structured telemetry review
-> existing policy-only deployment evaluation when required
-> CONTINUE | PAUSE-REPAIR | STOP-DESIGN
```

The first formal transaction at K16/M3 and K32/M4 is also the first-live
runtime audit for that predeclared stage. Existing telemetry must prove active
K/M, phase, actor/std freeze, Critic delta, exact-one update, KKT facts,
committed receipt and unchanged schedule fingerprint. A separate tiny sentinel
is prohibited when those facts are complete.

Training telemetry owns learning continuation, not held-out efficacy. Review
rolling Intent improvement, repair cost, Contact preservation, loaded-support
phase-ZMP, survival, sustained lateral lean, unplanned support changes, action
non-collapse, value calibration and constraint/KKT health. Do not redesign or
stop from one bad transaction.

## Checkpoint-v6 Identity

```text
checkpoint_schema = frontres-v015-checkpoint-v6
method_contract_id = FRS-METHOD-v016
gain_contract_id = FRS-GAIN-v006
optimization_contract_id = FRS-PPO-v004
training_contract_id = FRS-TRAIN-v011
scalar_target_id = paired-intent-minus-repair-v1
constraint_schema_id = contact-loaded-phase_zmp-survival-physical-v2
projection_schema_id = grouped-first-order-constraint-projection-v1
```

The payload binds the observation/normalizer identity, complete K x M schedule
and fingerprint, selected Segment count, maximum absolute iteration, review
boundaries, stage/K/M/local iteration/phase/actor weight, derived role and
policy row counts, physical budgets/scales, solver fingerprint, absolute
committed update, sampler state, optimizer state and matching committed
transaction receipt. The projection owns no persistent dual state.

Full resume requires exact v6 equality before mutable restoration. Different
K/M schedules, maximum iteration, review boundaries, selected Segment count,
active-stage environment width, partial transaction, HSL-as-full-resume,
tampered solver schema or absent target identity reject pre-mutation.

## Required Evidence

The one-shot engineering closure must prove:

- schedule parsing, fingerprint and absolute boundary mapping;
- exact two-Segment x M layouts at M=2/3/4 and homogeneous K;
- deterministic environment-width and legacy state-driven-M rejection;
- critic-only isolation, same-Critic stage transition and projected actor ramp;
- grouped equal mass, exact-one update and committed-only advancement;
- checkpoint-v6 strict save/resume and checkpoint-v5 pre-mutation rejection;
- final serialized K/M/phase/row-count/quality telemetry.

Only the first official transaction of a later predeclared stage requires new
live runtime evidence. Subsequent blocks reuse the same formal route and are
owned by policy-quality continuation review.

## Stop Conditions

Stop if actor/std drift during critic-only, the Critic is reinitialized at a
stage transition, Physics reaches the scalar Critic, the sampler changes exact
M, a transaction mixes K/M or Segment mass, schedule/max/boundaries mutate on
resume, checkpoint-v5 mutates v011, exact-one update fails, or a sufficiently
informative quality block shows systematic no-op, sustained lean, unplanned
support change or Physics regression.
