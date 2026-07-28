---
contract_id: FRS-TRAIN-v010
status: superseded
effective_date: 2026-07-23
updated_date: 2026-07-28
supersedes: FRS-TRAIN-v009
superseded_by: FRS-TRAIN-v011
scope: fresh scalar Intent-Critic initialization, strict v006 target identity, per-global-K critic-only recalibration, projected actor ramp, joint exact-one update, and checkpoint-v5 persistence
---

# Intent-Critic K-Stage Curriculum

## Design Delta

FRS-TRAIN-v009 established global homogeneous K stages and repeated Critic
recalibration, but its scalar Critic predicted the superseded v004 mixed
Physics/Intent utility. FRS-TRAIN-v010 retains one Critic and the same K-stage
curriculum while changing its target to v005 paired Intent improvement minus
repair cost.

This target change is not checkpoint-compatible. A v004 Critic cannot be
relabelled, padded, or partially restored.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-03` | K-step Curriculum | `M-06` | Global K Identity |
| `FRS-DP-08` | HSL Warmup | `M-03` | Actor-Only Initialization |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Fresh Target And Per-K Recalibration |

## First Entry Into The New Target

A fresh v010 run may initialize actor, std, and the 158D actor-prefix
normalizer only from strict `frontres-v015-hsl-proposal-v1`. It must then:

```text
fresh-initialize the scalar Critic
fresh-initialize Critic normalizer/state
fresh-initialize the Stage-3 optimizer state
set phase=critic_only and actor_loss_weight=0
freeze actor and std exactly
```

Any Stage-3 checkpoint whose Critic target is v004, whose contract set is
METHOD-v015/GAIN-v004/PPO-v003/TRAIN-v009, or whose target identity is absent
rejects before mutating actor, Critic, normalizers, optimizer, sampler, or
curriculum. Old Stage-3 actor-only migration is also forbidden; HSL-v1 remains
the only cold-start actor source.

## Scalar Critic Authority

At active global stage `j`:

```text
V^I_j(o_critic_t) ~= E[y_I,K_j | o_critic_t]
A^I_Kj = y_I,K_j - V^I_old_j(o_critic_t)
```

`y_I` is exactly FRS-GAIN-v006 paired Intent improvement minus repair cost.
Contact, ZMP, survival, admissibility, constraint residuals, projection status,
or replay priority may not enter value targets, value loss, or value
normalization.

There is one Critic, no K input, and no K-specific head. One transaction uses
one active K for every Segment and attempt.

## Global K Schedule

The explicit immutable schedule remains:

```text
C_K = [(K_j, N_c_j, N_a_j, N_joint_j)]
```

K values are positive, strictly increasing, bounded by Kmax, and fingerprinted
before collection. Every stage has `N_c_j>0` and `N_a_j>0`; every non-final
stage has `N_joint_j>0`; the final stage remains joint after warmup. No implicit
formal default or per-Segment K is permitted.

## Per-Stage Recalibration And Actor Ramp

For stage-local committed iteration `l_j`:

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

On the first v010 stage, the fresh Critic calibrates while actor/std freeze. At
every later global K increase, the same Critic parameters and compatible v010
Critic optimizer state continue, but the next committed transaction re-enters
critic-only. Actor/std freeze again; they are not reinitialized. After
recalibration, FRS-PPO-v004 projects the actor direction and the ramp weight is
applied after projection.

The Critic remains trainable in all phases. A critic-only transaction may
commit one optimizer step with actor/std delta exactly zero.

## Transaction Atomicity

Curriculum, target, constraint, and solver identities are resolved before a
transaction opens and remain sealed. Collection performs no optimizer step.
One complete multi-Segment x M transaction produces exactly one optimizer
step. A failed/partial transaction cannot advance phase, K, optimizer count, or
checkpoint receipt.

Stage advancement happens only after a committed receipt. The next transaction
observes the new K at stage iteration zero and critic-only phase.

## Checkpoint-v5 Identity

The active persistence identity is:

```text
checkpoint_schema = frontres-v015-checkpoint-v5
method_contract_id = FRS-METHOD-v016
gain_contract_id = FRS-GAIN-v006
optimization_contract_id = FRS-PPO-v004
training_contract_id = FRS-TRAIN-v010
scalar_target_id = paired-intent-minus-repair-v1
constraint_schema_id = contact-loaded-phase_zmp-survival-physical-v2
projection_schema_id = grouped-first-order-constraint-projection-v1
```

The payload also binds observation/normalizer identity, explicit K schedule and
fingerprint, stage/K/local iteration/phase/actor weight, physical budgets and
scales, solver tolerance/fingerprint, absolute committed update, sampler state,
optimizer state, and matching transaction receipt. The projection owns no
persistent dual state.

Full resume requires exact v5 equality before any mutable restoration.
Checkpoint-v4/v009, v004/v003 identities, HSL payloads used as full resume,
partial transactions, different schedules, tampered solver schemas, or absent
target identity reject pre-mutation.

## Required Diagnostics

- initialization source and fresh Critic/optimizer identities;
- stage/K/local iteration/phase/actor weight and schedule fingerprint;
- scalar Intent target, value, raw/scaled advantage, value error and Critic
  gradient/delta;
- independent constraint/projection status without Critic contamination;
- actor/std delta, exact-one optimizer count, committed receipt and v5 save
  identity.

## Forbidden Behavior

- restoring or migrating a v004 Critic or old Stage-3 actor;
- retaining optimizer moments from v004/v009;
- Physics in Critic target/loss/normalizer;
- Critic reinitialization at later K changes;
- actor/std update during critic-only;
- mixed K, phase change during collection, or advancement before commit;
- Multi-Critic, K actor input, second optimizer, or HSL Stage-3 target.

## Acceptance And Stop Conditions

P2 S1/S2/S3 must prove fresh-target initialization, exact phase mapping,
critic-only isolation, same-Critic K transition, projected actor ramp,
checkpoint-v5 exact resume, and v4/v004 pre-mutation rejection. P3 must prove
one real committed transaction with scalar-Critic-only target and the declared
gradient authority.

Stop if any old target/state mutates v010, actor/std drift during critic-only,
constraint evidence reaches value loss, a K transition reinitializes Critic,
the checkpoint cannot bind the complete coordinated identity, or more than one
optimizer step is required.
