---
contract_id: FRS-METHOD-v013
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-19
supersedes: FRS-METHOD-v012
superseded_by: FRS-METHOD-v014
scope: FrontRES Stage 3 local reference repair with fixed Noisy future context, fixed-policy multi-attempt Segment Replay, and one grouped cross-Segment PPO transaction
---

# Future-Conditioned Double Segment Replay Method Contract

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| FRS-DP-01 | Perturbation Data | M-02 | Fixed Noisy Segment |
| FRS-DP-02 | Segment Replay | SR-01 | Double Segment Replay Transaction |
| FRS-DP-03 | K-step Curriculum | M-06 | K-step Execution And Evidence |
| FRS-DP-04 | FrontRES 6D Repair | M-04 | Actor Observation And Action |
| FRS-DP-05 | Frozen GMT | M-10 | Method Boundary |
| FRS-DP-10 | Future Motion Context | M-11 | Future Noisy Context |

## Method Boundary

FrontRES remains a local reference-repair policy before frozen GMT:

    replayable local dynamics x_t
    + one fixed Noisy reference segment
    + current and sparse future Noisy reference context
    -> full-6D Delta SE(3) repair
    -> K-step frozen-GMT execution
    -> paired executable Gain

It does not generate a replacement motion, own joint-space tracking, infer a
Clean sequence, or recover arbitrary historical state error. The deployment
assumption is the existing non-streaming reference-file path: GMT receives a
complete reference sequence from a .npz source. Streaming teleoperation, where
future reference availability itself changes, is outside this contract.

The active learned action remains:

    [dx, dy, dz, droll, dpitch, dyaw]

Perturbation family only defines the corruption distribution. It never narrows
the six-dimensional repair output.

## Dynamic Start And Local-Recovery Boundary

For a selected segment s beginning at t, x_t is the dynamic start of the
K-step segment. It contains the replayable Clean robot/controller state needed
to reproduce local frozen-GMT dynamics, including root and joint state,
velocity, phase/contact-relevant state, and required controller caches.

Resetting every attempt to Clean replay state x_t is a reproducibility
requirement for comparing repeated attempts. It is not an actor observation
and does not give the actor a Clean reference.

The active method deliberately does not require a Noisy physical rollout before
x_t. Adding that prefix would change the scientific object from local
reference repair to recovery from accumulated historical state error. That is a
separate method and is not silently included here.

If an actor input explicitly contains reference-bearing history before t, its
provenance must be audited as an observation-interface issue. It must not leak
Clean reference, but it also does not create a requirement for a Noisy physical
roll-in in this method version.

## Fixed Noisy Segment

Each selected segment has one immutable corruption realization:

    Clean reference segment R_s
    -> fixed corruption realization
    -> Noisy reference segment R_tilde_s

The same R_tilde_s is installed for all M policy attempts from x_t. Its current
frame, sparse future context, and every reference frame executed during the
K-step rollout come from that one segment. A new perturbation seed, a Clean
future frame, or a mixed-reference window inside one attempt is forbidden.

R_tilde_s may contain a distributed or persistent artifact over most or all
frames. The method does not assume a single known disturbance time. The actor
does not receive a perturbation family, noise label, noise timing, Clean
counterpart, or artifact truth.

Clean reference is retained only for paired training and evaluation evidence.
It is never appended to the deployable actor reference window.

### Selection-Time Lifecycle

The segment-selection owner samples the corruption realization exactly once
when it selects scenario s, before policy attempt 1, and binds the resulting
R_tilde_s to that scenario as an immutable replay artifact. Its coverage
includes the current actor reference, every H future offset, and every
reference frame that frozen GMT executes over K.
Resetting x_t for a repeated attempt must not invoke the perturbation sampler.
All M_s attempts read the same R_tilde_s and carry the same scenario identity,
noisy_segment_hash, provenance, and coverage metadata. An implementation may
store a deterministic seed instead of all values only when rematerialization
is byte-equivalent and verifies the same noisy_segment_hash.
The scenario binding ends after its M_s attempts and their evidence aggregation.
This is a semantic lifetime, not a memory-allocation rule: an immutable cache
may retain the artifact for replay evidence, but it may not resample or mutate
it under the same scenario identity. A later selection is a new scenario and
must receive a new identity before it draws another realization.

### Canonical Carrier And Runtime Owner

For the active G1 command surface, the sealed reference carrier is the
following ordered Noisy reference tape, not the legacy joint-only window:

    T_tilde_s: [L, 65]
      = [joint_pos(29), joint_vel(29), anchor_pos(3), anchor_quat(4)]
      with L >= K + max(H)

`MultiMotionCommand` is the single runtime materializer and cursor owner for
this carrier. The sampler-domain lifecycle remains the semantic selection owner:
it invokes the command-owned materializer once per selected source, seals the
returned values and hash, and fans the immutable result out to its M rows.
Reset and actor code are consumers; they may install/read the carrier but may
not materialize, mutate, or replace it.

The 3D anchor position and 4D anchor quaternion are existing deployable motion
command fields. They are included because the active `local_rp` corruption
changes that reference surface while leaving the 58D joint command unchanged.
They are not robot-state, Clean-root, contact, label, timing, or truth channels.
If a later perturbation changes another executable reference field, the carrier
layout must receive an explicit new version and fail closed rather than silently
reuse the 65D layout.

## Future Noisy Context

The actor receives its current deployable reference plus an ordered sparse set
of positive future offsets:

    H = {h_1, ..., h_n}, 0 < h_1 < ... < h_n <= H_max
    H_max is a local window of approximately 0.5 seconds

At every rollout frame t+i, both the current reference and each future
reference at t+i+h_j are read from R_tilde_s. The representation is the same
GMT-compatible reference-command representation already available at
deployment, rather than a new Clean-root, contact, or perturbation-truth
channel. The exact offset list is configuration, but it must be nonempty,
ordered, fixed in each run, serialized with the actor, and identical in
training and deployment.

This context makes locally similar current frames distinguishable when their
upcoming motion demands opposite repairs. It does not make future Noisy
reference privileged information: the deployed .npz reference provides the
same Noisy sequence. It would be privileged only if the actor received its
Clean counterpart or metadata unavailable to deployment, both of which are
forbidden.

`frontres_future_offsets` is a required nonempty serialized configuration for
the v013 route; there is no implicit runtime default. The existing current
command/anchor observations must be tape-backed. The additional actor tail is
the ordered future projection `[B, |H|, 65]` flattened as `[B, |H| * 65]`.
It reads the same carrier without advancing its execution cursor.

The future-information horizon H and execution/evidence horizon K are distinct:

    H controls what the actor can observe.
    K controls how long frozen GMT executes and how Gain/returns are measured.

K may vary through its curriculum. For every assigned K, the fixed R_tilde_s
must cover both the actor context and all executed reference frames through
the required maximum of K and H.

## Actor Observation And Action

The active actor input is:

    existing robot, balance, and tracking observation
    + current R_tilde_s command reference
    + sparse future R_tilde_s command references at H
    -> full-6D FrontRES distribution

Stage 2 HSL and Stage 3 PPO must use the same resulting actor input layout,
normalizer layout, and full-6D action head. Checkpoint compatibility is an
engineering requirement, not an exception that permits Stage 3 to omit the
future input.

The action is sampled from the old policy, projected only through named
full-6D safety constraints, written as a residual reference, and executed by
frozen GMT. Local-rp corruption does not become an RP action mask. Upward dz
constraints remain safety constraints, not a perturbation-family policy.

## Double Segment Replay Transaction

One collection transaction freezes a single policy snapshot pi_old and selects
multiple replay segments. For each selected segment s:

    1. restore the same Clean dynamic start x_t;
    2. install the same fixed R_tilde_s;
    3. collect M_s >= 2 independently policy-sampled repair attempts under pi_old;
    4. store old distribution statistics, action, return, advantage, valid
       steps, motion/segment/trial identity, and noisy-segment identity;
    5. aggregate rollout evidence for replay priority only after all attempts.

No optimizer step may occur between these attempts, between selected segments,
or before the entire transaction is assembled. One completed transaction has:

    one policy_snapshot_id
    + multiple selected segment groups
    + all eligible M_s policy attempts
    -> one grouped PPO update
    -> one post-collection sampler-evidence update

All M_s attempts sampled by pi_old are PPO-eligible if their ordinary validity
requirements hold. The first attempt must not be the only row labelled policy
while later policy samples are relabelled search. Search, manually edited,
oracle, or counterfactual actions may improve replay evidence but must not
receive PPO credit.

Best-of-M is a replay-evidence statistic, not a PPO sample selector. Selecting
only the best sampled action as though it represented pi_old would change the
policy-gradient distribution and is forbidden.

## Cross-Segment PPO Aggregation

PPO consumes every valid policy-sampled attempt from the frozen transaction,
but no motion, segment, repeat count, or horizon may acquire excess gradient
mass merely because it produced more rows or a larger replay priority.

The loss is reduced hierarchically:

    equal mass per motion group
    -> equal mass per selected Segment in that motion
    -> equal mass per eligible policy attempt
    -> mean over that attempt's valid K-step rows

Priority, source quota, M_s, K, and raw Gain affect what is sampled or what
evidence exists; none may multiply the PPO loss after a batch has been
constructed. The exact sign-preserving grouped advantage rule is owned by
FRS-PPO-v002.

This normalization does not assert that all gradient directions should agree.
Genuine opposite repairs remain valid when their future Noisy contexts differ.
Its purpose is narrower: prevent one motion or repeated/high-row segment from
silencing the others. Future context resolves observation aliasing; grouped
aggregation prevents data-mass domination.

## K-step Execution And Evidence

K remains the executable evidence horizon:

    short K: immediate local repair evidence
    longer K: delayed regret and accumulated execution consequence

The curriculum may assign different K values after a segment has been chosen,
but each policy attempt retains its own K and valid-step mask. K must not be
used as an implicit loss multiplier. The grouped PPO reduction gives each
eligible attempt one normalized contribution regardless of valid-step count.

Paired Clean/Noisy/Repaired execution continues to provide:

    gain_total = w_style * style_gain
               + w_physics * physics_gain
               - w_repair * repair_cost

Clean is the immutable comparison target; Noisy/GMT and Repaired/GMT are
paired executable branches. Generic environment, teleoperation, velocity
command, unrelated task reward, and legacy RP-only score remain forbidden
return or priority authorities.

## Required Diagnostics And Evidence

The formal path must expose:

- policy_snapshot_id, transaction_id, motion_id, segment_id, trial_index, and
  per-attempt valid-step count;
- immutable noisy_segment_hash plus future-offset layout and provenance;
- a proof that actor-visible current/future/K-step reference is from that
  Noisy segment and that no Clean reference reaches the actor;
- Clean x_t reset fidelity without claiming a Noisy prefix rollout;
- attempted, valid, PPO-eligible, and non-policy row counts for every M_s;
- optimizer-step count exactly one per completed transaction;
- H and K distributions reported separately;
- grouped loss mass share by motion, Segment, attempt, and valid step;
- advantage sign-flip count, group scale statistics, and absence of
  priority/focal/Gain/M/K loss multipliers;
- full-6D raw, bounded, and executed actions; paired Gain; harmful-repair
  fraction; and replay-priority evidence.

Before live training, deterministic tests must prove the fixed-reference,
policy-snapshot, grouped-loss, and formal-route connectivity contracts. A live
sentinel may validate only the real runtime identities and diagnostics; it does
not by itself establish long-horizon policy quality.

## Forbidden Active-Path Assumptions

- Clean x_t means the actor observes Clean reference.
- A Noisy physical prefix before x_t is required for local repair.
- Future actor context may use Clean reference, noise labels, artifact timing,
  perturbation metadata, or any deployment-unavailable signal.
- Every-frame Noisy reference makes local repair meaningless by definition.
- Repeated attempts may use a fresh perturbation realization.
- First-policy plus later-search rows implement M on-policy attempts.
- An optimizer step may occur before a frozen-policy transaction is complete.
- Best-of-M or manually searched actions may be treated as ordinary PPO samples.
- Priority, source, M, K, valid-step count, raw Gain, or focal advantage power
  may silently multiply PPO loss mass.
- K and H are interchangeable horizons.
- A current generic observation history silently proves accumulated-state
  recovery capability.

## Owned Subcontracts

- Formal Stage 3 training route:
  ../training/FRS-TRAIN-v004-future-context-double-segment-transaction.md.
- Grouped PPO and advantage semantics:
  ../optimization/FRS-PPO-v002-cross-segment-group-normalization.md.
- Paired reward semantics:
  ../reward/FRS-GAIN-v002-style-physics-repair.md.
- Periodic and sequence evaluation:
  ../evaluation/FRS-EVAL-v002-segment-evaluation.md.
