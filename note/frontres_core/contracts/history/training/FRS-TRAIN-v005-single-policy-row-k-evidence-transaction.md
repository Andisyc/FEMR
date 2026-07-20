---
contract_id: FRS-TRAIN-v005
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-19
supersedes: FRS-TRAIN-v004
superseded_by: FRS-TRAIN-v006
scope: HSL initialization, warmup continuity, and formal Stage 3 transaction routing with one PPO policy row per policy attempt
---

# Single-Policy-Row K-Evidence Training Contract

## HSL Interface Continuity

Stage 2 HSL and Stage 3 consume the same deployable FrontRES interface:

    existing robot, balance, and tracking observation
    + current Noisy command reference
    + sparse future Noisy command references at H
    -> one full-6D actor distribution

The active carrier is the 65D sealed Noisy tape
[joint_pos(29), joint_vel(29), anchor_pos(3), anchor_quat(4)]. Future H reads
form [B, |H| * 65]; current reference is tape-backed. The legacy 870D layout
cannot silently load or sample under this route. Checkpoint/normalizer
migration remains a separate S3 gate.

## Fixed Scenario And Reset Route

    select source -> materialize one fixed Noisy tape -> seal hash/provenance
    -> restore Clean dynamic x_t for each M attempt -> install the same tape
    -> read current/H actor context and execute K from that tape

MultiMotionCommand is the tape/cursor owner. Selection binds the tape once;
reset only restores dynamics and installs the sealed artifact. No consumer may
resample, mutate, or mix the tape. Clean x_t does not imply Clean actor
reference, and no Noisy physical prefix belongs to this route.

## One Policy Row Per Attempt

For every policy-sampled attempt, collect exactly one old-policy action tuple:

    observation_t, action_t, old mean/sigma/log_prob/value_t
    + K-step executable evidence
    -> one return_K, one advantage_K, one policy_row_valid

The K execution can contain multiple reference frames and evidence steps, but
it does not create multiple policy rows. horizon_k, evidence_valid_step_count,
and effective-K accompany the single row as diagnostics/evidence metadata:

    PPO policy-row count = eligible policy-attempt count

not the sum of valid K execution steps.

## Formal Transaction Route

    verify future layout/checkpoint compatibility
    -> freeze one pi_old snapshot
    -> select multiple Segments and seal their Noisy tapes
    -> collect all M_s >= 2 policy attempts, one row each, without optimizer mutation
    -> aggregate each row's K-step executable evidence
    -> seal complete transaction metadata
    -> grouped v003 PPO over the policy rows
    -> exactly one optimizer step
    -> update replay priority from rollout evidence only
    -> checkpoint and diagnostics

Clean, Noisy baseline, repaired counterfactual, search, manual, and oracle
branches provide paired Gain or replay evidence only. They are never extra PPO
policy rows.

## K And H Ownership

    H: actor-information layout and deployment-visible future reference.
    K: frozen-GMT execution, delayed-regret evidence, return_K, advantage_K.

K may vary by attempt after tape coverage is verified. It may affect the
return/advantage from that action, but must not create policy rows or alter
motion/Segment/attempt mass. FRS-PPO-v003 owns the resulting reduction.

## Required Metadata And Diagnostics

Every policy row carries:

    transaction_id, policy_snapshot_id, motion_id, segment_id, trial_index,
    noisy_segment_hash, horizon_k, policy_sampled, policy_row_valid,
    evidence_valid_step_count

The formal route reports both ppo_policy_row_count and
evidence_valid_step_count; they must never be conflated. It additionally
reports future-layout identity, H offsets, current/H/K provenance, Clean x_t
reset fidelity, row-role counts, grouped mass/scale, full-6D actions, and one
post-collection optimizer step.

## Acceptance Gates

| Gate | Required proof | Status at activation |
| --- | --- | --- |
| S0 | config/owner/legacy-route isolation | code-confirmed by Step 4-S0 |
| S1 | one-row-per-attempt storage and v003 candidate reduction | pending K-A rebase |
| S2 | formal transaction -> grouped loss -> exact-one update diagnostics | blocked by S1 |
| S3 | versioned checkpoint/resume identity | blocked by S1/S2 |
| S4 | bounded live identity sentinel | user-gated after S1-S3 |

No formal route, checkpoint/resume claim, or live run is authorized by this
document update. The legacy direct update path remains isolated.
