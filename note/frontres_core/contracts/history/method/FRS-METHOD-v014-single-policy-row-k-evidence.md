---
contract_id: FRS-METHOD-v014
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-19
supersedes: FRS-METHOD-v013
superseded_by: FRS-METHOD-v015
scope: FrontRES Stage 3 local reference repair with fixed Noisy future context, fixed-policy multi-attempt Segment Replay, one PPO policy row per attempt, and K-step executable evidence
---

# Future-Conditioned Double Segment Replay with Single-Row K Evidence

## Design Delta

This version resolves the formal storage meaning of K:

    one policy-sampled attempt
    -> one old-policy action tuple
    -> one PPO policy row
    -> K-step frozen-GMT executable evidence
    -> one aggregated return and advantage for that row

K is not a PPO row axis. It never creates K policy actions, K old-policy
statistics, or K contributions to actor-loss mass. This replaces the v013
wording that treated an attempt as a mean over valid K-step PPO rows.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| FRS-DP-01 | Perturbation Data | M-02 | Fixed Noisy Segment |
| FRS-DP-02 | Segment Replay | SR-01 | Frozen-Policy Transaction |
| FRS-DP-03 | K-step Curriculum | M-06 | Single-Row K Evidence |
| FRS-DP-04 | FrontRES 6D Repair | M-04 | Actor Observation And Action |
| FRS-DP-05 | Frozen GMT | M-10 | Method Boundary |
| FRS-DP-10 | Future Motion Context | M-11 | Fixed Noisy Future Context |

## Method And Recovery Boundary

FrontRES remains a local reference-repair policy before frozen GMT:

    replayable Clean dynamic state x_t
    + one fixed Noisy reference segment R_tilde_s
    + current and sparse future Noisy context
    -> full-6D Delta SE(3) repair action
    -> K-step frozen-GMT execution
    -> paired executable Gain

The deployment assumption is the existing non-streaming .npz reference-file
path. It does not generate a replacement motion, infer Clean reference, own
joint-space tracking, or solve historical accumulated-state recovery.

For a selected Segment s at t, Clean x_t is the replayable robot/controller
state needed to reproduce local dynamics. Resetting every attempt to x_t is
required for repeatability only. It is neither an actor observation nor Clean
reference. No Noisy physical prefix before x_t is required or introduced.

## Fixed Noisy Segment And Future Context

At selection time, exactly one immutable Noisy tape is materialized:

    T_tilde_s: [L, 65]
      = [joint_pos(29), joint_vel(29), anchor_pos(3), anchor_quat(4)]
      with L >= K + max(H)

MultiMotionCommand owns tape materialization and its execution cursor. The
sampler lifecycle owns selection-time binding: it materializes the tape once,
seals scenario_id, noisy_segment_hash, coverage, and provenance, then fans the
same values out to all M_s attempts. Reset, actor, and GMT are consumers; they
may install or read the tape but may not resample, mutate, mix, or replace it
under that scenario identity.

The actor sees the current tape frame and ordered positive future offsets H.
H is a deployment-available actor-information horizon. K is the independent
frozen-GMT execution/evidence horizon. H reads do not advance the command
cursor; K execution does. Clean future, perturbation labels, timing, and truth
metadata are forbidden actor inputs.

## Single Policy Row And K-step Executable Evidence

For each motion g, Segment s, and policy attempt m, storage holds:

    P_gsm = (observation_t, sampled_action_t, old statistics,
             return_K, advantage_K, policy_row_valid,
             transaction/motion/segment/trial/noisy identities,
             horizon_k, evidence_valid_step_count)

There is exactly one P_gsm for one policy-sampled action at the dynamic segment
start. The K execution frames are evidence used to compute return_K and
advantage_K. An evidence-step mask may be retained for diagnostics and return
construction, but it may not materialize additional PPO policy rows.

An attempt is PPO-eligible only when its one old-policy tuple and aggregated
executable evidence satisfy ordinary finite/reset/valid requirements. A shorter
effective K can change its return and advantage, but cannot change policy-row
count or outer loss mass.

## Frozen-Policy Double Segment Transaction

One transaction freezes pi_old, selects multiple Segments, and for each Segment:

    1. restore the same Clean dynamic start x_t;
    2. install the same fixed T_tilde_s;
    3. collect M_s >= 2 independently sampled actions under pi_old;
    4. store one P_gsm per eligible policy attempt;
    5. aggregate executable evidence for replay priority only after all attempts.

No optimizer step occurs during collection or between selected Segments. All
ordinary-valid policy samples remain PPO candidates. Best-of-M, search, manual,
oracle, Clean, Noisy baseline, and counterfactual branches are evidence-only.

## Cross-Segment Grouped PPO Meaning

Actor-loss mass is reduced only across policy rows:

    equal motion mass
    -> equal selected-Segment mass within motion
    -> equal eligible-attempt mass within Segment
    -> one PPO surrogate for the attempt's one policy row

valid_step_count, K, replay priority, source quota, M_s, raw Gain, and
best-of-M evidence do not multiply an assembled actor loss. K affects the
policy signal through return_K and advantage_K, not through row count or loss
mass. The sign-preserving scale and exact formula are owned by FRS-PPO-v003.

## Required Diagnostics And Evidence

The formal path must report:

- transaction_id, policy_snapshot_id, motion_id, segment_id, trial_index,
  noisy_segment_hash, and horizon_k;
- one ppo_policy_row_count per policy attempt and separate
  evidence_valid_step_count/effective-K fields;
- current/H/K provenance from the same tape and absence of Clean actor input;
- Clean x_t reset fidelity without a Noisy physical prefix;
- selected, attempted, valid, PPO-eligible, and non-policy action counts;
- per-motion, per-Segment, and per-attempt loss mass; no per-K-row mass;
- advantage sign/scale statistics, K distribution, and exact-one update count.

The S0 audit code-confirms that current live storage already captures one
first-step policy tuple with K-step accumulated return evidence. It does not
make the legacy formal route compliant: that route still has policy/search role
mixing, immediate updates, and no active v003 grouped connector.

## Forbidden Active-Path Assumptions

- Clean x_t means Clean reference is observable.
- A Noisy physical prefix is required for local repair.
- H and K are interchangeable horizons.
- K execution steps are K PPO policy rows.
- A valid-step count or horizon K increases actor-loss mass.
- Repeated attempts redraw or mutate the fixed Noisy tape.
- First-policy plus later-search rows implements M on-policy attempts.
- Best-of-M evidence becomes a PPO sample selector.
- An optimizer step occurs before the frozen-policy transaction closes.

## Owned Subcontracts

- Formal Stage 3 route: ../training/FRS-TRAIN-v005-single-policy-row-k-evidence-transaction.md.
- Grouped PPO: ../optimization/FRS-PPO-v003-single-policy-row-k-evidence-grouped-reduction.md.
- Paired Gain: ../reward/FRS-GAIN-v002-style-physics-repair.md.
- Evaluation: ../evaluation/FRS-EVAL-v002-segment-evaluation.md.
