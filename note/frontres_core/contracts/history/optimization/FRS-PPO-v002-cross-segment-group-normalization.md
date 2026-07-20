---
contract_id: FRS-PPO-v002
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-19
supersedes: FRS-PPO-v001
superseded_by: FRS-PPO-v003
scope: Stage 3 Segment PPO grouped advantage scaling and cross-motion, cross-Segment, cross-attempt, and cross-horizon loss aggregation
---

# FrontRES Segment PPO Cross-Segment Group Normalization Contract

## Design Delta

The prior scale-only rule protected the sign of positive executable Gain, but
its flat row mean did not define how a transaction with several motions,
segments, repeated attempts, and different K horizons shares gradient mass.

The active rule adds a minimal hierarchy:

    motion group
    -> selected Segment
    -> eligible policy attempt
    -> valid K-step row

It replaces flat valid-row averaging for the active Segment Replay route. It
does not add a confidence head, acceptance gate, noise detector, or a separate
policy objective.

## PPO Eligibility Domain

An eligible row must have all of:

- an action sampled from the transaction's frozen pi_old;
- matching old mean, sigma, log probability, return, and advantage;
- matching transaction_id, policy_snapshot_id, motion_id, segment_id, and
  trial_index;
- a fixed Noisy-segment identity that matches the actor reference window;
- an ordinary valid/reset/finite mask.

Search, manually edited, oracle, Clean, and counterfactual rows are excluded
from PPO regardless of their Gain. An M-attempt transaction includes every
eligible policy-sampled attempt, not merely trial index zero.

## Grouped Reduction

Let G be the motion groups in one completed old-policy transaction. Let S_g be
the selected valid Segments of motion g, M_gs the eligible policy attempts for
Segment s, and V_gsm the valid K-step rows in one attempt. The actor surrogate
is reduced as:

    L_actor =
      mean_g [
        mean_s [
          mean_m [
            mean_i in V_gsm [ l_ppo(A_hat_gsmi, old_policy, new_policy) ]
          ]
        ]
      ]

The critic/value and entropy terms must consume the same group metadata or
state explicitly why they use a different non-actor reduction. The actor
surrogate is the authority boundary for this contract.

Consequences:

- each represented motion receives equal outer mass;
- each selected Segment of that motion receives equal mass;
- each eligible policy attempt of that Segment receives equal mass;
- each attempt is a mean over its own valid K-step rows;
- duplicating a valid row, adding extra M attempts, or choosing a longer K
  cannot silently increase that group's actor-loss mass.

Empty or entirely invalid groups are excluded before all means are formed and
must be reported. They never become zero-filled groups that alter another
group's scale.

## Sign-Preserving Grouped Advantage Scaling

The default mode is grouped_scale_only. It never mean-centers advantages.

For a Segment group gs, define:

    r_gs = RMS(A_gs over all its eligible valid rows)
    r_txn = RMS(A over the completed eligible transaction)
    d_gs = max(r_gs, r_txn)
    A_hat_gsmi = A_gsmi / d_gs

The transaction RMS is a non-amplifying floor: a low-magnitude group is not
scaled up above its original relative size, while a high-magnitude group cannot
dominate merely through a larger raw advantage scale. Zero stays zero and the
sign of every nonzero advantage is preserved.

The active route forbids standard mean-centering. A standard PPO normalizer may
exist only as an explicitly named ablation outside the active FrontRES route.
The denominator must be detached and finite; invalid or missing group metadata
fails closed rather than falling back to a flat batch mean.

## Sampling Versus Loss Authority

Outer replay priority, source quota, staleness, and segment state decide which
segments enter a transaction. They are sampling authority only.

The following values must not multiply an already assembled actor loss:

- replay priority, source quota, or staleness;
- raw Gain, best-of-M gain, or a replay confidence statistic;
- M trial count or trial index;
- K horizon or valid-step count;
- focal powers such as absolute-advantage squared;
- post-update KL, logger values, or stale policy diagnostics.

Raw advantage remains the PPO signal after the sign-preserving scale. This
contract prevents group mass domination; it does not erase genuine conflicting
gradients from distinct, distinguishable motion contexts. Future Noisy context
is the active method mechanism for resolving observation aliasing.

## Required Batch Metadata

The PPO batch must carry, aligned with every stored valid row:

    transaction_id
    policy_snapshot_id
    motion_id
    segment_id
    trial_index
    policy_sampled flag
    valid_step mask
    horizon_k

The loss owner may derive group indices from these fields, but it must not
reconstruct identity from row order. Permuting rows without changing metadata
must preserve the loss and group-mass diagnostics.

## Ownership

    source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py
      owns grouped_scale_only construction and actor-loss reduction.

    source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py
      owns aligned stored old-policy tensors, advantages, masks, and group
      metadata.

    source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py
      owns transaction-complete batch delivery and exactly-one optimizer-step
      accounting.

    source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py
      owns selection and replay priority, not PPO loss weights.

## Required Tests

| Acceptance object | Tier / kind | Required proof |
| --- | --- | --- |
| grouped formula | S1 T-value/T-permute | Hand-computed two-motion, two-Segment, unequal-M, unequal-K loss equals the nested mean after row permutation |
| sign preservation | S1 T-sign/T-scale | All-positive rows remain positive; low-magnitude groups are not amplified; high-scale group is bounded by the transaction floor |
| M/K invariance | S1 T-metamorphic | Duplicating valid rows or adding an equivalent M/K repeat does not change the represented group's loss mass |
| loss isolation | S1 T-static/T-source | Priority, raw Gain, trial count, K, and focal advantage powers cannot enter actor-loss multiplication |
| metadata integrity | S1 T-schema/T-missing | Missing/misaligned motion, Segment, trial, snapshot, or mask metadata fails closed |
| formal route | S2 T-connect/T-order | The formal Stage 3 transaction reaches grouped loss once, then one optimizer step |
| live sentinel | S4 T-live/T-mass | Real transaction logs per-motion, per-Segment, per-attempt mass shares, scale statistics, and one optimizer step |

## Required Diagnostics

- chosen and valid motion/Segment/attempt counts;
- eligible versus excluded row-role counts;
- per-group r_gs and r_txn;
- sign_flip_count, nonfinite_group_count, and missing_metadata_count;
- per-motion, per-Segment, per-attempt, and per-step actor-loss mass shares;
- explicit flags for priority_weight_used, focal_advantage_power_used, and
  K_or_M_loss_multiplier_used, each false on the active route;
- optimizer_step_count and policy_snapshot_id for the transaction.

## Decision Boundary

If the group-normalized actor loss still shows poor policy improvement, first
separate three questions:

    observation aliasing
    -> future-context provenance and H layout;

    transaction correctness
    -> fixed pi_old, fixed Noisy segment, M attempts, one update;

    optimization behavior
    -> sigma, trust region, critic, and one-step policy response.

Do not reintroduce priority weighting, a focal advantage term, or a new
acceptance/gating variable as an unexamined remedy.
