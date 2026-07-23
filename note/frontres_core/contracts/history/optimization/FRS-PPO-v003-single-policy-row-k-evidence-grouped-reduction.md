---
contract_id: FRS-PPO-v003
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-20
superseded_by: FRS-PPO-v004
supersedes: FRS-PPO-v002
scope: Stage 3 grouped_scale_only PPO reduction over one policy row per policy-sampled attempt, with K-step executable evidence aggregated into that row
---

# Single-Policy-Row Grouped PPO Contract

## Design Delta

FRS-PPO-v002 used a final valid-K-step-row reduction for its offline candidate
fixture. K-A rejects that storage meaning. The active object is one
policy-sampled action tuple per attempt; K is only the executable-evidence
horizon used to construct that tuple's return and advantage.

This version retains equal motion -> Segment -> attempt mass and
sign-preserving scaling. It removes the valid-step PPO-row level. It adds no
confidence head, acceptance gate, noise detector, or alternative objective.

## Eligibility Domain

An eligible policy row has all of:

- one action sampled from the transaction's frozen pi_old;
- matching old mean, sigma, log probability, value, return_K, and advantage_K;
- matching transaction, snapshot, motion, Segment, trial, and fixed local-
  scenario identity (the compatibility carrier may remain `noisy_segment_hash`);
- policy_sampled, policy_row_valid, and finite/reset-valid evidence;
- horizon_k and evidence_valid_step_count as evidence metadata.

One eligible attempt contributes one and only one PPO row. Search, manually
edited, oracle, Clean, Noisy baseline, and counterfactual rows remain excluded.

## Reference-Scenario Boundary

FRS-METHOD-v015 owns the meaning of the row's scenario identity: current root
artifact, future q29 intent, shared Clean continuation, `x_t`, and K coverage.
This PPO contract only requires exact row alignment and never interprets the
carrier as evidence that all K frames are noisy. It does not authorize future
raw root input, later FEMR actions, or a third Clean scored role.

## Grouped Reduction

Let G be motion groups in one completed frozen-policy transaction, S_g valid
Segments of motion g, and M_gs eligible policy attempts of Segment s. Each
attempt has one surrogate:

    L_actor =
      mean_g [
        mean_s [
          mean_m [
            l_ppo(A_hat_gsm, old_policy, new_policy)
          ]
        ]
      ]

Consequences:

- each represented motion receives equal outer mass;
- each selected Segment within it receives equal mass;
- each eligible policy attempt receives equal mass;
- K, evidence-valid-step count, and effective horizon do not duplicate or
  divide a policy row;
- invalid attempts are excluded before means, never zero-filled.

Value and entropy reductions must use the same row-domain metadata or state
their distinct reduction explicitly. No formal path may silently fall back to a
flat batch mean.

## Sign-Preserving Grouped Scaling

    r_gs = RMS(A_gsm over eligible policy attempts)
    r_txn = RMS(A over the complete eligible transaction)
    d_gs = max(r_gs, r_txn)
    A_hat_gsm = A_gsm / d_gs

The denominator is detached and finite. The transaction floor prevents a
high-scale group from dominating while never mean-centering or flipping the sign
of a nonzero advantage. Missing, partial, or misaligned metadata fails closed.

## Sampling And Evidence Are Not Loss Weights

The following may affect selection, executable evidence, or return construction
but may not multiply an assembled actor loss:

- replay priority, source quota, staleness, or raw/best-of-M Gain;
- trial count M or trial index;
- horizon K, evidence-valid-step count, or effective-K;
- focal powers, post-update KL, logger state, or stale diagnostics.

K changes return_K and advantage_K, not policy-row count, attempt mass, or group
mass.

## Required Metadata And Diagnostics

The complete transaction batch must align:

    transaction_id, policy_snapshot_id, motion_id, segment_id, trial_index,
    noisy_segment_hash, policy_sampled, policy_row_valid,
    horizon_k, evidence_valid_step_count

Diagnostics report selected/eligible/excluded attempt counts,
ppo_policy_row_count, K/evidence-step distributions separately, per-motion /
per-Segment / per-attempt mass, r_gs, r_txn, sign flips, missing metadata, and
exact optimizer-step count.

## Required Tests

| Acceptance object | Tier / kind | Required proof |
| --- | --- | --- |
| K-A row carrier | S1 T-schema/T-value | each eligible attempt stores exactly one action/statistics/return/advantage tuple regardless of K |
| grouped formula | S1 T-value/T-permute | hand-computed unequal-motion/Segment/M transaction equals the three-level mean after row permutation |
| K isolation | S1 T-metamorphic | altering executable evidence cannot add a PPO row or change represented mass except through its returned advantage |
| sign preservation | S1 T-sign/T-scale | positive signs survive and low-scale groups are not amplified |
| isolation | S1 T-static/T-source | priority, Gain, M, K, evidence-step count, and focal terms cannot multiply actor loss |
| formal route | S2 T-connect/T-order | complete transaction reaches v003 once and then exactly one optimizer step |

## Ownership And Current Status

    frontres_segment_storage.py
      owns one-row carrier and K-evidence aggregation.

    frontres_segment_ppo.py
      owns v003 grouped_scale_only reduction.

    frontres_segment_live_sampler.py
      owns the immutable v015 expected-row plan and candidate-shard accumulator.

    frontres_segment_live_update_loop.py -> frontres_segment_live_probe.py
      own fake-S2 request dispatch, complete transaction sealing, and exact-one-update accounting.

The Step 4-S0 audit code-confirms the live storage side of K-A: one first-step
policy tuple plus K-step return evidence. The v002 candidate reducer and its
valid-step-row tests are not evidence for this v003 contract and must be
rebased before formal routing. grouped_scale_only therefore remains excluded
from the legacy formal route.

`E-FI-13` (2026-07-20) adds the v015 candidate-only adapter: storage seals one
metadata row per Repair policy attempt, preserves `x_t`/scenario/hash/q29/K and
actual evidence-step count, and delivers only a complete transaction to this
grouped reducer. `to_ppo_batch()` rejects that v015 carrier, and old fixed-tape
metadata cannot enter the candidate adapter. The grouped formula is unchanged;
no generic formal caller, checkpoint/resume, simulator, training, or live route
is active.

`E-FI-14` (2026-07-20) adds only the bounded fake-S2 caller: it rejects the
legacy adapter, non-grouped normalizer, HSL/warmup flags, and partial or mixed
metadata before loss/step; it then calls the unchanged grouped formula once and
requires one explicit optimizer counter increment. This does not enable the
legacy formal route, checkpoint/resume, or live training.

`E-FI-16` (2026-07-20) adds the separate opt-in pre-live sentinel provider:
selection materializes the split local carrier, the reset/one-action-K/Gain
owners build the same candidate batch, and the existing exact-one transaction
owner consumes it. The generic training route remains isolated; no real
environment or live optimizer evidence is claimed.
