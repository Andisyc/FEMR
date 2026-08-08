---
contract_id: FRS-PPO-v006
status: superseded
effective_date: 2026-08-08
updated_date: 2026-08-08
superseded_by: FRS-PPO-v007
supersedes: FRS-PPO-v005
scope: equal-motion, equal-Segment, equal-attempt grouped PPO with one shared future-conditioned state value per Segment, exact-M mean Critic targets, independent Actor/Critic gradient clipping, and exactly one optimizer update per sealed transaction
---

# Grouped PPO With A Future-Conditioned State-Value Baseline

## Design Delta

FRS-PPO-v005 used one scalar state-value Critic but optimized that value against
each action-specific return while all attempts in a Segment shared the same
state input. It also globally clipped the combined Actor and Critic gradient
vector, allowing the large value error to reduce Actor learning.

FRS-PPO-v006 preserves the grouped scalar Actor objective and exact-one Adam
call. The Critic now fits one exact-M mean target per Segment, all attempts use
one shared old `V(s)`, and Actor/Critic parameter groups are clipped separately.
This remains PPO with a state-value baseline; the 6D action never enters the
Critic and no Q-learning authority is introduced.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-02` | Segment Replay | `SR-01` | Grouped Equal-Mass Reduction |
| `FRS-DP-07` | Repair Gain | `Q-01` | Scalar Actor And Critic Signal |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Warmup Weight Boundary |

## Eligible Policy Row

One eligible Repair attempt contributes exactly one row containing:

```text
one full-6D action sampled from the transaction's frozen pi_old
matching old mean, std, log probability, and value
FRS-GAIN-v007 G_total return and resulting advantage
transaction, policy snapshot, motion, Segment, attempt, scenario, and hash
K and valid evidence-step metadata
policy_sampled, policy_row_valid, finite/reset-valid facts
```

K is executable-evidence horizon only. It never duplicates a row. Clean,
Noisy, search, oracle, manually edited, invalid, or partial rows are excluded
from the policy domain. Missing or misaligned identity rejects the complete
transaction; it is not zero-filled.

## Scalar Actor And Critic Signal

For Segment `s` with attempts `m=1..M`:

```text
return_sm       = G_total_sm
value_target_s  = mean_m(G_total_sm)
advantage_sm    = return_sm - V_old(s)
```

The Actor surrogate uses every attempt's complete Recovery-Aware advantage.
The value loss predicts the Segment mean once per same-state group; it does not
pretend that the state value should equal every sampled action outcome. The
shared baseline subtraction preserves all strict within-Segment return
orderings. No Contact/ZMP/survival constraint
advantage, scalar fallback, projection coefficient, KKT status, replay
priority, or best-of-M score may independently modify the actor or Critic loss.

## Grouped Equal-Mass Reduction

Let `G` be represented motions, `S_g` the valid selected Segments for motion
`g`, and `M_gs` the eligible Repair attempts for Segment `s`:

```text
L_actor = mean_g [
            mean_s [
              mean_m [ clipped_ppo_surrogate(A_hat_gsm) ]
            ]
          ]
```

Consequences:

- each represented motion receives equal outer mass;
- each selected Segment receives equal mass within its motion;
- every eligible attempt receives equal mass within its Segment;
- K, evidence length, attempt index, Gain magnitude, and sampler priority do not
  duplicate, divide, or multiply row mass;
- all valid attempts remain, including adverse candidates and the least harmful
  direction when every attempt has negative Gain.

Winner-only, argmax, top-k, best-of-M weighting, score-proportional mass, and
flat legacy batch mean are forbidden.

## Sign-Preserving Advantage Scaling

For each Segment group:

```text
r_gs  = RMS(A_gsm over eligible attempts)
r_txn = RMS(A over the complete eligible transaction)
d_gs  = max(r_gs, r_txn)
A_hat_gsm = A_gsm / d_gs
```

The finite detached denominator prevents a high-scale Segment from dominating
without mean-centering or changing the sign of a nonzero advantage. This is
scale normalization only; it is not a winner rule or a second reward.

## Warmup Weight Boundary

FRS-TRAIN-v016 provides one actor-loss weight `w`:

```text
critic_only: w = 0
actor_ramp:  0 < w < 1
joint:       w = 1
```

The scalar actor gradient is multiplied by `w`; Critic learning remains active
in every phase. During critic-only, actor/std parameters and their optimizer
state must remain exactly unchanged. No projection or recovery direction is
computed before applying the ramp.

## Transaction And Exact-One Update

Collection uses one frozen old policy and performs zero optimizer steps. The
batch adapter may open only after every selected Segment has exact M attempts,
both shared baselines are sealed, metadata are homogeneous, and all required
evidence is finite or semantically `N/A`.

One grouped actor loss and one Segment-mean value loss are installed into their
disjoint parameter sets. Their gradient norms are measured and clipped
independently at the configured threshold, followed by exactly one optimizer
call. A partial or
mixed transaction performs no update and cannot advance optimizer state,
sampler state, curriculum, absolute iteration, or committed receipt.

## Sampling Boundary

Replay priority and perturbation curriculum may select future scenarios. They
are detached selection facts only. They may not enter advantages, grouped mass,
actor-loss multipliers, Critic targets, or the current transaction after it
opens. Segment Replay ranking is expressed solely by the different valid
advantages produced by FRS-GAIN-v007.

## Legacy Isolation

The following FRS-PPO-v004 objects are retired from the active route:

```text
constraint advantages and grouped constraint gradients
active constraint set
projection/recovery solver and dual values
KKT pre/postconditions
Actor/std restore caused by no-common-descent status
projection schema in checkpoint identity
```

They may remain only in history or an explicitly isolated ablation that cannot
write active storage, losses, optimizer state, checkpoints, or active-looking
diagnostics. Near-zero weights do not count as isolation.

## Required Diagnostics

- raw and scaled `G_total` advantage, value, return, and value error;
- `r_gs`, `r_txn`, sign-flip count, and finite denominator facts;
- represented motion/Segment/attempt counts and each mass share;
- policy row count, active K/M, and excluded-row reasons;
- actor/std/Critic parameter deltas and optimizer step delta;
- separate Actor/Critic pre/post-clip norms and clip coefficients;
- shared Segment value, exact-M mean target and Segment value error;
- transaction/scenario/hash/policy snapshot identity and committed receipt;
- `optimization_contract_id=FRS-PPO-v006`, state-value identity and no active projection schema.

FRS-GAIN-v007 raw Intent and Physics diagnostics remain observable but read-only
to this optimization owner.

## Required Evidence And Stop Conditions

Deterministic evidence must prove hand-computed grouped reduction, permutation
invariance, sign preservation, K-row isolation, adverse-row retention,
winner/priority isolation, shared-baseline ordering preservation, exact-M mean
value target, separate clipping isolation, critic-only actor freeze,
partial/mixed rejection, and exact-one optimizer update. Formal connectivity
must prove every active row uses the same FRS-GAIN-v007 value.

Stop if an old projection/KKT path builds a live graph, valid attempts are
winner-filtered, Gain or priority changes row mass, mean-centering flips credit,
K creates policy rows, critic-only mutates actor/std or their optimizer state,
the Critic sees the 6D action, the value target is action-conditioned, one
gradient family sets the other's clip coefficient, or one sealed transaction
produces anything other than one optimizer call.
