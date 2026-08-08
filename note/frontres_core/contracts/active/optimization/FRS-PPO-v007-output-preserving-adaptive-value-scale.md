---
contract_id: FRS-PPO-v007
status: active
effective_date: 2026-08-08
updated_date: 2026-08-08
supersedes: FRS-PPO-v006
scope: equal-motion, equal-Segment, equal-attempt grouped PPO with one shared future-conditioned state value per Segment, exact-M mean Critic targets, output-preserving adaptive Critic loss scaling, independent Actor/Critic gradient clipping, and exactly one optimizer update per sealed transaction
---

# Grouped PPO With Output-Preserving Adaptive Critic Scale

## Design Delta

FRS-PPO-v006 established the future-conditioned scalar state-value Critic,
one exact-M mean target per Segment and separate Actor/Critic clipping. The
completed TRAIN-v016 K8/M2 run remained finite but exposed a numerical scale
problem: Segment targets were concentrated near zero with rare values above
`10^3`, and the Critic pre-clip gradient hit its clip boundary in most
transactions.

FRS-PPO-v007 preserves every raw prediction, target, advantage and Actor fact
from v006. It adds one non-amplifying adaptive scale to the Critic loss only.
This remains PPO with a state-value baseline; the 6D action never enters the
Critic, `G_total` is not transformed, and no Q-learning authority is introduced.

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

## Output-Preserving Adaptive Value Scale

Let `y_s` be the two exact-M Segment-mean targets in one complete transaction.
Starting from committed state `(mu, nu, n)`, the PPO owner previews:

```text
mu'    = 0.9 * mu + 0.1 * mean_s(y_s)
nu'    = 0.9 * nu + 0.1 * mean_s(y_s^2)
sigma  = max(1.0, sqrt(max(0, nu' - mu'^2)))

L_value_raw    = grouped mean of the existing clipped value MSE
L_value_scaled = L_value_raw / sigma^2
```

The initial committed state is `(0, 1, 0)`. The scale is finite, detached,
permutation-invariant and never below one, so it can reduce an extreme Critic
gradient but cannot amplify an ordinary one. Raw `V(s)`, raw targets, value
clipping, Actor advantages and all user-facing diagnostics remain in original
`G_total` units. This is an output-preserving normalized-SGD/PopArt-style
conditioning rule, not a logarithm or other transform of Gain.

The preview state is immutable. The formal transaction may commit
`(mu', nu', n+1)` only after the exact-one optimizer step and committed receipt
succeed. Failed, partial, read-only and evaluation paths perform no state
advance. Checkpointing owns exact persistence of the committed state.

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

FRS-TRAIN-v018 provides one actor-loss weight `w`:

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
- raw and scaled value loss, normalization identity and finite scale;
- committed/preview target mean, second moment and update count;
- transaction/scenario/hash/policy snapshot identity and committed receipt;
- `optimization_contract_id=FRS-PPO-v007`, state-value identity and no active projection schema.

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
produces anything other than one optimizer call. Also stop if raw `V(s)`, raw
targets or Actor facts change; the scale is non-finite or below one; statistics
advance on a failed/read-only transaction; or checkpoint state does not match
the committed iteration.
