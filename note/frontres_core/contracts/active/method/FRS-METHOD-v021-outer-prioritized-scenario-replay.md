---
contract_id: FRS-METHOD-v021
status: active
effective_date: 2026-08-10
updated_date: 2026-08-10
supersedes: FRS-METHOD-v020
scope: Recovery-Aware exact-M FrontRES training with committed outer prioritized sealed-Scenario replay
---

# Outer Prioritized Scenario Replay

## Design Delta

FRS-METHOD-v020 established the 158D Actor, 449D state-value Critic, symmetric-log
utility, exact-M transaction, Recovery-Aware Gain and frozen GMT. Those semantics
remain unchanged. This Contract closes the missing outer replay loop: later
transactions may revisit the same sealed local Scenario under the current
frozen old policy.

The two replay layers have different meanings:

```text
inner exact-M replay
  = estimate one Scenario's current-policy utility distribution

outer prioritized Scenario replay
  = decide which previously observed learning problem to revisit
```

Outer replay changes scenario scheduling only. It never reuses old actions,
log probabilities, returns or advantages and never adds an optimizer step.

## Stable Scenario Identity

The replay item is a validated immutable ScenarioKey:

```text
motion_id, start_frame, segment_id, x_t_identity
perturbation_family, perturbation_strength, perturbation_seed
noisy_segment_hash, K
future_intent_identity, planned_support_identity
```

Transaction and policy-snapshot IDs identify visits and are not part of the
stable key. A bare Segment ID is insufficient because resampling corruption
creates a different Critic state.

FrontRES owns the perturbation seed and isolates the materializer RNG while the
frozen MOSAIC command remains unchanged. Reusing the complete key must reproduce
the same root artifact and noisy hash. The current policy still samples fresh M
Repair actions after materialization.

## Admission And Priority

Only a complete, finite, valid exact-M Scenario visit may enter the seen table.
The pre-update learning value is:

```text
L_K(s) = mean_m abs(U(G_m) - V_old(s))
```

where `U(G)=sign(G)log1p(abs(G))`. This value is a scheduler signal only. It
does not change Gain, the Critic target, Actor advantage, PPO row mass or loss.
Negative-Gain attempts are eligible when their absolute value error is high.

Each Scenario stores a committed EMA score per K. Replay probability uses score
rank plus committed staleness, not raw score magnitude. A K8 score cannot rank
the K16 or K32 pool.

## Selection

Each formal transaction selects two distinct Scenario identities:

```text
global discovery  0.40
priority replay   0.50
stale review      0.10
```

Global discovers a valid Segment and seals a current-K Scenario. Replay selects
a seen current-K Scenario by rank-transformed score plus staleness. Review
selects a low-score or solved stale current-K Scenario to detect forgetting.
An empty replay or review pool falls back to global discovery. Distinctness is
enforced before collection.

## Transaction And Persistence

The replay update candidate is computed and validated from sealed pre-update
evidence. It mutates replay state only after the exact-one optimizer commit has
a matching committed receipt. Rejected, malformed or partial collection changes
no Scenario record, score, staleness, pool membership or replay RNG.

Checkpoint-v15 strictly persists Scenario records, per-K score and visit state,
staleness, pool classification and sampler generator state. Checkpoint-v14 and
earlier are incompatible with the active training route.

## Preserved Boundaries

- Actor remains the deployable 158D full-6D direct Delta SE(3) policy.
- Critic remains the 449D support-conditioned scalar state value `V(s)`.
- FRS-GAIN-v008, FRS-PPO-v008, symmetric-log utility and exact-M mean target remain.
- M=4 and K8 -> K16 -> K32 with the existing per-K DR curriculum remain.
- Each transaction has two Scenarios, equal attempt mass and one Adam step.
- Split learning rates, separate gradient clipping, GMT and simulator remain.
- No stale PPO buffer, best-of-M weighting, action-conditioned Critic, second
  optimizer, second Critic or MOSAIC host change is permitted.

## Falsifiers

- A replay visit changes x_t, perturbation artifact/hash, K, future intent or
  planned support identity.
- A replay visit reuses an old policy row.
- Negative Gain is excluded despite high absolute value error.
- A failed transaction changes replay state or RNG.
- Prior-K score changes current-K replay probability.
- Save/resume silently drops or reconstructs replay identity.
