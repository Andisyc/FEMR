# FRS-METHOD-v021 Proposal: Outer Prioritized Scenario Replay

Status: confirmed and activated as FRS-METHOD-v021 / FRS-TRAIN-v020
Date: 2026-08-10
Workflow stage: historical design proposal; implementation authority moved to active Contracts
Affected candidate contracts: `FRS-METHOD-v021`, `FRS-TRAIN-v020`

## Problem

The active route implements the inner replay transaction: two Segments are
sealed, exact M actions are sampled from one frozen `pi_old`, and one grouped
update is committed. It does not close the outer replay loop that revisits a
previously valuable local scenario under a later policy.

This leaves most globally selected states with one exact-M target and one
Critic update. The configured global/replay/review fractions therefore do not
by themselves guarantee that the same learning problem is revisited. A raw
positive-Gain priority is also the wrong object: a negative-Gain scenario can
be highly valuable when the Critic predicts it incorrectly.

The missing mechanism is scenario selection, not another PPO replay buffer.
Old actions, log probabilities, returns and advantages must remain ineligible
after the policy changes.

## External Scaffold And FEMR Adaptation

Prioritized Level Replay maintains seen/unseen identities, updates a
policy-dependent learning-potential score from fresh rollout evidence, samples
seen levels by transformed score, and mixes staleness into replay probability.
FEMR adapts that scheduling structure only:

```text
PLR level                 -> sealed FEMR local Scenario
fresh level rollout       -> fresh exact-M rollout under current pi_old
value-L1 learning score   -> mean absolute utility advantage
level staleness           -> committed visits since last Scenario replay
```

PHC-style progressive hard-sequence mining is retained only as supporting
rationale. Its new-primitive training stages are not copied because FrontRES
must preserve the existing Actor, Critic and frozen-GMT architecture.

Primary sources:

- https://proceedings.mlr.press/v139/jiang21b.html
- https://github.com/facebookresearch/level-replay
- https://github.com/ZhengyiLuo/PHC

## Candidate Design

### FRS-DP-02 / SR-01: Two Replay Layers

The two layers have different owners and must not be conflated:

```text
inner exact-M replay
 -> within one frozen-policy Transaction
 -> estimate the current policy's utility distribution for one Scenario

outer prioritized Scenario replay
 -> between committed Transactions
 -> choose which sealed learning problem to revisit under the current policy
```

Outer replay never causes an extra optimizer step. Every selected Transaction
still contains exactly two Scenarios, exact M attempts per Scenario and exactly
one grouped Adam step.

### Stable Replay Item

The replay item is a sealed local Scenario rather than a bare Segment index:

```text
ScenarioKey = {
  motion_id,
  start_frame,
  x_t_identity,
  perturbation_family,
  perturbation_strength,
  perturbation_seed_or_artifact_hash,
  noisy_segment_hash,
  K,
  future_intent_identity,
  planned_support_identity,
}
```

Revisiting a bare `segment_id` while resampling DR does not revisit the same
Critic state. Transaction IDs remain visit identities and are not part of the
stable ScenarioKey.

### Admission And Learning Value

Every complete, finite, valid committed Scenario visit is recorded in the seen
table. Invalid reset, malformed evidence and partial exact-M collection do not
enter any pool.

For one valid Scenario under frozen `pi_old`:

```text
learning_value(s) = mean_m abs(U(G_m) - V_old(s))
```

This is a sampling score, not a new reward or Actor loss. It gives replay value
to both positive and negative Gain cases. When the Critic already predicts the
Scenario and all actions have small utility advantages, the score falls. Raw
Gain magnitude is not clamped to its positive part and does not directly set
sampling probability.

The committed score is an EMA across visits. Replay probability uses rank of
the score rather than raw magnitude, so one outlier cannot dominate the pool.

### Global, Replay And Review Selection

At the start of each Transaction, each of the two distinct Scenario sources is
chosen from the existing fixed mixture:

```text
global = 0.40
replay = 0.50
review = 0.10
```

- `global` discovers a new valid Segment and seals one current-K Scenario.
- `replay` samples a seen, unsolved Scenario by rank-transformed learning value
  mixed with staleness.
- `review` samples a solved or low-value stale Scenario to detect forgetting.

An empty replay or review pool falls back to global discovery. Selection is
probabilistic; the highest-ranked Scenario is not repeated exclusively. Every
valid global Segment retains nonzero discovery probability.

### Fresh On-Policy Recollection

When a Scenario is replayed, the current Actor is frozen as a new `pi_old` and
fresh exact-M actions are sampled. Clean and fixed Noisy evidence are executed
or rematerialized under the same sealed Scenario identity required by the
active transaction. Old action rows, old log probabilities, old returns and
old advantages are never reused.

Every valid fresh attempt retains equal grouped PPO mass. Replay priority does
not become a loss weight and does not introduce winner-only, argmax or
best-of-M credit.

### Commit, K And Persistence Boundary

The candidate replay-state delta is computed from sealed pre-update evidence,
validated before mutation, and committed only with a successful exact-one
Transaction receipt. A rejected or partial Transaction changes neither seen
state, score, staleness, pool membership nor RNG state.

Scenario records are retained across K transitions, but learning value is
K-specific because changing K changes the return object. A prior-K score cannot
directly rank the new-K replay pool. The new persistence identity must restore
ScenarioKey records, per-K scores, staleness, pool membership and sampler RNG
exactly.

## Preserved Behavior

- Actor remains the deployable 158D full-6D Repair policy.
- Critic remains the 449D support-conditioned scalar state value.
- `U(G)=sign(G)log1p(abs(G))`, exact-M mean target and per-attempt advantage are
  unchanged.
- M remains 4 and each Transaction still selects two distinct Scenarios.
- K8/M4 -> K16/M4 -> K32/M4 and per-K DR remain unchanged.
- Gain, PPO reduction, split LR, gradient clipping, GMT and simulator remain
  unchanged.
- No stale PPO transition replay, second optimizer, second Critic or MOSAIC host
  change is introduced.

## Falsifiable Predictions

1. After the replay pool is seeded, committed logs contain repeated stable
   ScenarioKeys with new transaction and policy-snapshot identities.
2. Replaying one Scenario preserves its x_t, artifact, hash, K, future Intent
   and support identity while producing fresh actions and log probabilities.
3. Negative-Gain Scenarios with high absolute utility error can enter replay.
4. Visit-by-visit Critic error falls on learnable repeated Scenarios; a score
   remains high only while value error or action-dependent utility spread
   remains high.
5. Failed Transactions produce zero sampler-state delta.
6. A K transition cannot use a prior-K score as the active new-K priority.

## Activation Receipt

The complete `FRS-DP-02` decision was confirmed by the 2026-08-10 one-shot
execution authorization. Coordinated authority is now
`FRS-METHOD-v021 / FRS-TRAIN-v020 / checkpoint-v15`. This proposal remains
historical rationale and no longer owns implementation or training semantics.
