# FEMR Current Semantic Objects

Updated: 2026-07-13

## Observation Payload

```text
100D FrontRES prefix + 770D frozen-GMT suffix = 870D actor observation
```

Owners: `frontres_observation_layout.py`, `normalizer.py`,
`front_residual_actor_critic.py`, checkpoint/eval/export loaders.

Evidence: shape/order, prefix/suffix stats ownership, checkpoint identity, and
all sink dimensions.

## Full-6D Repair Action

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Owners: policy distribution, rollout step, task-space correction, Segment
storage, Segment PPO, checkpoint, eval.

Invariant: sampled action, old log-prob, old mean/sigma, stored action, and
executed repair share one representation. Perturbation family never narrows the
six dimensions.

## Segment Identity And Trial Role

One replay case owns motion id, start frame, perturbation, dynamic reset state,
source, trial role/index, horizon K, and rollout evidence.

Policy rows may reach PPO. Search/counterfactual rows may reach sampler evidence
but are invalid for PPO before the batch boundary.

## Effective Horizon K

Sampler states may assign `8/16/32/64`. K must survive batch construction,
quartet replication, reset, rollout accumulation, done masks, returns, sampler
evidence, diagnostics, and eval.

Implementation and formal-route integration are separate evidence claims.

## Gain Decomposition

```text
style_gain   = style_quality(Repaired | Clean) - style_quality(Noisy | Clean)
physics_gain = physics_quality(Repaired)       - physics_quality(Noisy)
repair_cost  = full6 magnitude + temporal change (+ valid Clean no-op cost)
gain_total   = w_style * style_gain + w_physics * physics_gain
             - w_repair * repair_cost
```

Owner contract:
`frontres_core/contracts/active/reward/FRS-GAIN-v002-style-physics-repair.md`.

Current code status: formal Segment live scoring is RP-only and therefore does
not implement this semantic object.

Required lifecycle:

```text
Clean/Noisy/Repaired paired states
-> shared raw components
-> shared normalization/scales
-> per-row K aggregation
-> PPO return + sampler evidence
-> periodic/sequence eval
-> decomposed diagnostics
```

## Rollout Transaction Identity
`audit_transaction_id` and `audit_batch_signature` are diagnostic evidence
metadata, not reward, action, or PPO features. One capture creates them from
the ordered row tuple `(segment_id, role, motion_id, start_frame, effective_K)`.
Cards 15/16/17 must preserve the same pair; storage rejects a mixed transaction.
Card 22 reports the set/count of transactions after update-loop aggregation and
must distinguish `single` from `aggregate`.

## PPO Tuple

The Segment PPO tuple contains full-6D action, old log-prob, old mean/sigma,
value, return, advantage, and valid mask. Old-policy tensors are detached.
Trial metadata and Gain diagnostics do not become new PPO action dimensions.

## Checkpoint State

Checkpoint identity includes policy architecture, observation stats, action
sigma, optimizer, Segment sampler, and any future persisted Gain scales. Load,
resume, evaluation, and export must agree on ownership and dimensions.

## Evaluation Evidence

Evaluation is independent of policy update and sampler mutation. It reports raw
Style metrics, raw Physics metrics, Repair cost, separate paired gains, total
Gain, motion identity, perturbation, K, reset/preroll, and action evidence.

Missing evidence is `UNCONFIRMED`, never zero.
