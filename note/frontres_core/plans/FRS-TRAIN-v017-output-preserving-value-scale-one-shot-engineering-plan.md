# FRS-TRAIN-v017 Output-Preserving Value Scale One-Shot Plan

```yaml
plan_id: FRS-TRAIN-v017-output-preserving-value-scale-one-shot
status: offline-ready
date: 2026-08-08
owners: [workflow-governance, frontres-segment-ppo]
```

## Decision And Boundary

The K8/M2 TRAIN-v016 log contains 2000 finite committed transactions, but the
Critic pre-clip gradient is clipped in 77.95% of them. Segment targets have a
median near zero while rare values reach `-414.77` and `1578.23`. The accepted
change addresses this numerical conditioning without changing the scientific
objective.

Requested behavior: fit the same exact-M Segment mean `G_total` target with an
adaptive, non-amplifying Critic residual scale. Raw `G_total`, raw `V(s)`, Actor
advantages, Actor loss, value clipping in raw units, and all return orderings
remain unchanged.

Single owner: `frontres_value_normalization.py` owns immutable normalizer state
and candidate statistic calculation; `frontres_segment_ppo.py` owns applying
the resulting scale to its value loss. The formal transaction commits the
candidate only after exactly one successful Adam step. Checkpointing owns
strict persistence transport, not normalization math.

Public input/output: two finite exact-M Segment means plus prior scalar state
`(mean, second_moment, update_count)` produce one candidate state and a finite
scale `>= 1`. The loss consumes `raw_MSE / scale^2`; telemetry and all policy
interfaces continue to expose raw target and raw value units.

Dependency direction: active config -> PPO functional owner -> formal
transaction commit -> checkpoint/telemetry projection. Gain, Actor, simulator,
GMT, sampler, curriculum and evaluation may not depend on or mutate the
normalizer.

State and persistence: one candidate update per sealed transaction, committed
only with the optimizer transaction, then saved under checkpoint-v12. Resume
must validate the complete state before any policy, optimizer, sampler or
normalizer mutation. Checkpoint-v11 and malformed/missing v12 state reject.

Legacy behavior: FRS-PPO-v006 / TRAIN-v016 / checkpoint-v11 remain historical
evidence. There is no compatibility conversion or partial resume.

Hotspot decision: the existing PPO loss is the WELC Pinch Point and the formal
transaction is the Unit-of-Work boundary. Adding a wrapper, service, new
optimizer or Critic module would increase caller knowledge, so none is added.

Stop if Actor loss/advantages change, raw value output changes, the scale can
amplify gradients or become non-finite, statistics advance on a failed or
read-only path, v11 resumes, or any test requires Gain/simulator changes.

## Coarse Execution

1. Version the optimization/training/checkpoint authority and confirm the four
   affected Module Test Cards.
2. Add deterministic normalizer and loss tests, then implement the smallest
   owner/commit/persistence/telemetry path that satisfies them.
3. Run focused and affected regressions, construction review, formal runtime
   Phase A, and stop before long training.

## Module Test Cards

- `TEST-15 Segment PPO`: ordinary, extreme, permutation, zero-variance and
  invalid-state cases prove raw targets/Actor facts are invariant while the
  Critic loss scale is finite and non-amplifying.
- `TEST-16 Checkpointing`: checkpoint-v12 round-trips exact statistics and
  rejects v11, missing, malformed, non-finite or inconsistent state before
  mutation.
- `TEST-18 Runtime Diagnostics`: final telemetry carries owner-produced
  normalization identity, pre/post statistics and committed update count
  without recomputation or feedback.
- `TEST-02 Training Config`: the official Stage-3 composition explicitly fixes
  normalization identity, EMA decay `0.9`, and scale floor `1.0`; alternate or
  partial identities fail closed.

Independent oracles are hand-calculated exponential moments, raw/scaled MSE,
parameter snapshots, permutation equality, and atomic checkpoint sentinels.
