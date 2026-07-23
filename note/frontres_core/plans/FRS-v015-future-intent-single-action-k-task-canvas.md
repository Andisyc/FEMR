# FRS-v015 Critic Curriculum Task Canvas

Status: active volatile control surface. Updated: 2026-07-23.

## Objective

Keep one 6D actor and one scalar Critic while making K curriculum target
stationary within each stage: one global K per transaction, Critic recalibration
after every K transition, then actor ramp and joint PPO at that same K.

## Method Authority

- Concept Figure: `M-06 K-step Curriculum` -> `M-05 Actor & Critic Warmup`
- Method: FRS-METHOD-v015
- Training: FRS-TRAIN-v009
- Gain/PPO: FRS-GAIN-v004 / FRS-PPO-v003 unchanged

## Current Cursor

`C1-C4 engineering complete; post-C4 policy-quality decision blocks long training`

## Confirmed

- Multi-Critic is rejected because it separates baselines but does not repair a
  truncated actor target.
- K values are ordered curriculum approximations toward one final horizon, not
  simultaneous objectives.
- Same transaction must not mix K.
- Critic parameters continue across K; actor/std freeze during each new-K
  recalibration.
- Stage transitions occur only after a committed transaction.
- E-FI-71 live-confirms the K8 critic-only -> actor-warmup -> joint -> K16
  critic-only transition, actor/std isolation, exact-one updates and v4/v009
  persistence.

## Closed Engineering Mismatch

- formal transaction planning now overrides legacy per-Segment K with the one
  active v009 stage K;
- phase resolution is stage-local and re-enters critic-only after K changes;
- checkpoint v4 binds the exact schedule fingerprint and K-stage identity;
- v008/checkpoint-v3 and mismatched schedules reject before mutation.

## Exposed Quality Blocker

- 0/16 Repair and Noisy rows were Physics-admissible;
- 0/16 final Gains were positive, while 3/16 raw advantages were positive;
- one-update critic-only phases prove isolation, not Critic calibration;
- when both roles share the same saturated worst Physics violation, v004 can
  preserve non-compensation yet lose useful within-unsafe-tier direction.

## Active Steps

```text
C0 contract/plan/Architecture
-> C1 pure schedule/config
-> C2 homogeneous-K formal transaction
-> C3 v009 persistence/resume
-> C4 bounded official K transition
-> policy-quality audit at final K
```

## Non-Scope

Multi-Critic, K actor input, new module, second optimizer, Gain/PPO/HSL changes,
long training, multi-seed, deployment composition, and paper experiments.

## Next Action

Do not start long training. First perform a bounded, read-only causal audit of
which Contact/ZMP/survival component saturates each of the 16 C4 rows and
whether v004 hides paired improvement at the worst-component reduction. Then
the user chooses evaluator refinement, sampling/curriculum adjustment, or only
longer Critic calibration.
