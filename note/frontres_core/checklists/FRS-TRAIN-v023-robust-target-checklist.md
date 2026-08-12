# FRS-TRAIN-v023 Robust Target Checklist

## Authority

- [x] Option B confirmed by the user.
- [x] DP02/DP09 Design Inspector projection updated.
- [x] METHOD-v024 / TRAIN-v023 / PPO-v011 / EVAL-v005 activated.
- [x] Governance activation receipt validated.
- [x] Engineering Plan review returns `READY`.

## Implementation

- [x] Replay owns bounded compatible utility-window state and uncertainty.
- [x] PPO consumes exactly eight robust targets without historical Actor data.
- [x] Formal Unit of Work previews before PPO and commits after exact-one.
- [x] Telemetry exposes target, variance, SE/h95, KL/reset and counts.
- [x] checkpoint-v18/replay-v4 roundtrip and v17 rejection are strict.
- [ ] EVAL-v005 held-out manifest/runner accepts v18 without mutating Replay.
  This is post-checkpoint evaluation work and does not block TRAIN-v023 startup;
  the current `policy_quality_eval` entrypoint remains historical EVAL-v004.

## Evidence

- [x] TEST-24A through TEST-24D pass through production public boundaries.
- [x] Controlled counterexamples prove test sensitivity, including policy-reset
  capacity maturity and historical-Actor target contamination.
- [x] Affected regression suite and `py_compile` pass: 58/58 Contract markers.
- [x] Construction and final code reviews contain no open P0/P1.
- [x] Formal route audit identifies exactly one live-only K8 fact.

## Terminal Gate

- [x] Publish one bounded official K8/B8/M4 command and stop.
- [x] Do not start bounded execution or long training.
