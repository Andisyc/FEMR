# FRS-TRAIN-v024 Current-Visit Target Checklist

## Authority

- [x] User confirmed deleting cross-visit numerical Experience Replay while preserving outer Scenario selection and fresh current Actor reruns.
- [x] METHOD-v025 / TRAIN-v024 / PPO-v012 / EVAL-v006 activated.
- [x] TEST-25A through TEST-25D confirmed by the same human semantic decision.
- [x] Engineering Plan Review: READY; local reversible edits authorized.

## Implementation

- [x] Replay target is exactly the current row-aligned M4 mean.
- [x] Policy anchor/KL/window/reset and historical utility state are absent.
- [x] Latest current-policy E_V/E_A, lifetime visits and staleness drive selection only.
- [x] PPO Actor credit, B8/M4, LR, Gain, K/DR and exact-one update are unchanged.
- [x] checkpoint-v19/replay-v5 roundtrip is strict and v18/v4 rejects.

## Evidence

- [x] TEST-25A..D pass at their production public boundaries.
- [x] Controlled pre-fix history-contamination case fails before the fix and passes after it.
- [x] Affected regression suite and py_compile pass.
- [x] Construction/final review contains no open P0/P1.
- [x] Formal offline route reaches target consumer and one exact commit.

## Terminal gate

- [x] Stop before server/simulator execution and long training.
- [x] Report the single remaining live-only K8 fact without claiming policy quality.
