# FRS-TRAIN-v019 Symmetric-Log Utility Checklist

Status: offline/module/formal Phase A passed; bounded Phase B ready
Date: 2026-08-10
Plan: `../plans/FRS-TRAIN-v019-symmetric-log-utility-one-shot-engineering-plan.md`

## Governance

- [x] Human confirmed fixed per-attempt signed-log utility for both Actor and Critic.
- [x] Activate METHOD-v020 / GAIN-v008 / PPO-v008 / TRAIN-v019 / checkpoint-v14.
- [x] Preserve raw Gain, 158/449/770 inputs, full-6D action, M4, K/DR, beta and split LR.
- [x] Update Design Inspector/Register, registry and derived Architecture.

## Module Gates

- [x] TEST-13 preserves raw Gain and rejects invalid transform input.
- [x] TEST-14 proves raw return plus utility-space carried advantage.
- [x] TEST-15 proves transform-before-M4-mean and shared Actor/Critic utility.
- [x] TEST-16 proves checkpoint-v14 roundtrip and v13 pre-mutation rejection.
- [x] TEST-17 proves read-only EVAL-v004 utility calibration with raw reporting.
- [x] TEST-18 proves formal raw/utility telemetry and exact-one commit.

## Verification Gates

- [x] Focused tests demonstrate red-green sensitivity and pass.
- [x] Python compile, affected and aggregate contract suites pass (55/55).
- [x] Atlas JSON/HTML validators and contract sentinel pass.
- [x] Construction review has no open P0/P1; one P2 identity duplication was resolved.
- [x] Official offline transaction and checkpoint-v14 persistence pass.
- [ ] Git push/pull identities match.
- [ ] One bounded official transaction passes under the exact server identity.
- [ ] Formal long-training manifest and one-shot long-run gate pass.
- [ ] Fresh HSL-v2 K8/M4 cold-start training begins in a new output directory.

## Stop Conditions

Stop on raw-evidence loss, wrong transform placement, Actor/Critic utility
mismatch, non-finite values, checkpoint identity drift, partial transaction
mutation, failed formal receipt, or any bounded-live invariant failure.
