# FRS-TRAIN-v018 Support-Conditioned M4 Checklist

Status: bounded live gate passed; K8/M4 cold-start training running
Date: 2026-08-09
Plan: `../plans/FRS-TRAIN-v018-support-conditioned-m4-one-shot-engineering-plan.md`

## Governance

- [x] Human confirmed action-pre support conditioning and M4 at every K.
- [x] Activate METHOD-v019 / TRAIN-v018 / checkpoint-v13; archive v018/v017.
- [x] Preserve Actor/GMT/Gain/PPO/split-LR/K-DR/full-6D semantics.
- [x] Define one physics Gateway, one deterministic layout owner, and forbidden
  action-dependent fields.

## Module Gates

- [x] TEST-02 proves K8/M4 -> K16/M4 -> K32/M4 and rejects old schedules.
- [x] TEST-05 proves exact 102D support ordering, 449D composition, row identity
  and fail-closed boundaries through the real observation consumer.
- [x] TEST-15 proves arithmetic M4 target and per-attempt Actor credit unchanged.
- [x] TEST-16 proves checkpoint-v13 exact roundtrip and v12 pre-mutation reject.
- [x] TEST-18 proves official owner-produced support/M4 telemetry.

## Verification Gates

- [x] Python compile and focused semantic tests pass.
- [x] Affected and aggregate contract suites pass.
- [x] Design Inspector JSON/HTML validation and contract sentinel pass.
- [x] Construction review has no open P0/P1 and records any scoped P2.
- [x] Formal Runtime Audit Phase A proves config -> observation -> PPO ->
  checkpoint -> telemetry connectivity.
- [x] Git push/pull identities match at server commit `0d8e412`.
- [x] One bounded official transaction proves 16 rows, 449D finite input,
  critic-only isolation, nonzero Critic delta, exact-one step and checkpoint-v13.
- [x] Fresh HSL-v2 cold-start long training started only after the bounded gate.

Live receipt: GPU 0 bounded log
`/hdd0/yuxuancheng/FEMR/log/FRS_TRAIN_V018_SUPPORT_M4_SENTINEL_R2.log`
contains exactly one each of `AUDIT-B01..B08`, zero traceback/runtime/OOM/assertion
matches, and a successful checkpoint-v13 readback of `model_1.pt`. The fresh
K8/M4 run is PID `3372457`, log
`/hdd0/yuxuancheng/FEMR/log/FRS_TRAIN_V018_K8_M4_FULL_COLDSTART_20260809.log`.

## Stop Conditions

Stop on action/outcome leakage, silent padding without mask, Actor/GMT/Gain
change, schedule drift, old-checkpoint acceptance, transaction mutation before
commit, failed finite/identity/atomic checks, or missing live receipt.
