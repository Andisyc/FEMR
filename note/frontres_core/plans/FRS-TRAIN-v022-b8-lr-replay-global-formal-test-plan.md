# FRS-TRAIN-v022 B8/LR/Replay One-Shot Engineering Plan

Status: active
Date: 2026-08-11
Contracts: FRS-METHOD-v023 / FRS-GAIN-v008 / FRS-PPO-v010 / FRS-TRAIN-v022
Terminal outcome: test-ready implementation and one user-run command; no test or training launched

## Engineering boundary record

Requested behavior: replace Actor loss scaling with actual parameter-group LR,
increase one formal update to B8/M4, bound Replay breadth with a visit-gated
curriculum, and provide one production-owner Global Simplified Formal Test.

Preserved behavior: 158D full-6D Actor, 449D `V(s)`, raw Gain-v008, per-attempt
symlog, exact-M mean target, K8/16/32, four DR classes, one Adam, separate 0.5
clipping, frozen GMT and simulator ownership.

Owners and boundaries:

- `frontres_segment_warmup.py`: committed iteration -> phase, DR and actual Actor LR;
- `frontres_outer_scenario_replay.py`: B8 slot purposes, active/archive/capacity preview and atomic commit;
- `frontres_segment_formal_transaction.py`: verifies B8/M4, installs group LR and owns exact-one update;
- `frontres_checkpointing.py`: strict checkpoint-v17 persistence;
- official Stage-3 composition: selects the deterministic external adapter only
  for the Global Simplified Formal Test; Gain/PPO/replay/optimizer are never faked.

Forbidden: MOSAIC host changes, Q(s,a), action-conditioned Critic, stale policy
rows, second optimizer, winner weighting, Gain change, K16+ test transition,
test-only loss/target/replay implementations, or automatic server/long run.

## Implementation batches

1. Activate the Design Inspector and Contract delta; mark v021 evidence stale.
2. Implement actual LR identity and B8 typed transaction boundaries.
3. Implement replay-v3 active/archive capacity, quotas, slot schedule,
   visit-gated expansion and replacement.
4. Implement checkpoint-v17, telemetry, CLI/config and strict rejection.
5. Add one production-owner Global Simplified Formal Test and manifests; stop
   before execution so the user runs the supplied command.

## Confirmed Module Test Cards

### TEST-23A Actual Actor LR

Boundary: schedule owner plus named optimizer-group installation.
Independent answer: at low-init/ramp-end/joint and K transition, Actor LR is
`3e-7`, `1e-6`, `1e-6`, `3e-7`; Critic LR is always `1e-5`; loss weight is one.
Falsifier: only loss magnitude changes or group LR remains at the joint value.

### TEST-23B B8/M4 transaction

Boundary: typed transaction shape and grouped PPO public boundary.
Independent answer: eight distinct Scenario states, four attempts each, 32
policy rows, 64 role rows, eight exact-M Critic targets and one update.
Falsifier: row count is treated as unique Critic states or two-Scenario logic remains.

### TEST-23C Replay Curriculum

Boundary: `FrontRESOuterScenarioReplay.plan/stage/commit`.
Independent answer: warmup slot purposes 1/6/1, joint 1/4/2/1; 64->128->256
only at joint after min visit 4; quotas preserved; evicted record remains archived.
Falsifier: fixed 40/50/10, simultaneous DR/capacity growth, missing joint E_V,
or failed commit mutation.

### TEST-23D Checkpoint-v17

Boundary: strict save/load/resume.
Independent answer: exact model/optimizer/normalizer/K/DR/LR/replay-v3 active
membership, archive, capacity, RNG and receipt roundtrip; v16 rejects before mutation.

### TEST-23E Global Simplified Formal Test

Boundary: dedicated bounded entry using the production FrontRES model, PPO,
optimizer and Replay owners, with only simulator evidence replaced. K8/B8/M4
and raw-Gain boundary->symlog->PPO->Adam->Replay are retained. This test does not
by itself prove IsaacLab entrypoint wiring; that remains the subsequent bounded
official transaction gate. Test pool ladder is 8->16->24, 32 transactions use phases 4/4/24,
tx8 saves, tx9 resumes through the simplified persistence carrier, and one final
invalid transaction proves zero mutation. Production checkpoint-v17 readback is
reserved for the subsequent bounded official transaction.
Any K16/K32/K64 transition, fake semantic owner or more than 33 transaction
attempts falsifies the test.

Human status: confirmed by the user's 2026-08-11 instruction to plan and execute
the previously reviewed Global Simplified Formal Test up to, but not including,
test execution.

## Stop condition

Stop with one command after code, manifests and static review are ready. Do not
run module tests, the global formal test, simulator, server transaction or long training.
