# FRS-TRAIN-v023 Robust Target One-Shot Engineering Plan

Status: human-confirmed Option B; implementation authorized; training must not
start in this unit

Date: 2026-08-12

Contracts: FRS-METHOD-v024 / FRS-GAIN-v008 / FRS-PPO-v011 /
FRS-TRAIN-v023 / FRS-EVAL-v005 / FRS-ENG-v001

Terminal outcome: production owners, persistence and official entrypoint are
offline-closed; publish one bounded official K8/B8/M4 command and stop.

## Engineering Boundary Record

Requested behavior: accumulate same-Scenario current-policy-compatible utility
evidence, use a robust expected-utility estimate for Critic supervision, and
rank calibration Replay by error outside the estimated mean confidence
interval.

Preserved behavior: 158D full-6D Actor, 449D `V(s)`, Gain-v008, per-attempt
symlog, current-attempt Actor advantage, B8/M4, K8/16/32, four DR classes,
actual Actor LR `3e-7 -> 1e-6`, Critic LR `1e-5`, one Adam, separate clipping,
frozen GMT and exact-one commit.

Owners and public boundaries:

- `frontres_outer_scenario_replay.py` owns one immutable bounded utility-window
  Value Object per Scenario, policy compatibility, robust target, uncertainty,
  replay score and candidate/commit persistence;
- `frontres_segment_ppo.py` consumes exactly eight detached Scenario targets;
  it does not know Replay windows or recompute statistics;
- `frontres_segment_formal_transaction.py` orders current utility -> Replay
  preview -> PPO -> exact-one -> Replay commit and publishes telemetry;
- `frontres_checkpointing.py` owns strict checkpoint-v18 save/load/rejection;
- existing entrypoint/config remains the Composition Root.

Dependency direction: transaction orchestration depends on Replay and PPO
public records. PPO never imports Replay. Replay never imports policy, runner,
checkpoint or simulator objects. Tests invoke production public boundaries and
may not implement a second PPO, Replay, optimizer or training route.

State boundary: Replay preview includes the current M4 but mutates nothing.
Only a matching exact-one receipt commits window, score, membership, capacity,
staleness and RNG. Incompatible checkpoint or partial transaction fails before
mutation. Windows retain at most 32 complete compatible M4 visits.

Legacy characterization and isolation: v023 Replay computes instantaneous
`abs(mean advantage)` after PPO and checkpoint-v17 stores replay-v3 without
window evidence. The Pinch Point is `FrontRESOuterScenarioReplay.stage`; the
existing plan/stage/commit seam is retained. v17/v3 reject rather than migrate.

Forbidden: old policy-row replay, action-conditioned or variance-head Critic,
median/max/min target, risk-sensitive Actor, target fallback, MOSAIC changes,
test-only composition roots, copied production formulas in an oracle, K16/K32
runtime, long training or automatic server work.

## Implementation Batches

1. Activate Inspector/Contract/registry and close stale Register identities.
2. Add the replay-owned window Value Object and change stage input/output.
3. Let PPO consume eight external robust targets and reorder the formal Unit of
   Work without changing Actor credit or update count.
4. Version checkpoint/config/telemetry/evaluation identities and reject v17.
5. Execute focused production-boundary cases, affected suite, construction and
   final review, then perform formal static/offline route audit.

## Confirmed Module Test Cards

### TEST-24A Robust expected-utility window

Boundary: `FrontRESOuterScenarioReplay.stage/commit`.

Independent answers: first M4 `[1,2,3,4]` targets `2.5`; a compatible second M4
`[5,6,7,100]` forms sorted eight samples, winsorizes one value per tail to
`[2,2,3,4,5,6,7,7]` and targets `4.5`; current M4 remains intact for Actor.
Changing one extreme cannot dominate the target. A failed commit changes no
window.

### TEST-24B Uncertainty-aware calibration priority

Boundary: the same Replay owner candidate.

Independent answer: `E_V_learn=max(abs(V-mu_hat)-1.96*SE,0)`. A high-variance
mean-matched Scenario reaches zero calibration priority, while a low-variance
wrong-value Scenario remains positive. Ranking by variance alone or raw current
M4 error falsifies the card.

### TEST-24C Policy compatibility and bounded lifecycle

Boundary: utility-window append/reset and replay-v4 state roundtrip.

Independent answer: identical diagonal Gaussians have symmetric KL zero and
append; a hand-separated mean exceeding KL `0.02` resets to the current M4 and
increments reset count; the fixed anchor does not drift. The 33rd compatible
visit drops exactly the oldest M4 batch. v3/v17 reject before mutation.
Capacity expansion requires four visits in every active current compatible
window; a large lifetime visit count cannot bypass this gate after reset.

### TEST-24D PPO and transaction ownership

Boundary: `compute_frontres_segment_ppo_loss` through the formal transaction.

Independent answer: eight keyed robust targets expand to four rows each and
drive Critic/normalizer; the 32 Actor advantages remain current
`U(G)-V_old`; row permutation is invariant; exactly one optimizer step and one
Replay commit occur. Missing/reordered target or optimizer failure fails closed.

## Proof Route And Stop

Run S1 owner cases first, then S2 transaction consumer, S3 replay/checkpoint
roundtrip and current affected tests. Do not use the Global Simplified Formal
Test as official evidence and do not add another simplified route. Formal audit
ends at `LIVE_REQUIRED` for one real IsaacLab K8 transaction. Stop before
executing it and give the user exactly one command, expected log path and
falsifiers.
