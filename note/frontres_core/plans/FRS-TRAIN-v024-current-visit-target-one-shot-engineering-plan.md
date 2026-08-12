# FRS-TRAIN-v024 Current-Visit Target One-Shot Engineering Plan

Status: human-confirmed semantic correction; local reversible implementation authorized; stop before server execution
Date: 2026-08-12
Contracts: FRS-METHOD-v025 / FRS-GAIN-v008 / FRS-PPO-v012 / FRS-TRAIN-v024 / FRS-EVAL-v006 / FRS-ENG-v001
Terminal outcome: activate the current-visit target, close module/persistence/official-offline evidence, and stop at the next live-only K8 boundary.

## Engineering Boundary Record

Requested behavior: retain outer Scenario selection, but rerun every selected
Scenario with current Actor M4 and supervise Critic only with that transaction's
M4 mean. Remove cross-visit numerical Experience Replay.

Preserved behavior: 158D full-6D Actor, 449D `V(s)`, Gain-v008, per-attempt
symlog, B8/M4, K/DR, low-DR coupled update, Actor LR `3e-7 -> 1e-6`, Critic LR
`1e-5`, one Adam, separate clipping, frozen GMT and exact-one commit.

Semantic owner and Pinch Point: `frontres_outer_scenario_replay.py`, specifically
`FrontRESOuterScenarioReplay.stage`. Its public input is current utility/value
rows plus Scenario identity; its public output is eight current targets and
latest priorities. The runner orchestrates, PPO consumes targets, checkpointing
persists state, and telemetry projects evidence. No wrapper or second route is
introduced.

State boundary: Replay persists keys, latest priority, lifetime visits,
staleness, membership/capacity, RNG and last receipt. Utility windows, policy
anchors/KL/resets and historical targets are deleted. Stage remains a preview;
commit remains exact-one-receipt gated. Replay-v4/checkpoint-v18 reject.

Forbidden: M16, repeated optimizer updates, old Actor rows/actions/log-probs,
historical utility targets, LR/Gain/K/DR/network changes, MOSAIC changes,
simulator test rigs, long training, Git operations or server work.

## Execution batches

1. Activate Contract/Inspector/Register and record TEST-25A..D.
2. Add failing production-boundary tests that expose history contamination and policy-window persistence.
3. Delete the window mechanism at the Replay owner and update its direct consumers, telemetry and identities.
4. Prove current-M4 target/Actor separation, lifetime-visit curriculum, exact-one rollback and strict v19 persistence.
5. Run the affected suite, final code review and formal offline route audit; stop before live execution.

## Confirmed Module Test Cards

### TEST-25A Current-only Critic target

Boundary: `FrontRESOuterScenarioReplay.stage/commit`.
Ordinary oracle: current M4 `[1,2,3,4]` targets `2.5`. On a later visit with
`[9,9,9,9]`, target is exactly `9`, independent of the committed first visit.
Changing serialized prior priority/visit metadata cannot change the target.

### TEST-25B Current uncertainty priority

Boundary: Replay candidate. For current M4, compute sample variance, `SE`,
`h95=1.96*SE`, and `E_V=max(abs(V-target)-h95,0)` by hand. The target remains
the arithmetic mean even when M4 variance is high. A variance-only target or
historical-window estimate falsifies the card.

### TEST-25C Replay lifecycle and persistence

Boundary: plan/stage/commit/state roundtrip. A replay selection reproduces its
stable key and is rerun before stage. Four committed visits, not a compatibility
window, satisfy maturity. Replay-v5 state contains no utility, policy anchor,
KL or reset fields; v4 rejects before mutation. Duplicate/failed commit is zero-delta.

### TEST-25D PPO and formal transaction

Boundary: PPO target consumer through the production formal transaction.
Eight current M4 means expand to their four rows; 32 Actor advantages remain
current `U(G)-V_old`; exactly one optimizer step and one Replay commit occur.
Missing/reordered target or post-step failure rolls back optimizer, Replay and normalizer.

## Review verdict and stop

Engineering Plan Review: READY. The plan uses the existing Replay aggregate and
Unit of Work, removes one obsolete Value Object and its consumers, adds no new
dependency direction or abstraction, and covers owner, consumer, persistence,
failure and official-offline edges. Residual live fact: one real IsaacLab K8
transaction must later prove physical collection and fresh rerun; this plan
does not execute it or claim policy quality.

