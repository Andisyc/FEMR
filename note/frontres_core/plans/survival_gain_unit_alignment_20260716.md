# C: Survival Gain Unit Alignment

Status: C implementation complete offline; formal consumer alignment remains open.
Origin: main-conversation decision on 2026-07-16, based on the side proposal.
Evidence status: E55 offline owner/connectivity evidence; E58 confirms the
formal route reaches v002 rewards/returns, but same-transaction diagnostic and
training Gain equality is not yet recorded. Quality comparisons are deferred.
Related contract: `FRS-GAIN-v002-style-physics-repair`.
Related design points: `FRS-DP-06` Paired Rollouts, `FRS-DP-07` Repair Gain.

## Problem

The pre-C live route exposed two survival units:

```text
pre-C per-step Gain path: survival_steps / (rollout_step + 1)
pre-C final Gain path:    repaired_survival_steps - noisy_survival_steps
```

The final raw difference can be tens of steps, while success, ZMP, contact,
and repair cost are small normalized quantities.

## Evidence

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
  now passes alive increments for the per-step path and raw cumulative steps
  for the final path, with the same `horizon_k` forwarded to the owner.
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
  now performs the only survival-quality conversion and paired subtraction.
- `note/testing/evidence_ledger_frontres_gain_2026-07-13.md:E51`
  records finite live components and the remaining unit-alignment gap.
- MOSAIC uses normalized `frontres_survival_rate` for quality diagnostics and
  normalizes episode length by a reference horizon for frontier scoring.
- Level Replay provides level-score aggregation and step-weighted partial
  updates, but does not define FEMR survival Gain semantics:
  `https://github.com/facebookresearch/level-replay`.

## Accepted Design

Accepted design:

```text
raw survival_steps:
  remain an evaluation/diagnostic quantity only.

survival_quality:
  survival_steps / effective_horizon_K.

physics_survival_gain:
  repaired_survival_quality - noisy_survival_quality.
```

Long sequence survival remains separately reported and must not be confused
with the short K-step Segment Gain.

## C Execution Scope

- `frontres_gain.py` owns the K-normalized quality and paired difference.
- Final Gain consumes cumulative raw survival steps divided by each row's K.
- Per-step PPO Gain consumes the current alive increment divided by that same K.
- Periodic and sequence reports keep raw survival steps separate from quality.
- The 120-step sequence horizon is a separate long-sequence report.

## Non-Scope

- No change to termination or `done_any`.
- No change to PPO optimizer, action, perturbation, or replay sampling policy.
- No long training or checkpoint selection.
- No active contract edit before user confirmation in the main conversation.

## C Stop Condition

Offline stop condition passed: the owner rejects missing K as unconfirmed, the
K=1/4/8 values match hand calculation, and per-step alive-increment Gain sums
to the final K-normalized survival Gain. The remaining formal step is to show
that one paired transaction feeds the same component values into diagnostics,
per-step reward, returns, and advantages. Model-pair quality and long-sequence
runtime evidence are separate post-training observations.
