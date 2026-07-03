# Stage 3 Training Eval Diagnostics Plan

## Problem

Stage 3 Segment Replay HRL can now run, but the current log mainly proves the
training path is alive.  It does not clearly answer whether the network is
fitting the segment data, whether repair improves executability, or whether the
repair remains motion-natural.

The diagnostic contract should answer three questions:

1. Does repaired rollout improve over noisy rollout?
2. Does the repair damage motion quality?
3. Does the local segment policy transfer to longer rollout behavior?

## Scope

Add training diagnostics for Stage 3 only.

Do not change the training objective in this plan.  These metrics are first
logs/tests, not losses.

## Repository Ownership

Use the existing ownership pattern.

- `source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py`
  - owns scalar names, compact formatting, and pure tensor/math summaries.
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
  - owns K-step rollout capture and per-sample live payload.
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`
  - owns sampler evidence and replay priority diagnostics.
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
  - owns normal train-loop summary printing.
- tests stay in `source/rsl_rl/rsl_rl/tests/`.

Runner helpers should pass detached facts into diagnostics.  They should not
own metric formulas.

## Diagnostic Group 1: Repair Effect

Purpose: check whether FrontRES is actually improving the noisy segment.

Required fields:

- `score.noisy`
- `score.repaired`
- `score.gain`
- `score.gain_pos_frac`
- `fall_rate`
- `valid_frac`
- `replay_candidates`
- `replay_pool_size`

Interpretation:

- `gain > 0`: average repair helps.
- `gain_pos_frac > 0.5`: improvement is not caused by a few lucky samples.
- falling `fall_rate`: repair is not destabilizing the robot.
- growing `replay_pool_size`: Segment Replay is accumulating useful samples.

Implementation owner:

- source data: live probe summary and sampler evidence.
- formatter/scalars: `frontres_segment_diagnostics.py`.
- printed by: `frontres_segment_live_training.py`.

Minimal test:

- Build a fake summary with noisy/repaired/gain/fall/replay fields.
- Assert scalar keys exist.
- Assert formatted block contains positive gain and replay pool count.

## Diagnostic Group 2: Motion Quality

Purpose: check whether the repair is physically natural and not just gaming the
short-horizon score.

Required fields:

- `mpjpe_clean_repaired`
- `mpjpe_noisy_repaired`
- `delta_vel_error`
- `delta_acc_error`
- `delta_se_norm`
- `delta_z_up_frac`

Interpretation:

- `mpjpe_clean_repaired` should not grow while gain improves.
- `delta_vel_error` and `delta_acc_error` detect discontinuous repairs.
- `delta_se_norm` detects excessive correction magnitude.
- `delta_z_up_frac` detects dangerous upward root correction behavior.

Implementation owner:

- pure math: `frontres_segment_diagnostics.py`.
- source tensors:
  - repaired / noisy / clean reference facts from live probe when available;
  - action delta from policy output or segment storage.
- printed by: `frontres_segment_live_training.py` after Group 1.

Minimal test:

- Use a 3-frame toy joint/root trajectory.
- Assert MPJPE, velocity error, acceleration error, delta norm, and upward dz
  fraction are finite and numerically expected.

## Diagnostic Group 3: Periodic Long-Rollout Eval

Purpose: check whether K-step segment learning transfers to longer behavior.

Required fields:

- `episode_length`
- `success_rate`
- `fall_rate`
- `mean_survival_steps`
- `continuous_rollout_gain`

Interpretation:

- local gain alone is not enough;
- long-rollout success should improve or at least not degrade;
- success/fall should be measured on a fixed small eval set.

Implementation owner:

- schedule and print cadence: `frontres_segment_live_training.py`;
- rollout collection: a small runner helper, not the algorithm;
- formatting/scalars: `frontres_segment_diagnostics.py`.

Minimal test:

- Fake eval summary with episode length, success, fall, and continuous gain.
- Assert formatter prints a separate eval block.
- Live IsaacLab eval remains a later sentinel test, not Step 1.

## Step Plan

### Step 1: Note contract

Write this file.  No code changes.

Stop condition:

- the three diagnostic groups, owners, and tests are explicit.

### Step 2: Repair Effect diagnostics

Add pure scalar/format helpers for Group 1 in
`frontres_segment_diagnostics.py`.

Test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py
```

Stop condition:

- fake summary prints gain, gain_pos_frac, fall_rate, valid_frac, replay pool.

### Step 3: Motion Quality diagnostics

Add pure tensor helpers for Group 2 in `frontres_segment_diagnostics.py`.

Test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_contract.py
```

Stop condition:

- toy trajectories produce expected MPJPE, velocity error, acceleration error,
  delta norm, and upward dz fraction.

### Step 4: Wire compact train log

Wire Group 1 and any available Group 2 scalars into the existing Stage 3 train
log.

Files:

- `frontres_segment_live_probe.py`
- `frontres_segment_live_training.py`

Test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py
```

Stop condition:

- printed log has separate blocks:
  - `FrontRES Segment Train Effect`
  - `FrontRES Segment Motion Quality`
  - `FrontRES Segment Replay`
  - `FrontRES Segment PPO`

### Step 5: Periodic eval contract

Add config/readiness contract for periodic long-rollout eval, disabled by
default.

Test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_periodic_eval_contract.py
```

Stop condition:

- fake eval summary formats `episode_length`, `success_rate`,
  `mean_survival_steps`, and `continuous_rollout_gain`.

### Step 6: Live sentinel

Run a tiny Stage 3 live test and inspect only the new diagnostic blocks.

Stop condition:

- no traceback;
- repair effect block shows non-empty gain fields;
- motion quality block shows finite values when tensors are available;
- periodic eval remains disabled unless explicitly enabled.

