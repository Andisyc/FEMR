# FrontRES Segment Replay Engineering Contract
Date: 2026-06-27

This note is Step 3 after `note/frontres_segment_replay/references/external_code_reuse_map.md`.
It defines the FEMR module contract before code implementation.

The core method is:

```text
motion segment
 -> dynamic reset
 -> noisy baseline
 -> HRL Delta SE(3) repair
 -> K-step rollout
 -> executable gain over Noisy
 -> PPO update
 -> prioritized segment replay update
```

HSL is initialization.  It is not the final training target.  The old
acceptance head remains an ablation path and must not run silently in Segment
Replay HRL.

## 1. Live Stage Contract

Add a new explicit stage:

```text
frontres_stage = stage3_segment_hrl
frontres_training_objective = segment_replay_hrl
```

Required stage defaults:

- `frontres_segment_replay_enabled=True`;
- `frontres_acceptance_preference_weight=0.0`;
- `frontres_split_acceptance_head=False` unless an ablation explicitly enables
  the old path;
- `frontres_authority_actor_critic_enabled=False` for the first implementation;
- `frontres_hsl_init_enabled=True`;
- `frontres_segment_k` set by config;
- `frontres_segment_sampler_global_frac`;
- `frontres_segment_sampler_replay_frac`;
- `frontres_segment_sampler_review_frac`;
- `frontres_segment_reset_mode=auto`.

Required live-path sentinel:

```text
FrontRES Segment HRL active: stage=stage3_segment_hrl objective=segment_replay_hrl k=...
```

Stop if:

- `stage2_acceptance` logging appears during `stage3_segment_hrl`;
- acceptance target/mask storage is written as the main HRL signal;
- the policy output is interpreted as acceptance instead of 6D repair.

## 2. Module Layout

New modules should live under:

```text
source/rsl_rl/rsl_rl/frontres/
```

Required modules:

- `frontres_segment_dataset.py`;
- `frontres_segment_sampler.py`;
- `frontres_segment_reset.py`;
- `frontres_segment_reward.py`;
- `frontres_hrl_action.py`;
- `frontres_segment_diagnostics.py`.

Optional module:

- `frontres_segment_storage.py`.

Runner connector should be thin and may live under:

```text
source/rsl_rl/rsl_rl/runners/frontres_segment_replay.py
```

Do not place dataset, sampler, reset, reward, or priority logic inside
`on_policy_runner.py`.

## 3. Shared Data Objects

### `FrontRESSegmentState`

Owner:

- `frontres_segment_dataset.py`.

Meaning:

- one batched dynamic state that can reset the simulator.

Fields:

- `root_pos`: tensor `[B, 3]`;
- `root_quat`: tensor `[B, 4]`, wxyz convention;
- `root_lin_vel`: tensor `[B, 3]`;
- `root_ang_vel`: tensor `[B, 3]`;
- `dof_pos`: tensor `[B, D]`;
- `dof_vel`: tensor `[B, D]`;
- `key_body_pos`: optional tensor `[B, K, 3]`;
- `key_body_quat`: optional tensor `[B, K, 4]`;
- `device`: torch device.

Invariant:

- pose and velocity are both required.

Stop if:

- reset can be built from `root_pos`, `root_quat`, and `dof_pos` only.

### `FrontRESSegmentSpec`

Owner:

- `frontres_segment_dataset.py`.

Meaning:

- one replayable training item.

Fields:

- `segment_id`: int;
- `motion_id`: int or string;
- `start_frame`: int or `start_time`: float;
- `phase`: float in `[0, 1]`;
- `horizon_k`: int;
- `perturbation_family`: string;
- `perturbation_strength`: float or tensor;
- `reset_mode_hint`: `direct`, `preroll`, or `auto`;
- `valid_for_training`: bool.

Invariant:

- `segment_id` is stable across sampler, reward, diagnostics, and checkpoint.

### `FrontRESSegmentBatch`

Owner:

- `frontres_segment_dataset.py`.

Fields:

- `segment_ids`: tensor `[B]`;
- `specs`: list or structured metadata;
- `clean_state`: `FrontRESSegmentState`;
- `reference_window`: tensor or adapter object for GMT reference;
- `phase`: tensor `[B]`;
- `horizon_k`: int or tensor `[B]`;
- `perturbation_family`: list or encoded tensor;
- `perturbation_strength`: tensor.

Invariant:

- batch object contains enough data for reset without going back to global
  runner state.

## 4. `frontres_segment_dataset.py`

Primary class:

```text
FrontRESSegmentDataset
```

Constructor inputs:

- `motion_source`;
- `dt`;
- `default_horizon_k`;
- `device`;
- `cache_policy`;
- optional `motion_normalizer`;
- optional `reference_builder`.

Required methods:

- `num_segments() -> int`;
- `sample_global(batch_size, generator=None) -> FrontRESSegmentBatch`;
- `get_segments(segment_ids) -> FrontRESSegmentBatch`;
- `build_clean_cache() -> None`;
- `validate_batch(batch) -> FrontRESSegmentValidation`;
- `state_dict() -> dict`;
- `load_state_dict(state) -> None`.

Optional methods:

- `write_noisy_baseline(segment_ids, evidence) -> None`;
- `read_noisy_baseline(segment_ids) -> evidence | None`;
- `update_validity(segment_ids, flags) -> None`.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_dataset_contract.py
```

Test cases:

- tiny fake motion with two clips and nonzero velocities;
- global sampling returns stable `segment_id`;
- `get_segments()` returns same state for same id;
- dynamic state includes root and dof velocities;
- reference window length covers `horizon_k`;
- invalid segment is excluded from global sampling.

Pass condition:

- dataset can be tested without IsaacLab and without GMT.

## 5. `frontres_segment_reset.py`

Primary class:

```text
FrontRESSegmentResetAdapter
```

Data objects:

- `FrontRESSegmentResetRequest`;
- `FrontRESSegmentResetResult`.

Required methods:

- `build_request(batch, mode="auto") -> FrontRESSegmentResetRequest`;
- `apply(env, request) -> FrontRESSegmentResetResult`;
- `validate_after_reset(obs, infos, request) -> FrontRESSegmentResetResult`;
- `needs_preroll(batch) -> tensor[bool]`.

`FrontRESSegmentResetRequest` fields:

- `segment_ids`;
- `root_pos`;
- `root_quat`;
- `root_lin_vel`;
- `root_ang_vel`;
- `dof_pos`;
- `dof_vel`;
- `reference_window`;
- `mode`;
- `preroll_steps`;
- `valid_mask`.

`FrontRESSegmentResetResult` fields:

- `success_mask`;
- `direct_reset_mask`;
- `preroll_mask`;
- `invalid_static_reset_mask`;
- `fall_at_reset_mask`;
- `contact_mismatch_mask`;
- `velocity_mismatch`;
- `diagnostics`.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_reset_contract.py
```

Test cases:

- fake env receives velocity fields;
- dynamic segment rejects static-pose-only request;
- `auto` mode chooses pre-roll for flagged unstable phases;
- partial batch failure returns masks, not an exception;
- diagnostics expose direct/pre-roll/failure counts.

Pass condition:

- reset adapter can be tested with a fake env before IsaacLab integration.

## 6. `frontres_segment_sampler.py`

Primary class:

```text
FrontRESSegmentSampler
```

Data objects:

- `FrontRESSegmentSample`;
- `FrontRESSegmentRolloutEvidence`;
- `FrontRESSegmentSamplerStats`.

Constructor inputs:

- `num_segments`;
- `global_frac`;
- `replay_frac`;
- `review_frac`;
- `priority_mode`;
- `staleness_weight`;
- `min_replay_score`;
- `max_hopeless_replay_frac`;
- `seed`.

Required methods:

- `sample(batch_size) -> FrontRESSegmentSample`;
- `update(evidence: FrontRESSegmentRolloutEvidence) -> None`;
- `mark_invalid(segment_ids, reason) -> None`;
- `stats() -> FrontRESSegmentSamplerStats`;
- `state_dict() -> dict`;
- `load_state_dict(state) -> None`.

`FrontRESSegmentSample` fields:

- `segment_ids`;
- `source`: global, replay, or review;
- `priority`;
- `staleness`;
- `valid_mask`.

`FrontRESSegmentRolloutEvidence` fields:

- `segment_ids`;
- `reset_success`;
- `score_noisy`;
- `score_repaired`;
- `score_clean`;
- `gain_over_noisy`;
- `fall_repaired`;
- `contact_consistency`;
- `action_norm`;
- `valid_reward`;
- `horizon_k`;

Priority rule:

- high priority means useful learning signal;
- high priority does not mean simply hard.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py
```

Test cases:

- unseen segments are sampled before replay dominates;
- useful unsolved segments receive higher replay probability;
- solved segments move to review instead of disappearing;
- hopeless segments are capped;
- stale segments re-enter sampling;
- `state_dict()` restores sampling state.

Pass condition:

- sampler can be tested as a pure Python module.

## 7. `frontres_segment_reward.py`

Primary class:

```text
FrontRESSegmentReward
```

Data objects:

- `FrontRESSegmentScoreWindow`;
- `FrontRESSegmentRewardResult`.

Constructor inputs:

- executable scorer or score function;
- reward weights;
- fall penalty;
- contact weight;
- valid score bounds;
- `use_full_env_reward=False`.

Required methods:

- `score_window(rollout, role) -> FrontRESSegmentScoreWindow`;
- `compute(noisy, repaired, clean, reset_result=None) -> FrontRESSegmentRewardResult`;
- `priority_evidence(result) -> FrontRESSegmentRolloutEvidence`.

`FrontRESSegmentRewardResult` fields:

- `reward`;
- `score_noisy`;
- `score_repaired`;
- `score_clean`;
- `gain_over_noisy`;
- `clean_gap`;
- `fall_flag`;
- `contact_consistency`;
- `valid_mask`;
- `solved_mask`;
- `hopeless_mask`;
- `diagnostics`.

Reward rule:

- main reward is improvement over Noisy;
- Clean is diagnostic and normalization reference;
- full environment reward is not the main signal.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_reward_contract.py
```

Test cases:

- repaired better than Noisy gives positive reward;
- repaired worse than Noisy gives negative reward;
- Noisy already good becomes low learning value;
- both Noisy and repaired fail becomes hopeless;
- invalid reset masks reward out;
- full env reward is ignored unless explicitly enabled.

Pass condition:

- reward can be tested from synthetic score tensors.

## 8. `frontres_hrl_action.py`

Primary class:

```text
FrontRESHRLActionProjector
```

Data objects:

- `FrontRESRepairAction`;
- `FrontRESHRLActionStats`.

Constructor inputs:

- `FrontRESActionCone`;
- active task dims;
- action scale;
- upward `dz` rule;
- optional HSL initialization mode.

Required methods:

- `project(raw_action, mode_groups=None) -> FrontRESRepairAction`;
- `apply_to_reference(command, repair_action) -> command`;
- `mask_for_segment(batch) -> tensor`;
- `stats(repair_action) -> FrontRESHRLActionStats`.

`FrontRESRepairAction` fields:

- `delta_se`: tensor `[B, 6]`;
- `active_mask`: tensor `[B, 6]`;
- `projected_delta_se`: tensor `[B, 6]`;
- `action_norm`: tensor `[B]`;
- `per_dim_norm`: tensor `[6]`.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_hrl_action_contract.py
```

Test cases:

- active dimension mask is respected;
- per-mode mask is respected;
- upward `dz` is constrained;
- output remains 6D Delta SE(3);
- no acceptance probability/logit is produced;
- HSL actor weights can initialize the 6D repair actor without an acceptance
  actor.

Pass condition:

- action projector can be tested without env rollout.

## 9. `frontres_segment_diagnostics.py`

Primary functions:

- `summarize_segment_batch(sample, reward_result, reset_result, action_stats)`;
- `format_segment_replay_log(summary) -> str`;
- `segment_summary_to_scalars(summary) -> dict[str, float]`.

Required diagnostics:

- `segment/global_frac`;
- `segment/replay_frac`;
- `segment/review_frac`;
- `segment/replay_pool_size`;
- `segment/priority_mean`;
- `segment/priority_p90`;
- `segment/solved_frac`;
- `segment/active_frac`;
- `segment/hopeless_frac`;
- `segment/reset_success_frac`;
- `segment/preroll_frac`;
- `segment/k`;
- `segment/score_noisy`;
- `segment/score_repaired`;
- `segment/score_clean`;
- `segment/gain_over_noisy`;
- `segment/fall_frac`;
- `segment/contact_consistency`;
- `segment/action_norm`;
- `segment/action_norm_dx`;
- `segment/action_norm_dy`;
- `segment/action_norm_dz`;
- `segment/action_norm_droll`;
- `segment/action_norm_dpitch`;
- `segment/action_norm_dyaw`.

Forbidden primary diagnostics:

- `acceptance_gt`;
- `acceptance_mask`;
- `acceptance_margin`;
- `acceptance_prob`.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py
```

Test cases:

- fake batch produces all required scalar keys;
- old acceptance keys are absent;
- partial invalid batch still logs reset and valid-mask statistics;
- log string includes stage, objective, K, sampler mix, and gain.

Pass condition:

- one log line can prove Segment Replay HRL is active.

## 10. Optional `frontres_segment_storage.py`

Create this only if current `RolloutStorage` cannot hold the tuple cleanly.

Primary class:

```text
FrontRESSegmentRolloutStorage
```

Required fields:

- observations;
- privileged observations if used;
- `segment_ids`;
- `segment_source`;
- 6D repair actions;
- log-probs;
- values;
- rewards;
- returns;
- advantages;
- valid masks;
- reset masks;
- priority evidence.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py
```

Stop if:

- the old acceptance storage tuple is expanded before this storage decision is
  made.

Step 12 implementation result:
- chose an independent Stage 3 storage instead of extending legacy
  `RolloutStorage`;
- reason: legacy `RolloutStorage` already carries supervised, acceptance, rho,
  state-alpha, and authority fields, so adding Segment HRL there would risk
  reintroducing old acceptance paths;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py` defines
  `FrontRESSegmentRolloutStorage`;
- storage fields are observations, optional privileged observations,
  `segment_ids`, `segment_source`, 6D repair actions, log-probs, values,
  rewards, returns, advantages, valid masks, reset masks, action masks, and
  detached priority evidence;
- segment returns default to K-step reward and advantages default to
  `return - value`;
- connector writes are accepted only when policy log-prob, value, and
  observations are present; missing PPO fields fail fast;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py` verifies
  6D action shape, invalid-mask removal, PPO batch conversion, state round-trip,
  overflow rejection, and connector payload validation;
- this is still a fake storage contract and does not enable live Stage 3
  training.

## 11. Thin Runner Connector

Candidate file:

```text
source/rsl_rl/rsl_rl/runners/frontres_segment_replay.py
```

Primary class:

```text
FrontRESSegmentReplayConnector
```

Responsibilities:

- request segment ids from sampler;
- fetch segment batch from dataset;
- call reset adapter;
- get HRL repair action from policy;
- run K-step rollout;
- call segment reward;
- write PPO transition or segment storage;
- update sampler priority;
- emit diagnostics.

Non-responsibilities:

- no priority formula;
- no reward formula;
- no reset-state construction;
- no action-cone math;
- no acceptance label construction.

Required live-path test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_replay_toy_chain.py
```

Test case:

- fake dataset;
- fake sampler;
- fake reset adapter;
- fake policy action;
- fake K-step rollout;
- fake reward;
- verify call order and written transition fields.

Pass condition:

- toy chain proves the new path is connected without IsaacLab.

Step 13 implementation result:
- `source/rsl_rl/rsl_rl/runners/frontres_segment_replay.py` accepts policy
  outputs that are either a raw 6D tensor or a dict/object containing `action`,
  `observations`, `log_prob`, `value`, optional `mean`, and optional `sigma`;
- connector writes both `raw_action` and `policy_output` into the transition
  payload, so `FrontRESSegmentRolloutStorage` can build a PPO tuple without
  touching legacy `RolloutStorage`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_runner_lifecycle_contract.py`
  verifies the fake lifecycle:
  segment sampling -> reset -> policy action/log-prob/value -> K-step rollout
  -> Noisy-relative reward -> sampler update -> segment storage write -> PPO
  batch -> optimizer step;
- the old tensor-only connector toy test remains valid;
- this remains a fake runner lifecycle and does not enable live Stage 3
  training.

Step 14 live runner sentinel:

- `frontres_segment_live_sentinel_only` is an explicit startup-proof flag, not a
  training flag;
- default Stage 3 keeps `frontres_segment_live_runner_enabled=False`;
- passing `--frontres_segment_live_sentinel_only` sets both
  `frontres_segment_live_runner_enabled=True` and
  `frontres_segment_live_sentinel_only=True`;
- `FrontRESSegmentRunnerBoundary.assert_live_runner_ready()` allows only this
  sentinel case through;
- `on_policy_runner.py` prints `[FrontRES Segment Live Sentinel] ...` with
  objective, K, reset mode, independent storage, 6D Delta SE(3) action, and
  `training_update=disabled`;
- `FrontRESUnified.update()` still raises if the sentinel reaches training,
  proving this step does not enable PPO/live learning.

Step 14 stop condition:

- the real Stage 3 startup boundary has a visible one-line proof;
- non-sentinel live runner still fails fast;
- update/training remains disabled until the actual live rollout/PPO connector
  is implemented.

Step 15 live rollout probe:

- `frontres_segment_live_probe_only` is an explicit live rollout probe flag, not
  a training flag;
- passing `--frontres_segment_live_probe_only` sets
  `frontres_segment_live_runner_enabled=True` and
  `frontres_segment_live_probe_only=True`;
- sentinel and probe flags are mutually exclusive;
- `train.py` forces `max_iterations=0`, constructs the real env and runner,
  loads the requested checkpoint if provided, then calls
  `runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)` and
  exits before `runner.learn()`;
- `run_frontres_segment_live_probe()` reuses the normal runner observation
  split, normalizers, FrontRES pair layout, `prepare_frontres_rollout_step()`,
  and `env.step()` for K steps;
- the probe prints `[FrontRES Segment Live Probe] ...` with obs shape, 6D action
  shape, env-action shape, valid fraction, K, reward mean, done fraction,
  `storage_write=False`, and `ppo_update=False`;
- it does not call `alg.process_env_step()`, does not write Segment Replay
  storage, and does not call `alg.update()`.

Step 15 stop condition:

- the real env can execute K live probe steps through the Stage 3 runner path;
- the log proves policy action and env action shapes;
- storage and PPO remain disabled until Step 16/17.

Step 16 live storage write:

- `frontres_segment_live_storage_write_only` is an explicit live storage probe
  flag, not a training flag;
- passing `--frontres_segment_live_storage_write_only` sets
  `frontres_segment_live_runner_enabled=True` and
  `frontres_segment_live_storage_write_only=True`;
- sentinel, probe, and storage-write flags are mutually exclusive;
- `train.py` forces `max_iterations=0`, constructs the real env and runner,
  loads the requested checkpoint if provided, then reuses
  `runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)` and exits
  before `runner.learn()`;
- the first live policy step supplies the PPO tuple:
  observation, privileged observation, 6D Delta SE(3) action, log-prob, value,
  optional mean/sigma, and segment ids;
- the following K-step live rollout accumulates the segment reward and done
  mask;
- the runner writes one `FrontRESSegmentTransition` into independent
  `FrontRESSegmentRolloutStorage`;
- the probe prints storage evidence:
  `storage_write=True`, `storage_size`, `storage_valid_frac`, and
  `storage_reward_mean`;
- it still does not call `alg.process_env_step()` and does not call
  `alg.update()`.

Step 16 stop condition:

- live env produces a valid 6D PPO tuple;
- independent Segment Replay storage accepts the live transition;
- PPO/update remains disabled until Step 17.

Step 17 live single-batch PPO update:

- `frontres_segment_live_single_update_only` is an explicit update sentinel,
  not the full Stage 3 training loop;
- passing `--frontres_segment_live_single_update_only` sets
  `frontres_segment_live_runner_enabled=True` and
  `frontres_segment_live_single_update_only=True`;
- sentinel, probe, storage-write, and single-update flags are mutually
  exclusive;
- `train.py` forces `max_iterations=0`, constructs the real env and runner,
  loads the requested checkpoint if provided, then reuses
  `runner.run_frontres_segment_live_probe(init_at_random_ep_len=True)` and exits
  before `runner.learn()`;
- the runner first follows the Step 16 path: live K-step rollout, first-step
  6D Delta SE(3) PPO tuple, independent `FrontRESSegmentRolloutStorage`;
- after storage write, the runner builds one
  `FrontRESSegmentPPOBatch`, re-evaluates the stored actions under the current
  FrontRES policy, computes `compute_frontres_segment_ppo_loss`, and performs
  exactly one optimizer step when `valid_count > 0`;
- critic evaluation uses the stored privileged observations when available,
  not the actor observation by default;
- `FrontRESUnified.update()` remains guarded for this mode; entering the full
  update loop is an error;
- the probe prints update evidence:
  `single_update=True`, `ppo_update`, `ppo_valid_count`, `ppo_total_loss`,
  `ppo_actor_loss`, `ppo_value_loss`, `ppo_approx_kl`, and `ppo_clip_frac`.

Step 17 stop condition:

- live storage can become a PPO batch;
- the current policy can re-evaluate stored 6D actions with gradients enabled;
- one optimizer step can run from the live segment batch;
- the command still exits before the expensive training loop.

Step 18 live short PPO update loop:

- `frontres_segment_live_update_loop_only` is an explicit short-loop sentinel,
  not the full Stage 3 training loop;
- `frontres_segment_live_update_steps` controls the number of live segment PPO
  updates and defaults to 4;
- passing `--frontres_segment_live_update_loop_only` sets
  `frontres_segment_live_runner_enabled=True`,
  `frontres_segment_live_update_loop_only=True`, and
  `frontres_segment_live_update_steps=N`;
- sentinel, probe, storage-write, single-update, and update-loop flags are
  mutually exclusive;
- `train.py` forces `max_iterations=0`, constructs the real env and runner,
  loads the requested checkpoint if provided, runs
  `runner.run_frontres_segment_live_update_loop(init_at_random_ep_len=True)`,
  and exits before `runner.learn()`;
- each loop iteration reuses the Step 17 path:
  live K-step segment rollout, independent segment storage, PPO batch,
  policy re-evaluation, one masked PPO optimizer step when `valid_count > 0`;
- the first loop iteration may randomize episode length; later iterations
  continue from the live env state instead of resetting through the training
  lifecycle;
- the loop prints per-segment Step 17 evidence plus one summary line:
  `[FrontRES Segment Live Update Loop]`, `update_steps`, `update_count`,
  `ppo_valid_count`, mean rewards/losses/KL/clip fraction, and
  `runner_learn=False`;
- `FrontRESUnified.update()` remains guarded for this mode; entering the full
  update loop is an error.

Step 18 stop condition:

- consecutive live segment batches can update the same policy without entering
  the normal runner training loop;
- update statistics are scalar-detached and printed once at the loop boundary;
- the command remains short enough to use as a server-side live smoke test.

Step 19 dedicated live training loop:

- `frontres_segment_live_train_enabled` is the explicit Stage 3 live training
  flag;
- when `--frontres_stage stage3_segment_hrl` is used without a sentinel flag,
  the Stage 3 preset sets `frontres_segment_live_train_enabled=True`;
- `train.py` constructs the real env and runner, loads the requested checkpoint
  if provided, then calls
  `runner.learn_frontres_segment_live(num_learning_iterations=max_iterations)`;
- this path still does not call the legacy `runner.learn()` and still does not
  call `FrontRESUnified.update()`;
- each training iteration reuses the Step 18 loop:
  `frontres_segment_live_update_steps` live segment PPO updates, scalar
  diagnostics, and detached storage;
- the runner prints `[FrontRES Segment Live Train]` with iteration, update
  count, valid sample count, reward mean, loss mean, and `runner_learn=True`;
- checkpoint saving uses the runner checkpoint helpers at the Segment Replay
  iteration boundary.

Step 19 stop condition:

- Stage 3 has a normal training entrypoint that can run for
  `max_iterations > 0`;
- the live path is still isolated from the legacy FrontRES update path;
- a server-side short command can prove the first real training iteration by
  observing `[FrontRES Segment Live Train] ... runner_learn=True`.

Step 19 pseudo-parameter contract:

- before server-side live smoke tests, the Stage 3 train loop must pass a fake
  parameter test without IsaacLab;
- the train loop is owned by
  `runners/frontres_segment_live_training.py`, not embedded directly in
  `on_policy_runner.py`;
- the fake runner supplies:
  `frontres_segment_live_train_enabled=True`, fake update summaries, fake
  checkpoint functions, and fake log paths;
- the test verifies:
  first iteration uses `init_at_random_ep_len=True`, later iterations use
  `False`, `runner_learn=True` reaches the update loop, checkpoints are named
  `model_{iteration}.pt`, and checkpoint probes receive the scalar summary;
- incomplete update summaries must fail before a real server run.

Required pseudo test:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py
```

Step 20 live training diagnostics and fail-fast contract:

- live Stage 3 training must not silently continue when the update loop returns
  an unusable summary;
- each live training iteration requires scalar diagnostics for:
  update steps, update count, valid PPO sample count, reward mean, storage valid
  fraction, total loss, actor loss, value loss, approximate KL, and clip
  fraction;
- `frontres_segment_live_fail_on_invalid_update=True` by default, so
  `update_count=0` fails before checkpointing;
- `frontres_segment_live_min_valid_count=1` by default, so a live iteration with
  no valid PPO samples fails before the next iteration;
- `frontres_segment_live_fail_on_nonfinite=True` by default, so NaN/Inf reward,
  loss, KL, or clip diagnostics fail immediately;
- the guards can be disabled explicitly for diagnosis, but the default server
  run should fail early rather than waste a long training job;
- the fake runner contract covers missing summary keys, non-finite diagnostics,
  zero update count, too few valid PPO samples, and disabled guards.

Step 20 stop condition:

- local pseudo-parameter tests catch bad live update summaries without IsaacLab;
- Stage 3 live train prints enough scalar evidence to judge whether the update
  path is actually doing PPO work;
- a server-side live run should now be used only after these local contracts
  pass.

Step 21 short training resumability contract:

- run a short server-side Stage 3 live train after Step 20 passes;
- confirm a checkpoint is saved into the FEMR run directory, not an old MOSAIC
  path or missing fallback path;
- resume from that checkpoint and confirm the resumed path still calls
  `runner.learn_frontres_segment_live(...)`;
- the resumed run must not fall back to legacy `runner.learn(...)`;
- checkpoint diagnostics must print the saved checkpoint path, loaded checkpoint
  path, resumed iteration, and `runner_learn=True`.

Step 21 stop condition:

- Stage 3 can save a live training checkpoint;
- Stage 3 can resume from that checkpoint;
- both cold-start and resume paths stay on `learn_frontres_segment_live`.

Step 21 implementation result:

- `frontres_segment_live_training.py` now prints a live checkpoint sentinel
  when Stage 3 saves a checkpoint: saved path, whether it is inside `log_dir`,
  iteration, and `runner_learn=True`;
- `frontres_checkpointing.py` records `_frontres_last_loaded_checkpoint_path`
  during `runner.load(...)`, so a resumed Stage 3 live run can print the loaded
  checkpoint path before training continues;
- `frontres_segment_live_training.py` prints a resume sentinel when a loaded
  checkpoint path is present: loaded path, resumed iteration,
  `runner_learn=True`, and `legacy_runner_learn=False`;
- `frontres_segment_live_resume_pseudo_contract.py` constructs a fake
  cold-start short train, resumes from the saved checkpoint path, and verifies
  the resumed runner advances from iteration 1 to 2 without calling legacy
  `runner.learn(...)`;
- the contract is included in both `frontres_segment_stage3_pseudo_suite.py`
  and `frontres_segment_all_contract_suite.py`.

Step 22 sampler strategy integration contract:

- only start after the live loop can run and resume stably;
- replace the temporary live path behavior of mostly current-env continuous
  probing with the Segment Replay sampler contract;
- connect global sampling, replay sampling, review sampling, and priority
  updates to the live Stage 3 loop;
- keep the sampler owner in `frontres_segment_sampler.py` and the live runner
  connector thin;
- print sampler boundary facts: sampled source counts, replay pool size,
  priority mean, solved fraction, hopeless fraction, and stale review count.

Step 22 stop condition:

- live Stage 3 batches are selected by the sampler, not by implicit continuous
  env progression only;
- rollout evidence updates sampler priority after each live update loop;
- sampler state can be saved and resumed with the Stage 3 checkpoint;
- the live loop remains stable after sampler integration.

Step 22 implementation result:

- `frontres_segment_live_sampler.py` owns the live sampler connector:
  initializes `FrontRESSegmentSampler`, samples `segment_ids/source` before
  each live probe, converts the detached live summary into sampler evidence,
  updates priority, and prints `[FrontRES Segment Sampler]`;
- `frontres_segment_live_update_loop.py` now calls the sampler connector for
  each update step and aggregates sampler diagnostics into the update-loop
  summary: source counts, replay pool size, priority mean, solved fraction,
  hopeless fraction, and stale review count;
- `frontres_segment_live_probe.py` writes sampled `segment_ids` and
  `segment_source` into independent Segment Replay storage instead of always
  using `arange(batch_size)`;
- `frontres_checkpointing.py` saves and restores
  `frontres_segment_sampler_state_dict` in the real runner checkpoint path;
- `frontres_segment_live_sampler_contract.py` verifies four local boundaries:
  live summary to sampler evidence, live update loop to priority update,
  sampled ids/source to storage, and checkpoint save/load of sampler state;
- `frontres_segment_stage3_pseudo_suite.py` and
  `frontres_segment_all_contract_suite.py` include the Step 22 contract.

Step 22 limitation:

- this step connects sampler strategy and priority persistence to the live
  Stage 3 loop, but it does not yet perform true motion-segment dynamic reset
  from an offline segment dataset.  The current live probe still executes on
  the current environment state; the sampled segment id now owns storage,
  priority, and checkpoint identity.  True dataset/reset-driven segment
  execution remains the next live-boundary integration.

Step 14 per-sample rollout evidence:

- scope: convert live rollout evidence from batch-level scalar summaries into
  per-sample evidence before sampler priority update;
- non-scope: no PPO loss change, no reference-window command injection, no
  training-command change;
- core parameter path:
  `segment_id -> reset_success/done_any/reward -> per_sample_evidence ->
  sampler priority`;
- `frontres_segment_live_probe.py` now exports detached list payloads for
  `reward_per_sample`, `done_any_per_sample`,
  `storage_reward_per_sample`, `storage_valid_mask_per_sample`, and
  `storage_segment_ids`;
- `frontres_segment_live_sampler.py` now builds
  `FrontRESSegmentRolloutEvidence` from those per-sample payloads before
  falling back to scalar means;
- failed reset and invalid rollout rows are masked at `valid_reward`, so they
  do not create useful replay priority;
- `[probe step14] evidence_path` prints compact boundary facts: sample count,
  segment id range, reward min/max, reset-valid count, rollout-valid count,
  valid-reward count, fall count, and gain mean;
- `frontres_segment_live_sampler_contract.py` includes a semantic two-sample
  test where one segment has positive valid reward and the other has negative
  done/reset-failed evidence, proving the batch mean is not copied to both
  segments.

Step 14 stop condition:

- two segment rows can carry different reward, fall, reset, and valid masks;
- sampler evidence preserves those row-level facts;
- pseudo and all-contract suites explicitly check `[probe step14]`.

Step 15 reference-window reset hook:

- scope: carry the sampled `reference_window` from the segment batch into the
  reset request and through a command-owned optional reference hook;
- non-scope: no PPO loss change, no sampler priority formula change, no real
  motion-loader time-step rewrite, and no training-command change;
- core parameter path:
  `segment_id -> batch.reference_window -> reset_request.reference_window ->
  env command reference hook -> reset diagnostics`;
- `frontres_segment_reset.py` now calls a command hook when available:
  `set_frontres_reference_window`, `apply_frontres_reference_window`, or
  `set_segment_reference_window`;
- if the command has no hook, dynamic state reset still works and
  `reference_window_applied_frac=0.0` exposes the missing boundary instead of
  silently pretending the reference was aligned;
- if the command hook exists, it receives the full batched tensor
  `[B, K+1, ...]` and the reset `env_ids`;
- `frontres_segment_live_probe.py` prints
  `segment_reference_window_applied_frac` in the live reset summary;
- `frontres_segment_live_reset_hook_contract.py` verifies a semantic fake
  command where two segment rows carry different reference-window values and
  both are written to the command hook.

Step 15 stop condition:

- `request.reference_window` and command-stored reference window match exactly
  in the fake contract;
- reset diagnostics expose `reference_window_applied_frac`;
- pseudo and all-contract suites explicitly check `[probe step15]`.

Step 16 MotionCommand reference consumer:

- scope: make the real `MultiMotionCommand` consume the Stage 3 segment
  `reference_window` after the reset hook passes it into the command;
- non-scope: no PPO loss change, no sampler priority change, no runner training
  loop change, and no server launch command change;
- core parameter path:
  `dataset.reference_window -> reset_request.reference_window ->
  MultiMotionCommand.set_frontres_reference_window -> command joint_pos/joint_vel
  gather override -> cursor advance -> clear/expire`;
- `frontres_segment_dataset.py` now builds the default `reference_window` from
  joint command payload `[joint_pos, joint_vel]`, not root position;
- `MultiMotionCommand` owns the override buffer, active mask, and per-env cursor;
- `command` remains the consumer boundary: `_gather_future_by_motion("joint_pos")`
  and `_gather_future_by_motion("joint_vel")` apply active overrides before GMT
  sees the flattened command tensor;
- `_update_command()` advances the cursor by one frame; `_resample_command()`
  clears active overrides for reset envs; exhausted windows expire automatically;
- `frontres_segment_motion_command_reference_contract.py` traces the same tensor
  through dataset payload construction, command first read, cursor advance,
  partial clear, and expiration.

Step 16 stop condition:

- contract prints `[probe step16] dataset.reference_window`,
  `[probe step16] command.first_read`, `[probe step16] command.after_advance`,
  `[probe step16] command.after_partial_clear`, and
  `[probe step16] reference_window_lifecycle`;
- contract ends with `frontres_segment_motion_command_reference_contract: ok`;
- pseudo and all-contract suites include the Step 16 command-consumer contract.

Step 17 local runner closed loop:

- scope: verify the local runner interface path after Step 16, using one segment,
  one env, and a two-step rollout;
- non-scope: no IsaacLab server launch, no PPO update, no training-performance
  claim, and no sampler formula change;
- core parameter path:
  `sampler.sample -> dataset.get_segments -> batch.reference_window ->
  reset request -> command reference hook -> K-step live probe -> segment storage
  -> rollout evidence -> sampler.update`;
- `frontres_segment_live_closed_loop_contract.py` uses the real
  `run_frontres_segment_sampler_step()` and the real
  `run_frontres_segment_live_probe()` with a semantic fake env;
- the fake env returns `reference_window_applied=True`, two rollout rewards, and
  no done flag, so storage writes one valid segment reward and evidence updates
  sampler priority.

Step 17 stop condition:

- contract prints `[probe step17] sampled.segment_ids`,
  `[probe step17] batch.reference_window`,
  `[probe step17] command.reference_window`,
  `[probe step17] sampler.seen`, `[probe step17] sampler.invalid`,
  `[probe step17] sampler.priority`, and
  `[probe step17] closed_loop_summary`;
- the closed-loop summary proves `reference_applied_frac=1.0`,
  `storage_segment_ids=[0]`, `storage_reward=[2.0]`,
  `storage_valid=[True]`, and positive sampler priority;
- pseudo and all-contract suites include the Step 17 local closed-loop contract.

## 12. Algorithm Contract

First implementation should reuse PPO only after segment reward and action tuple
are stable.

Required algorithm behavior:

- actor action is 6D Delta SE(3);
- value predicts segment return;
- return is K-step repair reward;
- invalid reset/reward samples are masked out;
- HSL initialization may initialize actor weights;
- acceptance loss is off;
- old acceptance tensors are not required.

Required gradient boundary:

- reward evidence is detached;
- sampler priority update is detached;
- diagnostics receive scalars or detached CPU tensors;
- HSL initialization does not keep Stage 1 graph alive.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py
```

Test cases:

- fake rollout batch updates actor on valid segment samples;
- invalid samples produce zero actor contribution;
- old acceptance loss is not called;
- PPO log-prob/value/advantage fields match 6D action shape.

Step 10 implementation result:
- `source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py` defines the pure
  Segment HRL PPO tuple and loss;
- tuple fields are `observations`, 6D `actions`, `old_log_probs`,
  `old_values`, `returns`, `advantages`, and sample-level `valid_mask`;
- `segment_ids` and `[B, 6] action_mask` are optional metadata, not the main
  loss signal;
- invalid samples are removed from actor/value/entropy loss, not merely
  down-weighted after the loss is built;
- the module has no dependency on old acceptance labels, masks, logits, or
  probability heads;
- this is still a fake-batch algorithm contract and does not enable live
  Stage 3 training.

## 13. Checkpoint Contract

Checkpoint behavior for `stage3_segment_hrl`:

- load Stage 1 HSL residual actor into 6D repair actor when requested;
- do not require `acceptance_actor`;
- reset optimizer unless config explicitly resumes optimizer;
- load normalizers normally;
- save sampler state if replay priority is persistent;
- save dataset cache metadata if cache ids must be reproducible.

Required tests:

```text
source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py
```

Stop if:

- Stage 3 load fails because a Stage 2 acceptance actor is missing.

Step 11 implementation result:
- `source/rsl_rl/rsl_rl/runners/frontres_segment_checkpointing.py` defines the
  Stage 3 checkpoint contract;
- Stage 1 HSL residual actor can initialize the 6D Stage 3 repair actor;
- Stage 1 two-head checkpoints map `trunk.*` and `proposal_head.*` into the
  6D repair actor and ignore `acceptance_head.*`;
- Stage 3 does not require `acceptance_actor`;
- optimizer state is reset by default and loaded only when explicitly requested;
- observation normalizers are restored when present;
- sampler state and dataset cache metadata are included in the Stage 3 payload
  when the runner exposes them;
- this remains a fake checkpoint contract and does not enable live Stage 3
  training.

## 14. Implementation Ladder

Implement in this order:

1. pure data objects and dataset toy test;
2. pure sampler and priority toy test;
3. action projector toy test;
4. reward toy test;
5. reset adapter fake-env test;
6. diagnostics toy test;
7. thin runner toy chain;
8. stage entrypoint contract;
9. algorithm contract;
10. checkpoint contract;
11. short live-path sentinel only after all above pass.

Do not start live training before steps 1-10 pass.

## 15. Stage 3 Stop Conditions

Stop and report mismatch if:

- static reset is used for dynamic phases;
- velocity fields are absent from segment state;
- sampler cannot explain replay choice;
- reward is absolute full env reward instead of Noisy-relative gain;
- Clean is used as direct supervised HRL target;
- HRL action is reduced to a scalar strength;
- HRL action is acceptance over HSL proposal;
- old acceptance diagnostics are the main Stage 3 diagnostics;
- runner owns dataset/sampler/reward internals;
- segment id is not stable across dataset, sampler, reward, diagnostics, and
  checkpoint;
- tests require IsaacLab for pure modules;
- stage flag can accidentally run old `stage2_acceptance`.

## 16. Step 3 Result

Step 3 turns the method into an implementation contract:

- new stage: `stage3_segment_hrl`;
- new objective: `segment_replay_hrl`;
- new module boundary under `rsl_rl.frontres`;
- toy-test-first implementation order;
- explicit separation from old acceptance training;
- exact stop conditions before any expensive run.

Next step:

- Step 4: implement `frontres_segment_dataset.py` and
  `frontres_segment_sampler.py` as pure modules with toy tests.

## 17. Stage 1/3 Cache IO Optimization Contract

Date: 2026-06-30

This section records the current Stage 1/3 cache IO contract after the Segment
Replay implementation reached the offline-cache path.  It is an engineering
contract, not a paper-method description.

### Problem

Stage 1 can generate a large cache.  Keeping all Clean/Noisy rollout states in
memory until the end is wrong.  Stage 3 should also not eagerly expand the whole
cache into memory before training.

The correct immediate contract is:

- Stage 1 writes payload shards during generation;
- Stage 1 writes final manifest and metadata only after generation completes;
- Stage 3 loads manifest records first and reads shard rows lazily;
- Stage 3 bounds repeated shard reads with an LRU cache;
- live server tests must still prove the IsaacLab path reaches these boundaries.

### Scope

This contract covers:

- Stage 1 Clean/Noisy cache payload writing;
- Stage 1 status/progress observability;
- Stage 3 lazy cache loading;
- Stage 3 Segment Replay sampler use of cached segments;
- local contract tests and live sentinel evidence required before formal runs.

### Non-Scope

This contract does not claim:

- HDF5, Zarr, mmap, or true partial-row storage;
- server-side IsaacLab verification;
- training quality improvement;
- final Stage 3 performance tuning;
- removal of all legacy/eager compatibility paths.

### Core Parameter Path

```text
cache_chunk_size
  -> clean_buffer / noisy_buffer
  -> write_*_chunked_shard(...)
  -> shards/.../*.pt payload file
  -> manifest record {path, row, segment, descriptor}
  -> metadata cache_storage_backend=torch_chunked_shard

frontres_segment_cache_dir
  -> load_stage1_cache_dataset(lazy=True)
  -> build_stage1_cache_lazy_records(...)
  -> segment_id -> manifest_record
  -> read_noisy_variant_record(cache_dir, record, shard_cache)
  -> FrontRESSegmentBatch
  -> sampler / storage / PPO update

shard_cache_size
  -> FrontRESSegmentShardLRU(max_shards=...)
  -> resident shard bound
  -> load_count / hit_count probe
```

### File Responsibility Map

`source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py`

- owns Stage 1 orchestration;
- owns `cache_chunk_size`;
- owns Clean/Noisy buffer flushing policy;
- owns status/progress event emission;
- must not own Stage 3 lazy dataset semantics.

`source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py`

- owns on-disk cache format;
- owns chunked payload shard writers;
- owns manifest record writers/readers;
- owns `FrontRESSegmentShardLRU`;
- must preserve backward compatibility with legacy per-file shard manifests
  until old caches are intentionally dropped.

`source/rsl_rl/rsl_rl/frontres/frontres_segment_dataset.py`

- owns eager `FrontRESSegmentDataset`;
- owns lazy `FrontRESStage1LazyCacheDataset`;
- owns `segment_id -> manifest record -> shard row -> batch` conversion;
- owns runtime invalidity filtering in `sample_global`;
- must not silently eager-load all Noisy payloads in the Stage 3 default path.

`source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`

- owns Stage 3 runner-side dataset loading and sampler connection;
- must call `load_stage1_cache_dataset(..., lazy=True)` by default;
- must print cache-load facts including lazy mode and shard-cache probe.

`scripts/rsl_rl/train.py`

- owns CLI/config routing only;
- should expose `cache_chunk_size` and `shard_cache_size` as explicit knobs;
- must not define cache format or lazy-read behavior.

`run/run_frontres_stage1_segment_cache.sh`

- owns user-facing Stage 1 command defaults;
- should pass `CACHE_CHUNK_SIZE` after the CLI flag exists.

`run/run_frontres_stage3_segment_hrl.sh`

- owns user-facing Stage 3 command defaults;
- should pass `SHARD_CACHE_SIZE` after the CLI flag exists.

### Invariants

- Stage 1 payload shards may appear before the build is complete.
- Stage 1 manifest and metadata are complete-build artifacts.
- A partial Stage 1 cache must not be treated as a valid Stage 3 training
  dataset unless a future explicit partial-cache mode is designed.
- `progress.jsonl` is runtime observability, not the canonical manifest.
- `metadata.json` is the completion boundary for normal Stage 3 cache loading.
- Stage 3 default cache loading must be lazy.
- Lazy dataset initialization must not load Noisy payload shards.
- `update_validity(..., False)` must remove a segment from future global
  sampling in both eager and lazy datasets.
- Boundary diagnostic samples may be present on disk, but are excluded from
  trainable Stage 3 dataset loading unless explicitly requested.
- Contract tests prove local semantics; only server logs prove IsaacLab live
  lifecycle.

### Current Evidence

Local contract evidence already exists for:

- chunked Stage 1 payload paths under `shards/...`;
- `build_status.json` and `progress.jsonl`;
- lazy records pointing to `{path, row}`;
- lazy LRU probe with `load_count` and `hit_count`;
- lazy runtime invalidity exclusion from `sample_global`;
- Stage 3 sampler fake path loading the lazy dataset.

Current evidence does not yet prove:

- live IsaacLab Stage 1 writes shard files while the process is running;
- live IsaacLab Stage 3 reads a large cache without memory pressure;
- final throughput is acceptable for a full AMASS cache.

### Known Gaps

- `cache_chunk_size` currently has a code default but no CLI/run-script knob.
- `shard_cache_size` currently has a code default but no CLI/run-script knob.
- `torch_chunked_shard` still loads a full shard file per cache miss.
- Lazy `get_segments()` currently loops records row-by-row instead of grouping
  sampled rows by shard path.
- The eager compatibility function remains available and must not be used as
  the Stage 3 production path.

### Required Next Steps

1. Expose `cache_chunk_size` through CLI and Stage 1 run script.
2. Expose `shard_cache_size` through CLI/config and Stage 3 run script.
3. Strengthen the Stage 1 contract test so chunk flush visibility is asserted
   at the writer boundary, not only after the full build.
4. Strengthen the Stage 3 lazy-read contract so `segment_id`, `path`, `row`,
   LRU probes, and batch semantics are checked together.
5. Run a server-side Stage 1 live sentinel and inspect shard files while the
   process is still running.
6. Run a server-side Stage 3 tiny update loop and inspect lazy cache logs plus
   PPO valid-count logs.

### Stop Conditions Before Formal Full Stage 1

Do not recommend full Stage 1 cache generation until:

- CLI prints the effective `cache_chunk_size`;
- Stage 1 progress logs include non-empty `flushed_shard_path`;
- shard files are visible during a live server run before process exit;
- validator passes after completion.

### Stop Conditions Before Formal Stage 3

Do not recommend full Stage 3 training until:

- CLI prints the effective `shard_cache_size`;
- Stage 3 logs show `lazy=True`;
- shard-cache probe starts with `load_count=0`;
- a tiny update loop reaches `ppo_update=True` and `ppo_valid_count > 0`;
- sampler evidence updates priority from per-sample rollout evidence.

## 18. Step 7 Server Stage 1 Sentinel Contract

Date: 2026-06-30

Step 7 is a live sentinel step.  It does not prove full-cache throughput.  It
only proves that the real IsaacLab Stage 1 path reaches the streaming cache
boundaries and exits cleanly.

### Scope

- use the repository Stage 1 wrapper instead of hand-written command fragments;
- run one or a few motions with a small segment count;
- write to a disposable sentinel cache directory;
- require validator after completion;
- inspect live progress and shard files before recommending a formal run.

### Non-Scope

- no formal full AMASS cache generation;
- no Stage 2 or Stage 3 training;
- no claim about final throughput;
- no acceptance/old Stage 2 path.

### Command

From `/hdd1/cyx/FEMR` on the server:

```text
CUDA_VISIBLE_DEVICES=0 \
RUN_FOREGROUND=1 \
LOG_PATH=/hdd1/cyx/FEMR/train_stage1_segment_cache_sentinel.txt \
MAX_MOTIONS=1 \
MAX_SEGMENTS=4 \
CACHE_CHUNK_SIZE=2 \
VARIANTS_PER_STRENGTH=1 \
VALIDATION_MIN_SEGMENTS=1 \
VALIDATION_MIN_NOISY=1 \
bash run_stage1.sh \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  1 \
  4 \
  /hdd1/cyx/AMASS_G1Segment_sentinel
```

Before launching IsaacLab, the command can be checked without side effects:

```text
FRONTRES_STAGE1_PREFLIGHT_ONLY=1 \
MAX_MOTIONS=1 \
MAX_SEGMENTS=4 \
CACHE_CHUNK_SIZE=2 \
bash run/run_frontres_stage1_segment_cache.sh \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  1 \
  4 \
  /hdd1/cyx/AMASS_G1Segment_sentinel
```

Expected preflight log:

```text
[FrontRES Stage1 startup preflight] PASS
```

### Required Runtime Probes

The Stage 1 sentinel log must include:

- `[FrontRES Stage1 Segment Cache] live_sentinel`;
- `cache_chunk_size=2`;
- `[FrontRES Stage1 Segment Cache] stage1_cfg_probe`;
- `[FrontRES Stage1 Segment Cache] motion_loader_probe`;
- `[FrontRES Stage1 Segment Cache] index_source`;
- `[FrontRES Stage1 Segment Cache] perturbation_plan`;
- `[FrontRES Stage1 Segment Cache] cache_readback`;
- `[FrontRES Stage1 Segment Cache] auto_exit`;
- validator `PASS`.

The disposable cache directory must include:

- `build_status.json`;
- `progress.jsonl`;
- `metadata.json`;
- `segment_index.jsonl`;
- `shards/clean_states/shard_000000.pt`;
- at least one `shards/noisy_variants/*/shard_000000.pt`;
- `manifests/clean_states/shard_000000.pt`;
- at least one `manifests/noisy_variants/*/shard_000000.pt`.

### Stop Condition

Step 7 is complete only when the server log and cache directory show the
runtime probes above.  Local contract tests only prove the command contract;
they do not replace the server sentinel.

## 19. Step 8 Server Stage 3 Tiny Update-Loop Contract

Date: 2026-06-30

Step 8 is a live sentinel step for Stage 3.  It uses a tiny Stage 1 cache and
an HSL checkpoint only to prove the real Stage 3 update-loop path reaches lazy
cache loading, sampler batching, rollout evidence, and PPO update.

### Scope

- use the repository Stage 3 wrapper;
- load a real HSL checkpoint;
- use the disposable Stage 1 sentinel cache;
- run `MODE=update_loop`;
- keep `NUM_ENVS`, `MAX_ITERS`, and `UPDATE_STEPS` tiny;
- inspect Stage 3 logs for lazy cache and PPO-valid evidence.

### Non-Scope

- no formal Stage 3 training;
- no reward-quality conclusion;
- no full AMASS cache;
- no old acceptance training path.

### Command

From `/hdd1/cyx/FEMR` on the server:

```text
CUDA_VISIBLE_DEVICES=0 \
RUN_FOREGROUND=1 \
LOG_PATH=/hdd1/cyx/FEMR/train_stage3_segment_hrl_tiny_update_loop.txt \
CACHE_DIR=/hdd1/cyx/AMASS_G1Segment_sentinel \
SHARD_CACHE_SIZE=2 \
bash run_stage3.sh \
  /hdd1/cyx/FEMR/model/model_warmup.pt \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  8 \
  1 \
  3 \
  update_loop
```

Before launching IsaacLab, the command can be checked without side effects:

```text
FRONTRES_STAGE_PREFLIGHT_ONLY=1 \
CACHE_DIR=/hdd1/cyx/AMASS_G1Segment_sentinel \
SHARD_CACHE_SIZE=2 \
bash run_stage3.sh \
  /hdd1/cyx/FEMR/model/model_warmup.pt \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  8 \
  1 \
  3 \
  update_loop
```

Expected preflight log:

```text
[FrontRES Stage3 startup preflight] PASS mode=update_loop
```

### Required Runtime Probes

The Stage 3 tiny update-loop log must include:

- `stage=stage3_segment_hrl`;
- `objective=segment_replay_hrl`;
- `segment_cache_dir=/hdd1/cyx/AMASS_G1Segment_sentinel`;
- `segment_shard_cache_size=2`;
- `[FrontRES Segment Dataset] cache_load`;
- `lazy=True shard_cache_size=2`;
- `[FrontRES Segment Dataset Ready]`;
- `shard_cache`;
- `[FrontRES Segment Sampler Ready]`;
- `[probe step22] sample`;
- `[FrontRES Segment Batch]`;
- `[FrontRES Segment Live Probe]`;
- `ppo_update=True`;
- `ppo_valid_count=` with a value greater than zero;
- `[probe step14] evidence_path`;
- `[FrontRES Segment Sampler]`;
- `[FrontRES Segment Live Update Loop]`;
- `runner_learn=False`.

### Stop Condition

Step 8 is complete only when the server log proves:

- Stage 3 uses the sentinel cache path;
- lazy cache loading is active;
- the sampler supplies a batch from cache;
- at least one PPO-valid sample is produced;
- the update loop exits through the sentinel path rather than normal training.

Local command contract tests only prove startup wiring.  They do not replace
the server tiny update-loop run.

## 20. Stage 1 Resumable Cache Contract

Date: 2026-06-30

This section defines the required design before a formal full-AMASS Stage 1
cache run.  The current Stage 1 code already has chunked payload writes,
`build_status.json`, and `progress.jsonl`.  That is not enough for true
checkpoint resume.  A resumable cache must treat each committed shard plus its
manifest record as the smallest recoverable unit.

### Problem

Stage 1 may run for a long time and may be interrupted by job timeout, server
restart, manual stop, or IsaacLab failure.  Restarting from zero wastes rollout
time.  Reusing a partially written cache without validation is also unsafe.

The current safe contract should be:

- a finished shard is committed atomically;
- a manifest row is the source of truth for resume;
- `progress.jsonl` is only an observability log;
- incomplete temp shards must be ignored;
- already committed Clean/Noisy records must be skipped on rerun;
- incomplete or corrupt records must be regenerated;
- Stage 3 must only consume a complete cache unless an explicit partial-cache
  mode is added later.

### Non-Scope

This contract does not require:

- HDF5, Zarr, mmap, or row-level random write;
- distributed writers;
- parallel IsaacLab workers;
- Stage 3 training from a partial cache;
- changing Segment Replay reward, sampler, or PPO semantics;
- using `progress.jsonl` as the canonical resume ledger.

### Mature Design Adaptation

The adapted pattern is the standard dataset-build pattern used by mature data
pipelines:

- write output to a temporary file first;
- validate the file before it becomes visible as a committed artifact;
- atomically rename temporary output to final output;
- record committed rows in a manifest or index;
- on rerun, scan committed manifest/shard pairs and rebuild only missing work;
- optionally compare a build signature so stale caches are not reused under a
  different configuration.

For FEMR, the unit is not a whole dataset.  The unit is:

```text
Clean: segment_key -> clean shard row -> clean manifest row
Noisy: segment_key + perturbation_key -> noisy shard row -> noisy manifest row
```

### Core Parameter Path

Clean resume path:

```text
FrontRESSegmentIndex
-> segment_key = (motion_rel_path, start_frame, end_frame)
-> clean capture
-> temp clean shard
-> shard validation
-> atomic rename to final clean shard
-> clean manifest row {path, row, segment}
-> resume scan completed_clean_keys
-> builder skips completed clean segment on rerun
```

Noisy resume path:

```text
FrontRESPerturbationDescriptor
-> noisy_key = (segment_key, perturbation_id, strength, seed, family_group, role)
-> noisy capture
-> temp noisy shard
-> shard validation
-> atomic rename to final noisy shard
-> noisy manifest row {path, row, segment, perturbation}
-> resume scan completed_noisy_keys
-> builder skips completed noisy variant on rerun
```

Build signature path:

```text
amass_root + horizon_k + frame_stride + loaded_motion_paths
+ perturbation curriculum config + cache schema version
-> build_signature
-> cache resume allowed only when signature matches
```

### File Responsibility Map

`source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py`

- owns atomic shard write helpers;
- owns manifest read/write helpers;
- owns committed Clean/Noisy key extraction;
- must ignore `.tmp` files during resume scans;
- must expose a compact probe with committed row counts and shard counts.

`source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py`

- owns builder-level resume policy;
- scans existing committed records before rollout;
- computes pending Clean/Noisy work;
- skips completed work;
- writes resume probes before the expensive env loop starts;
- must not trust progress logs as completion evidence.

`source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_validator.py`

- owns complete/partial/corrupt classification;
- verifies manifest rows point to existing final shards;
- verifies row indexes are readable;
- verifies expected Clean/Noisy counts after a complete build.

`scripts/rsl_rl/check_frontres_stage1_segment_cache_completion.py`

- owns server-side completion check against the expected AMASS segment set;
- should report missing, extra, partial, and corrupt counts separately.

`scripts/rsl_rl/train.py` and `run_stage1.sh`

- own user-facing resume/force-rebuild flags only;
- must pass explicit config into the builder;
- must not implement manifest parsing or skip logic.

### Invariants

- `progress.jsonl` never decides whether a segment is complete.
- A final shard path without a manifest row is not resumable.
- A manifest row whose shard is missing or unreadable is corrupt, not complete.
- Temporary shard files are ignored by resume and may be cleaned later.
- Rerunning Stage 1 with the same signature should not overwrite committed
  shard rows.
- Rerunning Stage 1 with a different signature must fail fast unless
  `force_rebuild=True`.
- Resume scan must print:
  `completed_clean`, `completed_noisy`, `pending_clean`, `pending_noisy`,
  `ignored_tmp`, and `corrupt_count`.
- A partial cache may become complete after rerun.
- A corrupt cache must be regenerated for the corrupt rows or rejected; it must
  not silently train Stage 3.
- Stage 3 default loading still requires `metadata.json` with complete status.

### Runtime Probes

Before the first env rollout, Stage 1 must print a resume probe:

```text
[FrontRES Stage1 Resume Probe]
signature_match=True
completed_clean=...
completed_noisy=...
pending_clean=...
pending_noisy=...
ignored_tmp=...
corrupt_count=...
resume_enabled=True
force_rebuild=False
```

After each shard commit, Stage 1 must print or log a compact commit event:

```text
[FrontRES Stage1 Shard Commit]
kind=clean/noisy
shard_path=...
row_count=...
manifest_path=...
committed_total=...
```

These probes are live sentinels.  They prove the real IsaacLab path reaches the
resume boundary.  They do not prove training quality.

### Tests Required Before Code Is Considered Complete

Local semantic tests:

1. temp shard exists but no manifest row -> resume scan reports zero completed;
2. final shard exists with manifest row -> resume scan reports completed key;
3. manifest row points to missing shard -> corrupt_count increases;
4. partial cache with two completed segments -> builder rerun only requests
   the remaining segments;
5. changed build signature -> resume fails unless force rebuild is set.

Server sentinel tests:

1. run Stage 1 tiny cache with small `CACHE_CHUNK_SIZE`;
2. interrupt after at least one shard commit;
3. rerun the same command;
4. confirm log shows completed keys and fewer pending keys;
5. confirm final validator passes.

### Stop Conditions Before Formal Full Stage 1

Do not recommend full Stage 1 cache generation until:

- resume scan is implemented and covered by a semantic local test;
- shard commit uses a temp path plus final commit boundary;
- committed manifest rows are written during generation, not only at the end;
- rerun skips already committed Clean and Noisy records;
- corrupt/missing shard rows are not treated as complete;
- a server tiny run proves interruption and rerun behavior.

### Next Step

Step 2 should implement cache IO resume primitives first.  It should not touch
IsaacLab runner logic.  The owner module is
`frontres_segment_cache_io.py`, and the test target is a semantic fake cache
with temp, committed, missing, and corrupt shard cases.

## 21. Step 2 Result: Cache IO Resume Primitives

Date: 2026-06-30

Step 2 implemented only the IO-layer primitives.  It did not connect resume to
the Stage 1 builder or live IsaacLab path.

### Implemented

`frontres_segment_cache_io.py` now owns:

- `FrontRESStage1CacheResumeScan`;
- `clean_resume_key(...)`;
- `noisy_resume_key(...)`;
- `scan_stage1_cache_resume_state(...)`;
- `write_clean_state_chunked_shard_atomic(...)`;
- `write_noisy_variant_chunked_shard_atomic(...)`.

### Verified Core Path

The semantic contract test checks:

```text
tmp shard + progress log only
-> completed_clean=0
-> completed_noisy=0
-> ignored_tmp=1

final shard + manifest rows
-> completed_clean=2
-> completed_noisy=2
-> corrupt_count=0

manifest row pointing to missing shard
-> corrupt_count=1
```

### Evidence

Fresh local commands:

```text
python -m py_compile \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py
```

Observed probe:

```text
[cache_io resume trace] tmp_only={'completed_clean': 0, 'completed_noisy': 0,
'ignored_tmp': 1, 'corrupt_count': 0, ...}
[cache_io resume trace] committed ... probe={'completed_clean': 2,
'completed_noisy': 2, 'ignored_tmp': 1, 'corrupt_count': 0, ...}
[cache_io resume trace] corrupt={'completed_clean': 2, 'completed_noisy': 2,
'ignored_tmp': 1, 'corrupt_count': 1, ...}
PASS: FrontRES Segment cache IO round-trips clean states and noisy variants.
```

### Remaining Gap

Stage 1 builder still does not call the resume scanner and still does not skip
completed work.  That belongs to Step 3.

## 22. Step 3 Result: Builder Resume Planner

Date: 2026-06-30

Step 3 connected the IO-layer resume scan to the Stage 1 builder.  It still
does not implement the final live IsaacLab interruption sentinel.

### Implemented

`frontres_segment_cache_builder.py` now:

- calls `scan_stage1_cache_resume_state(cache_dir)` after indexing and
  perturbation-plan construction;
- builds expected Clean keys from planned segments;
- builds expected Noisy keys from planned descriptors;
- loads committed Clean/Noisy manifest rows that match the current plan;
- starts new payload shard ids after the committed shard ids;
- reuses cached Clean state when Clean is already committed;
- skips a whole segment when Clean and all planned Noisy variants are already
  committed;
- writes a `[FrontRES Stage1 Resume Probe]` before the expensive env loop.

### Verified Core Path

The builder contract test performs:

```text
run 1: max_segments=1
-> completed_clean=1
-> completed_noisy=2

run 2: same cache dir, max_segments=2
-> resume scan sees completed_clean=1 and completed_noisy=2
-> builder only prepares segment 1
-> builder only captures Noisy variants for segment 1
-> final manifest contains segment 0 and segment 1
```

Observed probe:

```text
[FrontRES Stage1 Resume Probe] ... completed_clean=1 completed_noisy=2
pending_clean=1 pending_noisy=2 ...
[cache_builder resume trace] prepare_calls=[(1, 2, [0])]
perturb_calls=[(1, 2, 0.0, [0]), (1, 3, 0.5, [0])]
clean_ids=[0, 1]
zero_ids=[(0, 0), (1, 2)]
half_ids=[(0, 1), (1, 3)]
```

### Evidence

Fresh local commands:

```text
python -m py_compile \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py
```

All passed.

### Remaining Gap

Committed payload shards are resumable, and final manifests preserve old plus
new rows after rerun.  However, the builder still writes the canonical manifest
at the end of the run.  The next step should make shard commits immediately
visible through manifest records during generation, so an interruption after a
flush can be resumed without waiting for final metadata.

## 23. Step 4 Result: Flush-Visible Manifest Commit

Date: 2026-06-30

Step 4 made each Stage 1 shard flush immediately visible to resume scan.  It
still does not prove a live IsaacLab interruption; that remains a server
sentinel step.

### Implemented

`frontres_segment_cache_io.py` now writes Clean and Noisy manifest payloads via
the same temp-file plus atomic-replace primitive used by payload shards.

`frontres_segment_cache_builder.py` now commits the canonical manifest after
each buffer flush:

```text
buffer
-> atomic payload shard
-> append records in memory
-> atomic manifest rewrite
-> [FrontRES Stage1 Shard Commit]
```

The manifest remains the resume source of truth.  `progress.jsonl` remains only
observability.

### Verified Core Path

The builder contract now simulates:

```text
run 1:
  segment 0 Clean + Noisy flush
  crash before segment 1 prepare

resume scan after crash:
  completed_clean=1
  completed_noisy=2
  corrupt_count=0

run 2:
  resume scan sees segment 0 committed
  builder prepares only segment 1
  final manifest contains segment 0 and segment 1
```

Observed probe:

```text
[cache_builder crash_resume trace] after_crash
probe={'completed_clean': 1, 'completed_noisy': 2, 'ignored_tmp': 0,
'corrupt_count': 0, 'clean_manifest_count': 1, 'noisy_manifest_count': 2}

[cache_builder crash_resume trace] after_rerun
probe={'completed_clean': 2, 'completed_noisy': 4, ...}
prepare_calls=[(1, 2, [0])]
baseline_calls=[(1, 2, [0]), (1, 3, [0])]
```

### Evidence

Fresh local commands:

```text
python -m py_compile \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_io.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_io_contract.py
```

All passed.

### Remaining Gap

The local fake env proves the commit/resume contract.  The next required proof
is a server-side live Stage 1 sentinel:

```text
small CACHE_CHUNK_SIZE
-> observe [FrontRES Stage1 Shard Commit]
-> interrupt after at least one commit
-> rerun same command
-> observe completed_clean/completed_noisy > 0 before rollout
-> final validator passes
```

## 24. Step 5 Result: Build Signature Guard

Date: 2026-06-30

Step 5 added the smallest resume-safety guard: a cache can only resume when
the current Stage 1 build signature matches the signature stored by the cache.

### Implemented

`frontres_segment_cache_builder.py` now computes:

```text
amass_root
+ motion source
+ loaded motion paths when present
+ horizon_k
+ frame_stride
+ perturbation curriculum mode and levels
+ variants_per_strength
+ base_seed
+ curriculum active dims
-> stable JSON
-> sha256 build_signature.hash
```

The builder stores `build_signature` in:

- `build_status.json` after indexing;
- final `metadata.json`;
- final complete status.

Before resume scan, the builder compares the current signature with the
previous `metadata.json` signature, or with the previous `build_status.json`
signature when the cache was interrupted before final metadata.

### Verified Core Path

The builder contract now checks:

```text
run 1:
  horizon_k=2
  complete one segment

run 2:
  same cache_dir
  horizon_k=3
  signature mismatch
  fail before env prepare
```

Observed probe:

```text
[FrontRES Stage1 Resume Probe] signature_match=False
existing_hash=...
current_hash=...
resume_enabled=False
force_rebuild=False

[cache_builder signature trace]
status=signature_mismatch
prepare_calls=[]
```

The test also verifies that `segment_index.jsonl` is not rewritten on mismatch.

### Evidence

Fresh local commands:

```text
python -m py_compile \
  source/rsl_rl/rsl_rl/frontres/frontres_segment_cache_builder.py \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_cache_builder_contract.py
```

All passed.

### Deliberate Non-Scope

No `force_rebuild` implementation was added in this step.  That should only be
added when we decide the exact destructive behavior, because it may delete or
quarantine existing cache files.

## 25. Step 6 Result: CLI And Run-Script Knob Audit

Date: 2026-06-30

Step 6 checked whether the Stage 1/Stage 3 runtime knobs already exist before
adding more code.  They do exist, so no implementation change was needed.

### Verified Path

Stage 1 chunking path:

```text
CACHE_CHUNK_SIZE
-> run/run_frontres_stage1_segment_cache.sh
-> --frontres_segment_cache_chunk_size
-> scripts/rsl_rl/train.py
-> FrontRESStage1CacheBuilderConfig.cache_chunk_size
-> live sentinel prints cache_chunk_size
```

Stage 3 lazy shard cache path:

```text
SHARD_CACHE_SIZE
-> run/run_frontres_stage3_segment_hrl.sh
-> --frontres_segment_shard_cache_size
-> scripts/rsl_rl/train.py
-> alg_cfg.frontres_segment_shard_cache_size
-> FrontRESSegmentDataset lazy shard cache
```

### Test Class

This is a secondary contract path.  It routes simple scalar config values, not
a tensor parameter transformed across formulas or masks.  Semantic asserts and
the existing sampler probe are sufficient.

### Evidence

Fresh local commands:

```text
/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
```

Observed facts:

```text
PASS: FrontRES Stage 1/2 live presets and Stage 3 Segment Replay contract are explicit.
[probe step23] shard_cache_size: alg_value=1 metadata=... 'shard_cache': {'max_shards': 1, ...}
frontres_segment_live_sampler_contract: ok
```

### Stop Condition

The run scripts, train entrypoint, algorithm config, and lazy dataset path all
expose the required knobs.  Step 6 is complete without new code.

## 26. Step 7 Local Result: Stage 1 Sentinel Preflight

Date: 2026-06-30

Step 7 is a live sentinel path, so local execution can only verify the startup
contract.  It cannot replace the server IsaacLab run.

### Test Class

Live sentinel path.  The required live facts are the Stage 1 cache log, shard
files, manifest files, and validator result produced on the server.

### Local Evidence

Fresh local preflight commands:

```text
FRONTRES_STAGE1_PREFLIGHT_ONLY=1 \
MAX_MOTIONS=1 \
MAX_SEGMENTS=4 \
CACHE_CHUNK_SIZE=2 \
VARIANTS_PER_STRENGTH=1 \
VALIDATION_MIN_SEGMENTS=1 \
VALIDATION_MIN_NOISY=1 \
bash run/run_frontres_stage1_segment_cache.sh \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  1 \
  4 \
  /hdd1/cyx/AMASS_G1Segment_sentinel

FRONTRES_STAGE1_PREFLIGHT_ONLY=1 \
RUN_FOREGROUND=1 \
LOG_PATH=/private/tmp/femr_stage1_step7_preflight.txt \
MAX_MOTIONS=1 \
MAX_SEGMENTS=4 \
CACHE_CHUNK_SIZE=2 \
VARIANTS_PER_STRENGTH=1 \
VALIDATION_MIN_SEGMENTS=1 \
VALIDATION_MIN_NOISY=1 \
bash run_stage1.sh \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  1 \
  4 \
  /hdd1/cyx/AMASS_G1Segment_sentinel

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python \
  source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py
```

Observed facts:

```text
[FrontRES Stage1 startup preflight] PASS
--frontres_stage stage1_segment_cache
--frontres_segment_cache_max_motions 1
--frontres_segment_cache_max_segments 4
--frontres_segment_cache_chunk_size 2
--frontres_segment_cache_dir /hdd1/cyx/AMASS_G1Segment_sentinel
[FrontRES Stage1 validator preflight] enabled cache_dir=/hdd1/cyx/AMASS_G1Segment_sentinel expect_mode=hrl_curriculum_bank min_segments=1 min_noisy=1
PASS: FrontRES Stage 1/2 live presets and Stage 3 Segment Replay contract are explicit.
```

### Server Stop Condition

Step 7 is not fully complete until the server run shows:

- `[FrontRES Stage1 Resume Probe]`;
- `[FrontRES Stage1 Shard Commit]`;
- `[FrontRES Stage1 Segment Cache] cache_readback`;
- `[FrontRES Stage1 Segment Cache] auto_exit`;
- validator PASS;
- committed shard and manifest files under the disposable cache directory.
