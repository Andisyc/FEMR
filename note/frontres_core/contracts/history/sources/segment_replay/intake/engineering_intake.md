# FrontRES Segment Replay Engineering Intake

Date: 2026-06-27

This note records the current FEMR implementation before adding Segment Replay
HRL.  It is not a new design contract.  Its purpose is to prevent the next
engineering step from being built on a wrong mental model of the live code.

## 1. Current Live Training Entry

The live training command enters through:

- `scripts/rsl_rl/train.py`
- Hydra task config
- `whole_body_tracking.utils.my_on_policy_runner.MotionOnPolicyRunner`
- `rsl_rl.runners.on_policy_runner.OnPolicyRunner`

Important entry behavior:

- `train.py` forces local FEMR source paths into `sys.path`, so this checkout is
  intended to win over installed packages or MOSAIC source trees.
- `--frontres_stage` currently has only two choices:
  - `stage1_hsl`
  - `stage2_acceptance`
- log root is:
  - default: this FEMR repo root
  - override: `FEMR_LOG_ROOT`
- run directory is:
  - `{FEMR_LOG_ROOT or repo_root}/{experiment_name}/{timestamp}_{run_name}`
- checkpoint loading can use direct `student_checkpoint_path`; otherwise it
  resolves inside the current experiment log root.

Current stage presets:

- `stage1_hsl`
  - experiment defaults to `g1_flat_frontres_stage1_hsl`
  - objective becomes `supervised_restore`
  - actor is trained by supervised restore
  - authority actor-critic is disabled
  - structured joint RL is disabled
  - exits after warmup if `frontres_stage1_exit_after_warmup=True`

- `stage2_acceptance`
  - experiment defaults to `g1_flat_frontres_stage2_acceptance`
  - objective becomes `hsl_hybrid`
  - `frontres_acceptance_preference_weight=1.0`
  - split acceptance head is enabled
  - authority actor-critic is disabled
  - structured joint RL is disabled
  - temporal perturbation mode is forced to one-step/single

Conclusion:

- The live CLI still expresses the old two-stage HSL plus acceptance design.
- There is no Segment Replay stage flag yet.
- There is no live segment dataset, segment sampler, dynamic reset adapter, or
  segment replay PPO route yet.

## 2. Current Config Surface

Main config classes and defaults:

- `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py`
  - `RslRlFrontRESUnifiedAlgorithmCfg`
  - default `frontres_training_objective="hsl_hybrid"`
  - default `frontres_acceptance_preference_weight=0.0`
  - default `frontres_hsl_rollout_label_enabled=False`
  - structured joint RL and authority actor-critic default off

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`
  - live G1 FrontRES config overrides the generic defaults
  - objective is `hsl_hybrid`
  - `frontres_acceptance_preference_weight=1.0`
  - `frontres_split_acceptance_head=True`
  - `frontres_hsl_rollout_label_enabled=True`
  - active task dims include proposal dims and acceptance dims

Conclusion:

- The task config is already specialized for HSL proposal plus acceptance.
- New Segment Replay HRL should not reuse `stage2_acceptance` silently.
- It needs a new explicit mode or stage flag, otherwise old acceptance logic will
  remain active and confuse diagnostics.

## 3. Runner Construction

`OnPolicyRunner.__init__` owns these decisions:

- algorithm class decides training type:
  - `FrontRESUnified` -> `frontres`
- policy class is created from config through `eval`
- FrontRES policy can use GMT normalizer and partial obs normalization
- algorithm gets runner normalizers
- FrontRES helper modules are instantiated:
  - `FrontRESAlphaRhoBridge`
  - `FrontRESActionCone`
  - `FrontRESExecutabilityScorer`
- rollout storage action shape uses `policy.total_output_dim` if present

Important current guard:

- `_validate_frontres_active_hsl_acceptance_path()` verifies the active HSL
  acceptance path before rollout.

Conclusion:

- Runner is already the integration point.
- Runner is too large, but it already delegates several FrontRES pieces to
  modules.
- Segment Replay should add a thin runner adapter only.  Segment dataset,
  sampler, reset, reward, action construction, and diagnostics should live in
  separate modules.

## 4. Current Policy Object

Current FrontRES policy class:

- `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`
- `FrontRESActorCritic`

Current output semantics:

- task-space mode uses `num_task_corrections=6`
- `task_conf_dim` can be:
  - `1`: scalar trust
  - `2`: legacy pos/rpy coefficients
  - `6`: per-axis coefficients
- when `frontres_split_acceptance_head=True`:
  - residual actor outputs the HSL proposal
  - separate `acceptance_actor` outputs acceptance logits
- optional `authority_actor` exists, but it is not active in current stage
  presets.

Conclusion:

- Current policy already has the old split HSL/acceptance structure.
- The new HRL repair policy should not be implemented as another acceptance
  head.
- The cleanest next policy target is either:
  - reuse residual actor as full 6D repair policy after HSL initialization; or
  - add a clearly named repair actor/residual mode, guarded by a new config.

## 5. Rollout Construction

Current rollout path:

- `OnPolicyRunner.learn()`
- `prepare_frontres_rollout_step()`
- environment step
- FrontRES reward/evidence construction
- `alg.process_env_step()`
- `alg.update()`

`prepare_frontres_rollout_step()` currently:

- calls `alg.act()`
- masks FrontRES task actions
- rewrites task-space log-prob
- optionally applies authority rollout action
- applies task correction before asking GMT for env action
- writes supervised target before step when enabled
- captures HSL correction snapshot before step when rollout labels are enabled

Current environment layout still uses split env groups:

- train/FrontRES envs
- candidate envs
- Noisy/GMT baseline envs
- Clean envs

Conclusion:

- Current rollout is built around parallel Clean/Noisy/Candidate comparison
  inside one rollout batch.
- Segment Replay needs a different object: a replayable segment task with a
  dynamic reset state and K-step local consequence horizon.
- The current split-env logic may provide useful reward/evidence code, but it is
  not the segment replay environment.

## 6. Current HSL Target

Current HSL rollout label builder:

- `source/rsl_rl/rsl_rl/runners/frontres_hsl_rollout_target.py`
- `build_frontres_hsl_rollout_target()`

What it does:

- reads FrontRES, Noisy, and Clean root states after rollout
- computes FrontRES-to-Clean residual
- adds this residual to current correction
- projects the label through the action cone
- masks labels by perturbation mode
- builds sample weights:
  - repair weight when Noisy is in a repairable range
  - no-op/harm weight when Noisy is safe, broken, or FrontRES hurts
- in `hsl_hybrid`, writes:
  - supervised target = full HSL label
  - supervised weight = repair weight
  - supervised harm weight = no-op/harm weight

Physical meaning:

- HSL target is still a Clean-oriented correction direction.
- It is partly rollout-calibrated, but its role is still proposal learning.
- It does not directly learn a full dynamic repair policy from segment-level
  trial-and-error.

Conclusion:

- HSL remains useful as initialization.
- It should not define the final HRL object in the new design.

## 7. Current Acceptance Label / Mask Path

Current acceptance payload owner:

- `source/rsl_rl/rsl_rl/frontres/frontres_transition_payload.py`
- `_write_active_hsl_acceptance_payload()`
- `build_and_write_frontres_acceptance_payload()`

What it does:

- uses Candidate-vs-Noisy executable margin from rollout context
- converts margin into accept / reject / ignore labels
- expands label to 6 task dimensions
- masks by perturbation mode and active task dims
- writes these fields into transition:
  - `acceptance_action`
  - `acceptance_logit`
  - `acceptance_prob`
  - `acceptance_gt`
  - `acceptance_mask`
  - `acceptance_margin`
  - `acceptance_target`

Current acceptance loss owner:

- `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`
- `_compute_acceptance_preference_loss()`

What the loss trains:

- it reads policy acceptance logits from `mu[:, 6:12]`
- it reads rollout label from `acceptance_gt` / `acceptance_target`
- it reads valid dimensions from `acceptance_mask`
- it applies BCE with optional class balance and focal weighting
- it only runs when:
  - acceptance preference weight > 0
  - acceptance target/mask exist
  - objective is `hsl_hybrid`
  - task confidence mode is acceptance-only
  - authority actor-critic is not active

Conclusion:

- The current Stage 2 training signal is acceptance over an HSL proposal.
- It is not PPO over full 6D repair.
- This path should be treated as an old baseline/ablation when Segment Replay
  HRL is added.

## 8. Current Storage Contract

Current storage:

- `source/rsl_rl/rsl_rl/storage/rollout_storage.py`

FrontRES storage already has many fields:

- PPO fields:
  - observations
  - privileged observations
  - actions
  - values
  - returns
  - advantages
  - old log-probs
  - old mean/std
- supervised HSL fields:
  - `supervised_target`
  - `supervised_weight`
  - `supervised_harm_weight`
- acceptance fields:
  - `acceptance_action`
  - `acceptance_logit`
  - `acceptance_prob`
  - `acceptance_gt`
  - `acceptance_mask`
  - `acceptance_margin`
  - `acceptance_target`
- older authority/rho fields:
  - `proposal_delta_se`
  - `authority_action`
  - `authority_log_prob`
  - `authority_rho`
  - `authority_return_k`
  - `authority_return_zero_k`
  - `authority_return_one_k`
  - `authority_mask`
  - event metadata

Conclusion:

- Storage is already overloaded.
- Segment Replay should not casually add more unrelated fields to the same
  transition tuple.
- If segment PPO needs storage, define the exact new tuple first:
  - segment id
  - segment phase/start
  - perturbation family/strength
  - 6D repair action
  - log-prob
  - K-step return
  - valid mask
  - replay priority evidence
- If these do not fit current storage cleanly, add a dedicated segment storage
  object instead of expanding the old acceptance tuple again.

## 9. Current Algorithm Update

Current algorithm:

- `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`
- `FrontRESUnified`

Current update modes:

- generic PPO + supervised restore
- HSL hybrid acceptance BCE path
- structured joint rho path
- authority actor-critic path
- state alpha auxiliary path

Important current behavior:

- `hsl_hybrid` disables generic PPO actor update for the old acceptance path.
- supervised loss trains proposal direction.
- acceptance BCE trains the acceptance head.
- value loss still uses critic/returns.
- gradient guards exist to isolate old acceptance/state-router/authority paths.

Conclusion:

- There are useful utilities here, but the class is already dense.
- Segment Replay HRL should not be added as another large branch inside
  `_update_ppo_supervised()` unless the branch is only a thin dispatch.
- A new module should own segment PPO loss/update semantics, or a new algorithm
  class should be created if the route diverges too much from the old path.

## 10. Checkpoint Behavior

Current checkpoint owner:

- `source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py`

Current save behavior:

- saves residual actor
- saves acceptance actor if present
- saves critic/optimizer/normalizers/iteration state
- records FrontRES checkpoint probe metrics
- can copy a best probe checkpoint

Current load behavior:

- supports Stage 1 -> Stage 2 migration
- can map old `student` weights into `residual_actor`
- can migrate two-head residual actor into split Stage 2 proposal actor
- can initialize missing split `acceptance_actor`
- can load or skip critic/optimizer depending on resume mode
- treats GMT normalizer as frozen
- can load Stage 1 anchor-error normalizer stats for extra obs dims

Conclusion:

- HSL initialization is already supported in spirit.
- New Segment HRL should define checkpoint migration deliberately:
  - HSL proposal weights initialize the 6D repair actor
  - old acceptance actor should not be required
  - optimizer/critic reset behavior should be explicit

## 11. Diagnostics

Current diagnostics are split across:

- `frontres_runner_logging.py`
- `frontres_diagnostics.py`
- `frontres_reward_diagnostics.py`
- checkpoint probe JSONL

Existing useful diagnostics:

- HSL acceptance loss enabled/path enabled
- acceptance target/mask/prob/error/correlation
- reward delta/gain/harm
- safe/fragile/broken distributions
- active dims and perturbation modes
- authority fields if authority mode is active

Conclusion:

- New Segment Replay needs its own diagnostics.
- Reusing acceptance labels will confuse interpretation.
- Required new diagnostics:
  - global/replay/review sample ratio
  - replay pool size
  - segment id or segment bucket
  - K-step horizon
  - Noisy score
  - repaired score
  - repair gain over Noisy
  - fall rate
  - contact consistency
  - 6D action norm by dimension
  - replay priority distribution
  - solved / active / hopeless counts

## 12. Current Useful Boundaries

Reusable boundaries:

- `frontres_action_cone.py`
  - action projection
  - active dimension masks
  - per-mode masks
- `frontres_executability.py`
  - executable scoring ideas
- `frontres_reward_window.py`
  - Clean/Noisy/Repaired reward context ideas
- `frontres_post_step_connector.py`
  - reward/evidence connection after env step
- `frontres_checkpointing.py`
  - HSL-to-next-stage checkpoint migration ideas
- `frontres_runner_logging.py`
  - diagnostic routing style
- `frontres_training_setup.py`
  - curriculum/config setup style

Do not reuse directly as-is:

- acceptance label/mask semantics as the new HRL target
- split-env Clean/Noisy/Candidate layout as the segment replay design
- `hsl_hybrid` objective name for the new method
- acceptance head as the final HRL action object

## 13. Main Mismatch With Segment Replay HRL

Discussed target concept:

- slice long motions into replayable dynamic segments
- reset each task to a cached clean dynamic state or pre-roll state
- perturb segment reference/state
- HRL outputs full 6D `Delta SE`
- rollout K steps
- reward measures improvement over Noisy under executable metrics
- prioritized segment replay decides which segments are repeated
- HSL is initialization, not the final proposal-plus-acceptance object

Current live code:

- long rollout batch with split env groups
- HSL learns Clean-oriented proposal
- Stage 2 learns acceptance over proposal
- acceptance label comes from Candidate-vs-Noisy margin
- no segment dataset
- no segment sampler
- no dynamic reset adapter
- no segment replay priority
- no explicit segment K-step environment
- no full 6D repair PPO objective as the active stage

Conclusion:

- We can move to engineering, but only after writing a new engineering contract.
- The first code step should not touch `on_policy_runner.py` except for a thin
  future connector.
- The next step should map external code reuse into these module targets:
  - `frontres_segment_dataset.py`
  - `frontres_segment_sampler.py`
  - `frontres_segment_reset.py`
  - `frontres_segment_reward.py`
  - `frontres_hrl_action.py`
  - `frontres_segment_diagnostics.py`

## 14. Step 1 Result

Step 1 confirms:

- FEMR has a working old FrontRES architecture, but it is acceptance-centered.
- The codebase already has useful modular seams, but runner and algorithm are
  overloaded.
- Segment Replay HRL is not currently implemented.
- New work should begin as new modules plus tests, then connect to runner with a
  minimal adapter.
- Old acceptance path should stay available as a baseline/ablation, but it must
  default off for the new Segment Replay stage.

Recommended next step:

- Step 2: external code reuse map.
- The map should inspect motion library / dynamic reset / reference state
  sampling code before writing FEMR modules.
