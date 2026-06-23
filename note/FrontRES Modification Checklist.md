# FrontRES Modification Checklist

Use this checklist after every nontrivial FrontRES change.  The goal is to keep
concept, code path, diagnostics, and short-run evidence aligned.  Do not mark a
change as ready for training until each relevant item has concrete evidence.

## Active Change Record: 2026-06-23 Authority Actor-Critic Refactor

- [x] Design note updated:
  `note/FrontRES Design Contract.md`, section
  `2026-06-23 Authority Actor-Critic Contract`.
- [x] Concept sentence:
  FrontRES should be `Clean-oriented Delta SE proposal -> authority actor-critic
  over rho -> K-step executable return -> burst perturbation curriculum -> frozen
  GMT execution`; `rho` is an execution-authority action, not endpoint
  acceptance label and not sampled-rho PPO advantage.  Rechecked after Step 9:
  active config and diagnostics now name the objective as authority
  actor-critic, not PPO acceptance.
- [x] Stage-1 freeze contract decided:
  Stage 1 `Delta SE_HSL` is trained by HSL and treated as a fixed/detached
  proposal for Stage 2 authority.  Verify whether this is implemented by config
  freeze, optimizer parameter groups, checkpoint resume mode, or a new explicit
  Stage-2 training phase.  Launch contract added 2026-06-24:
  `run/run_frontres_stage1_hsl.sh` trains proposal only, and
  `run/run_frontres_stage2_authority.sh` transfers from the Stage 1 checkpoint
  with `--is_full_resume False` plus `algorithm.lambda_supervised=0.0`, so Stage
  2 authority training does not keep applying the HSL proposal loss.
  Stage 2 launch also sets `--supervised_warmup_iterations 0`, because Stage 1
  warmup has already been completed before checkpoint transfer.
  Launch scripts use CLI `--frontres_stage ...`, not Hydra deep overrides such
  as `algorithm.xxx` or `experiment_name=...`; `scripts/rsl_rl/train.py` applies
  Stage presets after Hydra has loaded the typed runner config.
- [x] Stage-2 critic warmup / actor takeover contract decided:
  Stage 2 is a new learning problem because the authority critic starts from
  scratch.  Reuse the old warmup/takeover idea with Stage-2-specific controls:
  `critic_warmup_iterations=200` keeps runner-side DR/curriculum conservative,
  `algorithm.frontres_authority_actor_warmup_iterations=200` disables authority
  actor loss while the critic warms up, and
  `algorithm.frontres_authority_actor_ramp_iterations=200` ramps actor takeover
  instead of switching it on abruptly.  Diagnostics must show
  `authority_actor_phase_weight`, `authority_actor_warmup_active`, and
  `authority_actor_ramp_active`.
- [x] Module 1, policy architecture:
  `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`.
  Current code already has `frontres_split_acceptance_head`; audit and adapt it
  so `acceptance_actor` explicitly receives `full/current state observation +
  detached Delta SE_HSL`, while Stage 1 proposal gradients do not flow through
  acceptance loss.  Audit result: this route already exists in
  `_frontres_raw_task_output`; no production code change was needed for Step 1.
- [x] Module 2, authority network surface:
  `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`, or a new
  `source/rsl_rl/rsl_rl/modules/frontres_authority.py`.
  Add the Stage-2 authority critic `Q_phi(state/proposal/rho)`.  Keep it
  separate from the existing PPO value critic, because the existing critic is
  `V(obs)` for environment returns, while the new critic owns proposal/rho
  executable value.  Implemented as optional network surface with checkpoint
  save/load support; runner, storage, and algorithm integration remain pending.
- [x] Module 3, authority action representation:
  implement the fixed authority space from the design contract: 6D continuous
  authority with `rho_i in [0, 1]` for each task-space correction dimension.
  Do not silently mix scalar, grouped, discrete, and per-axis rho semantics.
  Startup logs must print the active authority parameterization and active dim
  mask.  The earlier grouped-discrete helper/test is now obsolete prototype
  evidence, not the active mainline.  Implemented in
  `source/rsl_rl/rsl_rl/frontres/frontres_authority_space.py` and verified by
  `source/rsl_rl/rsl_rl/tests/frontres_authority_space.py`; startup route now
  prints authority actor-critic when enabled.
- [x] Module 4, config surface:
  `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`.
  Add explicit authority actor-critic flags and turn the old structured-rho
  advantage path off by default for the new experiment.  Preserve old modes as
  ablations only.  Implemented in the generic policy cfg
  `source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py`, generic
  algorithm cfg `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py`, and active G1
  config.  Active G1 now builds authority actor/critic, enables
  `frontres_authority_actor_critic_enabled`, and sets structured-rho enabled
  false with zero weight.
- [x] Module 5, rollout application:
  `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`,
  `source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py`, and
  `source/rsl_rl/rsl_rl/frontres/task_space_correction.py`.
  Ensure the executed correction is exactly `Delta SE_exec = rho * detached
  Delta SE_HSL` for the sampled authority action.  Training and inference must
  use the same authority space.  Implemented for training rollout by keeping
  Stage-1 proposal in action columns `[:6]` and writing authority rho into
  `[6:12]`, so existing task-space correction applies exactly one multiplication.
  Post-Step-9 audit also fixed the inference helper path so
  `get_task_correction_inference()` uses `authority_actor(full_obs, detached
  proposal)` instead of the legacy residual coefficient head when authority
  actor-critic is enabled.
- [x] Module 6, K-step executable return:
  create a small pure helper under `source/rsl_rl/rsl_rl/frontres/`, e.g.
  `frontres_authority_return.py`.  It builds done-masked K-step executable
  returns for event frames and supports optional detached bootstrap when a valid
  bootstrap state exists.  This is still a pure helper and is not yet wired into
  runner, storage, or algorithm.
- [x] Module 7, storage contract:
  `source/rsl_rl/rsl_rl/storage/rollout_storage.py`.
  Prefer new authority-specific fields rather than reusing
  `acceptance_target/mask`: `authority_action`, `authority_log_prob`,
  `authority_rho`, `authority_return_k`, `authority_mask`, and
  `proposal_delta_se`.  Implemented as explicit fields in `RolloutStorage`,
  copied by `add_transitions`, yielded by feedforward and recurrent FrontRES
  mini-batches, and parsed compatibly by `FrontRESUnified`.
- [x] Module 8, algorithm loss:
  `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`.
  Add a dedicated authority actor-critic loss: critic regression to K-step
  executable return and actor update over authority actions.  This path must not
  use sampled-rho PPO advantage, underwrite bonus, boundary prior pull, or legacy
  acceptance BCE.  Implemented as an explicit optional loss path that is mutually
  exclusive with old structured-rho loss and consumes `proposal_delta_se`,
  `authority_action`, `authority_return_k`, and `authority_mask`.
- [x] Module 9, perturbation scheduler:
  `source/rsl_rl/rsl_rl/frontres/frontres_dr_curriculum.py`,
  `source/rsl_rl/rsl_rl/frontres/training_schedule.py`, and
  `source/rsl_rl/rsl_rl/runners/frontres_training_setup.py`.
  Add temporal perturbation modes: single-frame, burst, persistent.  Print mode,
  burst duration, and clean/recovery tail duration in runner diagnostics.  Pure
  authority event helper and toy test are complete.  Live burst integration is
  implemented in `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py`
  and wired through `source/rsl_rl/rsl_rl/frontres/perturbation_runtime.py`.
  The live event source covers both new IID temporal burst events and the
  existing local-root artifact burst state, so authority queries are not limited
  to one perturbation family;
  the active G1 config sets `frontres_perturbation_temporal_mode="burst"` and
  `frontres_authority_return_horizon=8`.  Persistent-mode live validation is
  left as an ablation, not the active path.
- [x] Module 10, runner integration:
  `source/rsl_rl/rsl_rl/runners/on_policy_runner.py` and runner FrontRES helper
  modules.  Verify the live path writes authority action/log-prob/proposal and
  K-step target into storage, applies sampled authority to execution, and does
  not let old rho-advantage branches overwrite authority fields.  Implemented
  for the K=1 live-route case in
  `source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py` and
  `source/rsl_rl/rsl_rl/runners/frontres_post_step_connector.py`: Stage-1
  proposal stays in action columns `[:6]`, Stage-2 authority rho is written to
  columns `[6:12]`, transition authority fields are populated before storage
  write, and `r_delta` is stored as a placeholder.  After rollout collection,
  `finalize_frontres_authority_k_step_returns(...)` rewrites event-start frames
  with K-step authority returns before `compute_returns()`.  Generic PPO
  surrogate is disabled when authority actor-critic is active so old PPO does
  not train the authority rho head.  Persistent-mode validation remains an
  ablation.
- [x] Module 11, diagnostics:
  `source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py` and
  `source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py`.
  Add live sentinels: authority level fractions, return by level, proposal
  magnitude by level, authority critic loss, actor loss, critic prediction by
  level, K-step horizon, perturbation temporal mode, and burst duration.
  Remove or clearly mark old rho-advantage diagnostics when they are
  ablation-only.  Step 9 implemented detached authority diagnostics in
  `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py` and a dedicated
  authority actor-critic console block in
  `source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py`.  The log now prints
  authority actor/critic loss, return/Q, rho distribution, per-dimension rho,
  low/mid/high rho buckets, temporal mode, K horizon, event count,
  active/query fraction, mean burst duration, and generic PPO disablement.
  `source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py` also labels the
  live objective as `HSL ΔSE proposal + authority actor-critic` when the
  authority sentinel is active.
  Post-Step-9 audit hides legacy route-rho diagnostics entirely in authority
  mode and sets `frontres_cuda_memory_debug=False` in the active G1 config so
  CUDA memory probes are opt-in rather than console spam.
  Burst duration is now live because the runner path stores event metadata and
  rewrites K-step authority returns after rollout collection.
- [x] Test 1, architecture input test:
  `source/rsl_rl/rsl_rl/tests/frontres_split_acceptance_architecture.py`
  proves that split acceptance sees `full_obs + detached bounded Delta SE`
  proposal features and that acceptance-only loss trains Stage 2 without
  backpropagating into Stage 1.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_split_acceptance_architecture.py`.
- [x] Test 1A, authority network surface test:
  `source/rsl_rl/rsl_rl/tests/frontres_authority_network.py` proves that the
  Stage-2 authority actor sees `full_obs + detached bounded Delta SE`, outputs
  bounded 6D rho, applies active dim masks, and that `Q(state, proposal, rho)`
  receives gradients in Stage 2 without leaking gradients into Stage 1.  It
  also proves `authority_actor` and `authority_critic` survive FrontRES
  checkpoint save/load.
  Post-Step-9 audit added an inference-contract assertion: deterministic
  task-space correction must use the authority actor rho, not the legacy
  residual coefficient head.
  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_network.py`.
- [x] Test 2, continuous authority parameterization test:
  create/update a toy test proving raw Stage-2 authority output maps to bounded
  6D continuous rho, active-task-dim masks zero forbidden dimensions, gradients
  flow through the bounded rho mapping, and continuous rho diagnostics report
  mean/std/min/max plus near-zero/near-one fractions.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_space.py`.
  The previous grouped discrete authority test is replaced and retained only in
  history as obsolete prototype evidence.
- [x] Test 3, K-step return construction test:
  create a pure test for executable reward sequences, done masks, horizon K, and
  authority returns.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_return.py`;
  the test covers event masks, done truncation, detached bootstrap, and blocked
  bootstrap after done.
- [x] Test 4, storage -> authority critic loss test:
  create/update a formal storage minibatch test proving authority fields flow
  into `FrontRESUnified`, critic loss fits K-step return, and actor update moves
  probability toward higher-return authority levels.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py`;
  the test runs the formal `FrontRESUnified.update()` path and checks actor rho
  increases, critic MSE decreases, and Stage-1 proposal weights remain unchanged.
- [x] Test 4A, authority storage contract test:
  `source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py` proves that
  `proposal_delta_se`, `authority_action`, `authority_log_prob`,
  `authority_rho`, `authority_return_k`, and `authority_mask` survive
  transition copy, feedforward mini-batch yield, recurrent mini-batch yield, and
  default inactive zero initialization.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py`.
- [x] Test 5, perturbation scheduler test:
  create a toy test proving single-frame, burst, and persistent temporal modes
  produce the intended perturbation masks and diagnostics.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_event.py`;
  the test covers single, burst split, persistent segment hold, persistent
  refresh, and inactive frames.
- [x] Test 5A, runner authority integration test:
  `source/rsl_rl/rsl_rl/tests/frontres_authority_runner_integration.py` proves
  the live runner helper keeps Stage-1 proposal in `action[:6]`, replaces only
  `action[6:12]` with Stage-2 authority rho, writes proposal/rho/mask into the
  transition, reuses one authority query throughout a burst event, and rewrites
  event-start frames with K-step authority returns.  Verified
  with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_runner_integration.py`.
- [x] Test 5C, burst perturbation test:
  `source/rsl_rl/rsl_rl/tests/frontres_burst_perturbation.py` proves IID burst
  perturbations hold one sampled XY/Z/RP/Yaw event value for the configured
  duration and reset cleanly on env reset.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_burst_perturbation.py`.
- [x] Test 5B, authority diagnostics formatter test:
  `source/rsl_rl/rsl_rl/tests/frontres_authority_diagnostics.py` proves the
  Optimization / Update block prints authority actor-critic sentinels, hides old
  structured-rho diagnostics when inactive, and does not depend on runner-local
  variables such as `locs`.  Verified with
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_diagnostics.py`.
- [ ] Test 6, live-path sentinel:
  run a short resume only after Tests 1-3 pass.  The log must prove the active
  path is authority actor-critic, not old structured-rho advantage.
- [x] Live-branch retirement audit:
  search old branch names after implementation:
  `structured_joint`, `rho_advantage`, `underwrite`, `repair_bce`,
  `frontres_acceptance_preference`, `state_alpha`, `Stable Frame`, and verify
  each is active-new, ablation-only, or dead/hard-gated.  No zero-weight live
  graph is acceptable.  Step 8 hard-gated the old structured-rho runner payload
  behind `_structured_joint_rl_enabled()` and `not _authority_actor_critic_enabled()`,
  changed active G1 config to `frontres_structured_joint_rl_enabled=False` and
  weight `0.0`, and strengthened the authority algorithm test so generic PPO is
  reported as disabled even when scheduled `ppo_actor_weight=1.0`.  Step 9 still
  owns cleanup/renaming of old visible diagnostics.

## Previous Change Record: 2026-06-22 Update Memory Safety

- [x] Failure observed:
  CUDA OOM during `FrontRESUnified._update_ppo_supervised()` around iteration 200,
  while evaluating the critic inside `self.alg.update()`.
- [x] Cause narrowed:
  the active run uses a large 288k-sample FrontRES rollout.  The original live
  path compounded this with retained actor graphs; after that was fixed, the run
  still OOMed at iteration 207 during `value_term.backward()`.  This proves the
  previous retained-graph hypothesis was incomplete, but it does not yet prove
  whether the remaining cause is mini-batch peak memory, cross-iteration tensor
  retention, allocator fragmentation, or external GPU occupancy.
- [x] Algorithm memory fix:
  `FrontRESUnified` now calls `optimizer.zero_grad(set_to_none=True)` before
  backward and after skipped NaN-gradient updates.
- [x] Active experiment memory fix:
  `G1FlatFrontRESUnifiedRunnerCfg.algorithm.num_mini_batches` is increased from
  4 to 16, reducing per-minibatch activation memory without changing rollout
  sample count or the rho objective.
- [x] Region-direct graph fix:
  `FrontRESUnified` no longer builds PPO action-log-prob or per-dim rho-log-prob
  graphs when `frontres_structured_joint_rl_loss_mode="region_direct"`, because
  the active rho loss is a direct logit loss rather than a sample-log-ratio loss.
- [x] Retired legacy-branch graph cleanup:
  disabled legacy acceptance preference loss now returns a detached zero scalar
  instead of `mu_batch.sum() * 0`, so `frontres_acceptance_preference_weight=0`
  and `frontres_structured_joint_rl_keep_legacy_bce=False` do not leave a tiny
  obsolete actor graph on the live path.
- [x] Active update peak-memory fix:
  in active `region_direct` mode, `FrontRESUnified` now uses a memory-safe update
  route: critic value loss is backpropagated first, proposal supervised loss is
  backpropagated next, then the actor is recomputed for rho acceptance loss.
  This removes the previous live-path `retain_graph=True` peak while preserving
  the rho-only acceptance gradient boundary.
- [x] Update-path toy test:
  `source/rsl_rl/rsl_rl/tests/frontres_region_direct_update_path.py` runs one
  CPU `FrontRESUnified.update()` through the active region-direct branch and
  verifies `generic=0`, `repair_bce=1`, storage clear, and returned diagnostics.
- [x] Update memory pipeline test:
  `source/rsl_rl/rsl_rl/tests/frontres_update_memory_pipeline.py` repeatedly
  fills synthetic FrontRES storage and calls the formal `FrontRESUnified.update()`
  path.  It is used to separate algorithm-update memory retention from
  runner/environment/live-storage memory growth.  It supports both `--policy tiny`
  for isolated update logic and `--policy frontres` for the real
  `FrontRESActorCritic` network with synthetic storage.
- [x] CUDA cache fragmentation mitigation:
  `FrontRESUnified._update_ppo_supervised()` now releases unused CUDA cache at
  update entry and after storage clear, matching the OOM hint that reserved but
  unallocated memory was large.
- [x] Evidence-first OOM diagnostic:
  `frontres_cuda_memory_debug=True` prints compact CUDA allocated/reserved/peak/free
  sentinels at update entry, after the first mini-batch value/supervised/rho
  backward stages, and at the exact OOM stage if an OOM occurs.  This is
  diagnostic-only and must be inspected before changing batch count, allocator
  settings, or loss structure again.
- [x] Live runner pipeline section sentinel:
  `on_policy_runner.py` prints compact `[FrontRES pipeline]` section lines when
  `frontres_cuda_memory_debug=True`, covering rollout start, selected env steps,
  reward compute, storage write, compute returns, algorithm update, and logging.
  This separates runner/env/live-storage failures from algorithm-update failures.
- [x] Runner log-loc CUDA retention fix:
  `on_policy_runner.py` no longer uses `locals().copy()` for FrontRES boundary
  stats or logging.  Boundary control now consumes only materialized diagnostic
  means, and logging/checkpoint probes receive sanitized values that detach small
  tensors and drop large live CUDA tensors.  This prevents log dictionaries from
  retaining rollout tensors or recursively chaining previous iteration locals.
- [x] Global retention-pattern sweep:
  searched `source` for `locals().copy()`, `locs=locals()`, `retain_graph=True`,
  `self.*.append(...)`, and suspicious `self.* = tensor` patterns.  The remaining
  `locs=locals()` in the reward connector call was replaced with an explicit
  scalar-only dictionary, and `FrontRESUnified.update()` now clears policy
  observation caches after storage clear.
- [x] Live-batch rho replay test added:
  `source/rsl_rl/rsl_rl/tests/frontres_live_batch_replay.py` reads a real
  minibatch dumped from `FrontRESUnified.update()` and trains a small
  state-conditioned rho head offline.  Use it to decide whether true rollout
  data can support `rho+ > rho-` before spending another long run on method
  changes.
- [ ] First short-run sentinel observed:
  the next run should print `mini_batches=16`, include `[FrontRES CUDA mem]`
  compact sentinel lines, and keep `[FrontRES pipeline]` `iteration_start` /
  `log_after` allocated memory roughly flat instead of rising by ~0.1 GiB per
  iteration.  If it still OOMs, report the OOM stage with memory numbers.
- [ ] First short-run behavior checked:
  after iteration 200, inspect active-dim `rho by adv +/−` separation over a
  small window.

## Previous Change Record: 2026-06-21 Logit-Level Rho Repair Loss

- [x] Design note updated:
  `note/FrontRES Design Contract.md`, section
  `2026-06-21 Logit-Level Rho Repair Loss`.
- [x] Test reference updated:
  `source/rsl_rl/rsl_rl/tests/frontres_rho_low_recovery_mechanism.py`;
  `source/rsl_rl/rsl_rl/tests/frontres_storage_algorithm_loss.py`.
- [x] Core concept: keep the existing rho advantage evidence, but apply
  repairable-region rho learning to the rho logit with BCEWithLogits.
- [x] Forbidden change: do not add a new evidence source, do not merge boundary
  prior into `rho_adv`, and do not remove the old post-sigmoid loss ablation.
- [x] Config surface verified for
  `frontres_structured_joint_repair_loss_kind` and
  `frontres_structured_joint_repair_loss_scale`.
- [x] Algorithm path verified in `FrontRESUnified` region-direct loss.
- [x] Diagnostics verified to print `repair_bce` and `rscale` in the same live
  branch as `rho region loss`.
- [ ] First short-run sentinel observed: startup prints `repair=bce_logit` and
  log prints `repair_bce=1, rscale=1.000`.
- [ ] First short-run behavior checked: rho should recover from near-zero more
  quickly without making harmful Projected samples positive.

## 0. Design Delta

- [ ] Write the intended concept in one sentence.
- [ ] Name the variable being changed, such as rho authority, reward evidence,
  perturbation schedule, action cone, storage payload, or diagnostics.
- [ ] State what must not change.
- [ ] State whether this is a formal change, an ablation, or a diagnostic-only
  change.
- [ ] If the change affects a method idea, update
  `note/FrontRES Design Contract.md` before code is edited.

Evidence to record:

- Design section or note line:
- Old behavior:
- New behavior:

## 1. Config Surface

Check every config layer, not only the one being edited.

- [ ] Runner config class fields are updated.
- [ ] Algorithm config dataclass fields are updated.
- [ ] Algorithm constructor accepts the new field.
- [ ] Algorithm stores or prints the new field when useful.
- [ ] Active experiment config sets the intended value.
- [ ] Generic defaults are conservative and do not silently alter unrelated
  experiments.
- [ ] Legacy aliases or duplicated config names are either updated or explicitly
  left unchanged.

Required search:

```bash
rg -n "NEW_CONFIG_NAME|OLD_RELATED_CONFIG_NAME" source note
```

Evidence to record:

- Generic default:
- Active experiment value:
- Constructor field:
- Startup/log sentinel:

Common failure this prevents:

- Editing only `algorithm = RslRlFrontRESUnifiedAlgorithmCfg(...)` while the
  runner-level cfg still supplies the old value.
- Adding a config field to cfg dataclasses but not to `FrontRESUnified.__init__`,
  causing an unexpected keyword error.

## 2. Live Route

Prove the value reaches the live training path.

- [ ] Runner reads the intended config key.
- [ ] The helper/function receives the value as an argument.
- [ ] The helper/function uses the value in the intended formula or branch.
- [ ] Older fallback branches cannot overwrite the result.
- [ ] Cold-start and disabled-branch initialization set safe defaults.

Required search:

```bash
rg -n "NEW_SYMBOL|HELPER_NAME|OLD_VISIBLE_BEHAVIOR" source/rsl_rl/rsl_rl
```

Evidence to record:

- Config read location:
- Helper call location:
- Formula location:
- Disabled/default branch location:

## 3. Formula And Test Consistency

When a test module mirrors formal logic, explicitly compare both sides.

- [ ] Test formula and formal formula are intentionally identical, or the test
  clearly labels baseline versus proposed formula.
- [ ] Test inputs use the same default margins and relevant config constants as
  the formal code, unless the test is deliberately sweeping them.
- [ ] The test covers positive, negative, boundary, and low-signal cases.
- [ ] A known-bad case remains bad after the proposed change.
- [ ] A known-good low-signal case receives stronger evidence after the change.

Evidence to record:

- Test file:
- Formal file:
- Positive case:
- Negative case:
- Low-signal case:
- Sweep result:

## 4. Storage And Algorithm Contract

If the change affects training, verify the storage-to-loss path.

- [ ] Runner writes the changed value to the correct transition/storage field.
- [ ] Storage preserves shape, dtype, and batch order.
- [ ] Minibatch unpacking reads the correct field.
- [ ] Algorithm loss consumes the intended field.
- [ ] Gradient applies only to the intended policy dimensions.
- [ ] Old losses are disabled or kept only as deliberate ablations.

Required check:

```bash
rg -n "storage_field|transition_field|loss_name|gradient_boundary" source/rsl_rl/rsl_rl
```

Evidence to record:

- Transition write:
- Storage field:
- Minibatch read:
- Loss use:
- Gradient boundary:

## 5. Diagnostics And Log Sentinel

A mechanism is not live unless the log can prove it.

- [ ] Runner local values are initialized in cold-start/default paths.
- [ ] Runner local values are updated on the live path.
- [ ] Diagnostic sums include the new keys.
- [ ] Mean materialization exposes the keys with the expected names.
- [ ] Formatting function has access to the variables it prints.
- [ ] The log contains a one-line sentinel with the new value or mode.
- [ ] The sentinel appears in the same live branch as nearby old diagnostics.

Required search:

```bash
rg -n "NEW_DIAG_KEY|OLD_VISIBLE_LOG_LABEL|format_frontres" source/rsl_rl/rsl_rl/frontres
```

Evidence to record:

- Initialized key:
- Updated key:
- Accumulated key:
- Materialized mean:
- Printed line:
- First short-run log:

Common failure this prevents:

- Printing `locs` inside a formatter that only receives `loss_dict`.
- Adding a diagnostic to one log branch while the live branch prints a duplicate
  old label.

## 5A. OOM Debug Pipeline

Use this when CUDA OOM appears during FrontRES training.  The goal is to classify
the failure before changing any memory-related knob.

- [ ] Stage 0, route sentinel:
  confirm startup prints the intended `epochs`, `mini_batches`, objective, and
  structured rho mode.
- [ ] Stage 1, update-entry baseline:
  compare `[FrontRES CUDA mem] label=update_entry` across iterations.  If
  allocated/reserved memory rises monotonically at update entry, suspect
  cross-iteration tensor retention.
- [ ] Stage 2, first-mini-batch peak:
  inspect only the first mini-batch sentinel lines:
  `value_backward_after`, `actor_supervised_backward_after`, and
  `rho_backward_after`.  These show which update stage creates the largest peak
  without flooding the terminal.
- [ ] Stage 3, exact failure location:
  if OOM happens, use `oom_value_backward`, `oom_actor_supervised_backward`, or
  `oom_rho_backward` as the decisive stage label.
- [ ] Stage 4, branch audit:
  if update-entry memory grows, search for active branches that retain tensors,
  graphs, cached observations, diagnostics, or storage references across
  iterations.  Do not change `num_mini_batches` first.
- [ ] Stage 5, runner pipeline section:
  inspect `[FrontRES pipeline]` lines.  The last printed section before failure
  identifies whether the failure is in rollout preparation, `env.step`, reward
  compute, storage write, `compute_returns`, `alg.update`, or logging.
- [ ] Stage 6, intervention:
  only after stages 1-5 identify the cause, choose one intervention:
  smaller per-mini-batch update, micro-batched critic/value backward, branch
  retirement, cache cleanup, allocator setting, or external GPU isolation.

Decision rules:

- Stable update-entry memory plus OOM at one backward stage:
  likely mini-batch peak memory in that stage.
- Rising update-entry memory:
  likely cross-iteration retention or an active branch keeping CUDA tensors.
- Large reserved/free mismatch with modest allocated memory:
  likely allocator fragmentation or external process pressure.
- Missing `[FrontRES CUDA mem]` lines:
  config route or algorithm constructor mismatch, not a memory conclusion.
- Missing `[FrontRES pipeline]` lines while CUDA mem debug is on:
  runner debug route mismatch, not a memory conclusion.

Evidence to record:

- Startup sentinel:
- Update-entry trend:
- Peak stage:
- OOM stage:
- Decision:
- Intervention:

## 6. Static Verification

Run syntax and route checks before training.

- [ ] `python -m py_compile` covers every modified Python file.
- [ ] `rg` confirms the new key appears in all required layers.
- [ ] `git diff --stat` is reviewed for unrelated files.
- [ ] `git diff` is reviewed around each edited block.
- [ ] No unrelated user changes are reverted.

Suggested command pattern:

```bash
python -m py_compile FILES_CHANGED
rg -n "NEW_SYMBOL|NEW_CONFIG|NEW_LOG_LABEL" source note
git diff --stat
```

Evidence to record:

- Compile result:
- Search result:
- Diff reviewed:

## 7. First Short-Run Acceptance

The first short run proves the path, not final performance.

- [ ] Startup summary prints the intended config value or mode.
- [ ] The new log sentinel appears in the first relevant iteration.
- [ ] The old visible behavior changes only where expected.
- [ ] The key diagnostic moves in the expected direction.
- [ ] No exception occurs in rollout, logging, storage, or algorithm update.
- [ ] Performance is judged over a small window, not from one lucky iteration.

Evidence to record:

- Iteration:
- Startup sentinel:
- Live diagnostic line:
- Expected diagnostic movement:
- Unexpected behavior:

## 8. After-Run Decision

Classify the outcome before making another edit.

- [ ] Implementation worked and concept improved.
- [ ] Implementation worked but concept failed.
- [ ] Implementation did not reach the live path.
- [ ] Diagnostic was insufficient to judge.
- [ ] New failure belongs to current method core.
- [ ] New failure is a boundary diagnostic or future research problem.

Decision:

- Keep:
- Revert:
- Tune:
- Add diagnostic:
- Stop and rethink concept:
