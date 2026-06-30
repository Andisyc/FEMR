# FEMR Repository Architecture Map

This document is the first-level entry map for the FEMR repository. It is not a history
log and not a paper outline.  Its job is to help a human or LLM agent answer:

```text
Which module owns this concept?
Which files should be changed?
Which files should not be touched?
Which tests should be written before a training run?
```

Read this before editing FrontRES, FEMR, MOSAIC training, perturbation
curriculum, storage, diagnostics, or validation scripts.

## 0. Repository As A System

MOSAIC has four large subsystems:

```text
scripts/
  owns command-line entrypoints, data utilities, validation scripts

source/whole_body_tracking/
  owns IsaacLab task, robot/env config, MDP terms, motion command, perturbations

source/rsl_rl/rsl_rl/
  owns policy modules, algorithms, storage, runners, FrontRES helper logic

note/
  owns design contracts, architecture diagrams, modification checklists
```

The active research path for this repository is currently FrontRES Segment
Replay HRL:

```text
motion sequence
  -> Stage 1 segment cache: Clean state + discrete Noisy perturbation bank
  -> Stage 2 HSL initialization for 6D Delta SE(3) repair
  -> Stage 3 sampler selects segment ids
  -> dynamic reset to cached segment state
  -> HRL 6D Delta SE(3) repair rollout
  -> per-sample rollout evidence
  -> sampler priority update + PPO update
```

The current method contract is:

```text
note/frontres_core/contracts/design_contract.md
```

The current engineering checklist is:

```text
note/frontres_core/checklists/modification_checklist.md
```

## 1. Training Entrypoints

### Owner

Training scripts own process startup only.  They should not own FrontRES method
semantics.

### Files

```text
scripts/rsl_rl/train.py
scripts/rsl_rl/play.py
scripts/rsl_rl/cli_args.py
scripts/rsl_rl/collect_expert_trajectories.py
scripts/rsl_rl/check_frontres_stage1_segment_cache_completion.py
check_stage1_completion.sh
```

### Responsibilities

- launch IsaacLab / Isaac Sim;
- parse CLI and Hydra configs;
- instantiate task env and runner;
- resume/load checkpoints;
- run training or play.
- check Stage 1 Segment Replay cache completion against the original AMASS
  motion index without launching IsaacLab.
- expose root-level convenience wrappers for common Stage 1/2/3 and cache
  completion commands.

### Forbidden Responsibilities

- do not define FrontRES reward, rho semantics, HSL target, authority critic, or
  perturbation curriculum here;
- do not patch config values in the entry script unless the CLI explicitly owns
  that override.

### When To Edit

Edit only when startup, resume, logging backend, CLI flags, or task registration
is wrong.

## 2. IsaacLab Task And Environment Layer

### Owner

`source/whole_body_tracking` owns the simulated task: robot config, command
manager, observations, rewards, terminations, and perturbation source.

### Files

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/observations.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/terminations.py
```

### Responsibilities

- define the tracking task and G1 environment;
- define observation layout seen by GMT / FrontRES;
- maintain raw, perturbed, and FrontRES-corrected reference state;
- apply reference perturbations;
- expose environment reward and termination;
- register task config and agent config.

### FrontRES-Specific Responsibilities

- `rsl_rl_mosaic_cfg.py` owns experiment-level config values;
- `commands.py` and `motion_perturbations.py` own reference perturbation state;
- `observations.py` owns whether Stage 2 has enough state/history information;
- `rewards.py` owns environment reward terms, but not FrontRES-specific
  executable reward aggregation.

### Forbidden Responsibilities

- do not implement RSL-RL loss here;
- do not store rollout minibatch fields here;
- do not hide method-specific labels inside observations without documenting the
  deployment availability of that signal.

### FrontRES Authority Actor-Critic Impact

The new design may require:

- temporal perturbation events: start frame, burst duration, recovery tail;
- event boundary or refresh interval signal;
- observation fields sufficient for `rho = pi(s, Delta SE)` to be conditional.

Those belong here or in the runner-facing perturbation helpers, not in the
algorithm loss.

## 3. Policy And Network Modules

### Owner

`source/rsl_rl/rsl_rl/modules` owns neural network parameterization.

### Files

```text
source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py
source/rsl_rl/rsl_rl/modules/actor_critic.py
source/rsl_rl/rsl_rl/modules/residual_actor_critic.py
source/rsl_rl/rsl_rl/modules/supervise_learning.py
source/rsl_rl/rsl_rl/modules/velocity_estimator.py
```

### FrontRES Class

```text
FrontRESActorCritic
```

### Responsibilities

- load frozen GMT policy and normalizer;
- parameterize Stage 1 proposal network;
- parameterize Stage 2 authority actor;
- provide inference and training forward paths;
- expose action distribution stats used by the algorithm.

### Current Useful Structure

`front_residual_actor_critic.py` already has:

```text
frontres_split_acceptance_head
acceptance_actor(full_obs, detached Delta SE proposal)
```

This matches the new proposal-conditioned authority design.  The architecture
test is:

```text
source/rsl_rl/rsl_rl/tests/frontres_split_acceptance_architecture.py
```

### New Authority Actor-Critic Responsibilities

The policy/module layer should own:

```text
Stage 1 proposal actor:
  d_t = Delta SE_HSL

Stage 2 authority actor:
  rho_group ~ pi_rho(state/history, detached d_t)

Stage 2 authority critic:
  Q_phi(state/history, detached d_t, rho_group)
```

The authority space is fixed by the design contract:

```text
rho_planar in {0, 0.5, 1} -> dx, dy, yaw
rho_z      in {0, 0.5, 1} -> dz
rho_rpy    in {0, 0.5, 1} -> roll, pitch
```

### Forbidden Responsibilities

- Stage 2 authority loss must not backpropagate into Stage 1 proposal unless a
  later design explicitly enables joint fine-tuning;
- do not let the ordinary PPO value critic silently become the authority critic;
- do not mix scalar, grouped, and per-axis rho semantics in one live experiment.

## 4. Algorithm Layer

### Owner

`source/rsl_rl/rsl_rl/algorithms` owns optimization and loss computation.

### Files

```text
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
source/rsl_rl/rsl_rl/algorithms/ppo.py
source/rsl_rl/rsl_rl/algorithms/mosaic.py
source/rsl_rl/rsl_rl/algorithms/distillation.py
```

### FrontRES Class

```text
FrontRESUnified
```

### Responsibilities

- update policy parameters from rollout storage;
- compute supervised HSL loss;
- compute PPO/value losses where still active;
- compute Stage-2 authority actor-critic losses;
- enforce gradient boundaries;
- return diagnostics used by runner logging.

### New Authority Actor-Critic Contract

The new active path should be:

```text
HSL loss:
  trains Stage 1 proposal only

authority critic loss:
  fits Q_phi(state, detached proposal, rho_group) to K-step executable return

authority actor loss:
  updates Stage 2 authority actor toward authority actions with higher predicted
  executable value
```

### Old Paths To Retire Or Hard-Gate

These old paths must not remain active in the new authority actor-critic run:

```text
structured_joint sampled-rho advantage
underwrite reward bonus
boundary prior rho pull
legacy acceptance BCE
state alpha / Stable Frame route
endpoint-only acceptance label
```

They can remain as ablations only if guarded by explicit config flags that are
off for the active experiment.

### Forbidden Responsibilities

- do not generate rollout evidence here;
- do not infer perturbation events here;
- do not keep zero-weight legacy tensor graphs alive.

## 5. Storage Layer

### Owner

`source/rsl_rl/rsl_rl/storage` owns rollout tensors and minibatch tuple shape.

### File

```text
source/rsl_rl/rsl_rl/storage/rollout_storage.py
```

### Responsibilities

- store per-step observations, actions, rewards, dones, values;
- store FrontRES-specific supervised and authority fields;
- flatten rollout into minibatches;
- preserve tuple compatibility with algorithm unpacking.

### New Authority Storage Fields

Preferred explicit fields:

```text
proposal_delta_se
authority_action
authority_log_prob
authority_level
authority_return_k
authority_mask
authority_event_id
authority_event_start
```

If old fields such as `acceptance_target` / `acceptance_mask` are reused
temporarily, the comments and tests must state their live meaning.  Do not let
the same storage names mean "rho advantage" in one branch and "authority return"
in another active branch.

### Forbidden Responsibilities

- storage must not compute rewards, labels, or K-step returns;
- storage must not reorder the minibatch tuple in a way that silently breaks
  older algorithm unpacking.

## 6. Runner Layer

### Owner

`source/rsl_rl/rsl_rl/runners` owns the rollout loop and live integration:
policy action, environment step, reward/evidence collection, storage write,
logging, checkpointing.

### Files

```text
source/rsl_rl/rsl_rl/runners/on_policy_runner.py
source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py
source/rsl_rl/rsl_rl/runners/frontres_post_step_connector.py
source/rsl_rl/rsl_rl/runners/frontres_training_setup.py
source/rsl_rl/rsl_rl/runners/frontres_warmup.py
source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py
source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py
source/rsl_rl/rsl_rl/runners/frontres_runtime.py
source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py
source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py
source/rsl_rl/rsl_rl/runners/frontres_segment_live_update_loop.py
source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py
```

### Responsibilities

- orchestrate training phases;
- call policy and env;
- apply task-space corrections before GMT execution;
- collect reward/evidence;
- build storage transitions;
- run update and logging;
- save and resume checkpoints.

### Segment Replay Runner Responsibilities

The Stage 3 runner helper modules should stay thin:

```text
frontres_segment_live_probe.py
  owns live K-step rollout capture, per-sample reward/done payloads, and
  independent Segment Replay storage writes

frontres_segment_live_sampler.py
  owns live sampler initialization, sampled segment batch construction, and
  conversion from per-sample rollout payloads to sampler evidence

frontres_segment_live_update_loop.py
  owns short repeated live update orchestration and summary aggregation

frontres_segment_live_training.py
  owns the normal Stage 3 training loop, checkpoint cadence, resume sentinel,
  and fail-fast diagnostics
```

The runner should not compute priority formulas, cache schemas, or algorithm
losses.

### New Authority Actor-Critic Responsibilities

Runner must connect:

```text
event starts
  -> Stage 1 proposal
  -> Stage 2 grouped authority action
  -> write Delta SE_exec = rho_group * Delta SE_HSL
  -> collect executable rewards over K steps
  -> write authority fields into storage
```

### Forbidden Responsibilities

- runner should not contain the authority critic loss;
- runner should not keep broad `locals()` dictionaries with CUDA tensors;
- runner should not leave old structured-rho carrier overwriting new authority
  fields.

## 7. FrontRES Domain Helpers

### Owner

`source/rsl_rl/rsl_rl/frontres` owns FrontRES-specific pure or near-pure helper
logic.  This folder should hold method concepts that are not runner orchestration
and not algorithm optimization.

### Files

```text
frontres_action_cone.py
frontres_alpha_rho_bridge.py
frontres_alpha_router.py
frontres_diagnostics.py
frontres_dr_curriculum.py
frontres_executability.py
frontres_executable_floor.py
frontres_metrics.py
frontres_oracle.py
frontres_reward_diagnostics.py
frontres_reward_window.py
frontres_rollout_evidence.py
frontres_structured_rho.py
frontres_transition_payload.py
frontres_segment_cache_builder.py
frontres_segment_cache_indexer.py
frontres_segment_cache_io.py
frontres_segment_cache_schema.py
frontres_segment_dataset.py
frontres_segment_reset.py
frontres_segment_sampler.py
frontres_segment_storage.py
perturbation_runtime.py
runtime_diagnostics.py
task_space_correction.py
temporal_reference_cache.py
training_schedule.py
```

### Responsibilities

- action cone and dimension masks;
- task-space correction application;
- executable score components;
- perturbation curriculum helpers;
- reward/evidence summarization;
- diagnostics formatting;
- temporal reference cache.

### Segment Replay Helper Responsibilities

```text
frontres_segment_cache_*.py
  own Stage 1 disk cache schemas, AMASS segment indexing, Clean/Noisy rollout
  state IO, chunked shard storage, lightweight manifest indexes, and cache
  builder orchestration

frontres_segment_dataset.py
  owns loading cached segments into semantic batches; Stage 1 cache loading is
  lazy by default: it reads manifest indexes at startup, loads noisy chunked
  shards on demand, and keeps a small LRU shard cache for Stage 3 sampling; for
  Stage 3 reference alignment its default `reference_window` payload is
  command-shaped `[joint_pos, joint_vel]`

frontres_segment_reset.py
  owns reset requests/results, real env dynamic reset hooks, and the optional
  command-facing reference-window hook boundary

frontres_segment_sampler.py
  owns PLR-style segment sampling, priority, solved/hopeless flags, and state
  dict persistence

frontres_segment_storage.py
  owns independent Stage 3 PPO tuple storage for 6D Delta SE(3) repair
```

Per-sample rollout evidence belongs at the boundary between
`frontres_segment_live_probe.py` and `frontres_segment_live_sampler.py`: probe
captures detached row-level reward/done/valid facts; sampler converts them into
`FrontRESSegmentRolloutEvidence`.

Segment reference alignment belongs at the boundary between
`frontres_segment_reset.py` and the environment motion command: reset receives
`request.reference_window` and calls a command-owned hook when available.  The
motion command, not the runner, should own any real time-step or reference
loader mutation needed to make GMT consume that window.  The real consumer is
`MultiMotionCommand`: it stores a per-env override window, applies active rows
inside `_gather_future_by_motion("joint_pos"/"joint_vel")`, advances the cursor
in `_update_command()`, and clears overrides in `_resample_command()`.

The local Segment Replay runner loop is split across runner helpers:
`frontres_segment_live_sampler.py` samples segment ids and builds the current
batch, `frontres_segment_live_probe.py` owns reset + K-step rollout + storage
write, and sampler evidence is converted back into PLR priority after the probe
summary is available.  This keeps runner orchestration as a connector instead of
moving sampling, reset, reward, storage, and priority logic into one file.

### New Helpers Needed

For authority actor-critic, add small focused modules rather than expanding
`on_policy_runner.py`:

```text
frontres_authority_space.py
  grouped authority levels and group-to-dim mapping

frontres_authority_return.py
  K-step executable return with done masks and optional detached bootstrap

frontres_authority_event.py
  perturbation-event metadata: event id, event start, burst duration, refresh
  interval
```

### Old Helpers To Treat As Ablation

```text
frontres_structured_rho.py
frontres_alpha_rho_bridge.py
frontres_alpha_router.py
```

These may remain in the repository, but the active authority actor-critic path
must not depend on them unless the design contract explicitly changes.

## 8. Perturbation And Curriculum Layer

### Owner

Perturbation has two owners:

```text
whole_body_tracking/mdp/motion_perturbations.py
  owns environment-side perturbation state

rsl_rl/frontres/frontres_dr_curriculum.py and runners/frontres_training_setup.py
  own training-side curriculum schedule and diagnostics
```

### New Event-Level Contract

The authority actor-critic design uses perturbation events:

```text
single-frame event:
  one corrupted frame, one proposal, one authority action, one K-step return

burst event:
  corrupted segment of length L, one proposal/authority decision for the event

persistent event:
  long corruption with explicit authority refresh interval
```

The scheduler must own:

```text
perturbation start time
burst duration
clean/recovery tail duration
authority query frame
authority refresh interval
temporal mode: single / burst / persistent
```

### Diagnostics Required

```text
temporal mode
burst duration
authority query frame
refresh interval
event count
event-level authority coverage
```

## 9. Diagnostics Layer

### Owner

Diagnostics are split:

```text
source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py
source/rsl_rl/rsl_rl/frontres/frontres_reward_diagnostics.py
source/rsl_rl/rsl_rl/frontres/runtime_diagnostics.py
source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py
```

### Responsibilities

- format live console output;
- expose scalar metrics for TensorBoard / logging;
- prove which live path is running;
- distinguish method failure from logging failure.

### New Authority Diagnostics

Required live sentinels:

```text
authority_space = grouped(planar,z,rpy) levels={0,0.5,1}
rho_level_frac by group and level
return_by_level by group and level
proposal_magnitude_by_level
authority_critic_loss
authority_actor_loss
authority_q_pred_by_level
K_step_horizon
temporal_perturb_mode
burst_duration
event_count
```

### Forbidden Responsibilities

- diagnostics must not build autograd graphs;
- diagnostics must not receive `locals()` or large CUDA tensors;
- diagnostics must not print old rho-advantage labels as if they were the new
  authority actor-critic path.

## 10. Tests

### Owner

Tests under `source/rsl_rl/rsl_rl/tests` are local method tests.  They should
catch concept-code mismatch before a long IsaacLab run.

### Existing Useful Tests

```text
frontres_split_acceptance_architecture.py
frontres_region_direct_update_path.py
frontres_storage_algorithm_loss.py
frontres_update_memory_pipeline.py
frontres_live_batch_replay.py
```

### New Test Ladder

Build tests in this order:

```text
1. architecture gradient-boundary test
2. grouped authority-level sampling and coverage test
3. K-step executable return construction test with done masks
4. storage -> authority critic loss test
5. perturbation scheduler event-mode test
6. live-path sentinel short run
```

Do not use long training runs to discover algebra bugs that can be tested
locally.

## 11. Validation Scripts

### Owner

`scripts/robustness_validation` owns post-training validation and paper figures.
It should not own training labels or FrontRES method semantics.

### Files

```text
scripts/robustness_validation/run_validation.py
scripts/robustness_validation/run_validation_batch.py
scripts/robustness_validation/run_validation_mujoco.py
scripts/robustness_validation/run_validation_mujoco_batch.py
scripts/robustness_validation/results_io.py
scripts/robustness_validation/plot_results.py
scripts/robustness_validation/metrics.py
scripts/robustness_validation/ou_injector.py
scripts/robustness_validation/push_controller.py
```

### Responsibilities

- run post-training robustness validation;
- save per-motion results;
- compute validation metrics;
- generate figures.

### Forbidden Responsibilities

- do not tune training reward here;
- do not alter FrontRES authority semantics here;
- do not compare MOSAIC and RobotBridge perturbation scales without explicit
  conversion.

## 12. Notes And Architecture Documents

### Owner

`note/` owns design continuity.

### Current Intended Roles

```text
00_repository_architecture_map.md
  first-level entry module map and ownership document

frontres_core/contracts/design_contract.md
  current method contract and research boundary

frontres_core/checklists/modification_checklist.md
  active engineering checklist for ongoing refactors

frontres_core/paper/method_outline.md
  paper-facing method material, not implementation truth

frontres_segment_replay/
  Segment Replay contracts, plans, intake notes, and code references

architecture/
  HTML/data visual maps
```

### Rule

If these documents disagree, use this priority:

```text
frontres_core/contracts/design_contract.md for current method concept
frontres_core/checklists/modification_checklist.md for active implementation steps
00_repository_architecture_map.md for module ownership
frontres_core/paper/method_outline.md only for writing
older sections as history, not active truth
```

## 13. Current FrontRES Authority Refactor: Module Ownership

The current refactor should land as follows:

```text
Policy/module:
  front_residual_actor_critic.py
  owns Stage 1 proposal actor, Stage 2 authority actor, Stage 2 authority critic

FrontRES helpers:
  frontres_authority_space.py
  frontres_authority_return.py
  frontres_authority_event.py

Runner:
  on_policy_runner.py
  frontres_rollout_step.py
  frontres_training_setup.py
  owns event orchestration and storage write

Storage:
  rollout_storage.py
  owns authority fields and minibatch tuple

Algorithm:
  frontres_unified.py
  owns authority actor-critic update

Diagnostics:
  frontres_diagnostics.py
  frontres_runner_logging.py
  owns live proof that the new path is active

Config:
  rsl_rl_mosaic_cfg.py
  owns flags and hyperparameters
```

Do not implement the refactor by adding another large block into
`on_policy_runner.py`.  Add small modules with one owner each.

## 14. Common Failure Patterns

### Concept-Code Drift

The note says one thing, but live config or old branches run another path.

Check:

```text
config -> runner -> storage -> algorithm -> runtime write -> diagnostics
```

### Storage Mismatch

Runner writes one field meaning; algorithm reads another meaning.

Check:

```text
rollout_storage.py Transition fields
add_transitions
mini_batch_generator tuple
frontres_unified.py unpacking
```

### Hidden Old Branch

An old branch has zero weight but still builds tensors or changes diagnostics.

Search:

```text
structured_joint
rho_advantage
underwrite
repair_bce
frontres_acceptance_preference
state_alpha
Stable Frame
```

### CUDA Tensor Retention

A debug/log dictionary retains rollout tensors across iterations.

Search:

```text
locals()
locals().copy()
dict(locals())
retain_graph=True
self.*append(...)
self.* = *obs/actions/rewards/loss/rho/advantage*
```

### Diagnostics Lie

A metric prints, but it belongs to an old branch or stale variable.

Rule:

```text
If old visible behavior remains, search every emitter of that label before
blaming resume, checkpoint, or server sync.
```

## 15. How To Use This Document

When starting a new task:

1. Identify the concept being touched.
2. Find the module owner in this document.
3. Check the design contract for current method semantics.
4. Check the modification checklist for active step status.
5. Write or update the smallest local test before long training.

When asking another LLM agent to help, give it this document first, then the
specific design contract section relevant to the task.
