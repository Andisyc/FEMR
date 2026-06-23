# FrontRES Engineering Plan

This document is the engineering plan for turning the current FrontRES design
contract into code.  It is different from the other note files:

```text
FrontRES Design Contract.md
  owns the research concept and method boundary

MOSAIC Repository Brief.md
  owns the repository/module map

FrontRES Engineering Plan.md
  owns the concrete code modification plan

FrontRES Modification Checklist.md
  owns execution status and verification checkboxes
```

When Dr. Cheng says "write the code modification plan into ./note", write it
here unless the request is explicitly about research concept or checklist
status.

## 1. Active Engineering Goal

Implement the FrontRES Authority Actor-Critic design:

```text
corrupted reference event
  -> Stage 1 Clean-oriented Delta SE proposal
  -> Stage 2 continuous 6D authority rho
  -> Delta SE_exec = active_task_dims * rho * Delta SE_HSL
  -> K-step executable return
  -> Stage 2 authority actor-critic update
  -> frozen GMT execution
```

The active method is event-level, not per-frame rho.  A perturbation event has
one Stage-1 proposal, one continuous 6D authority action, and one K-step executable
return.

## 2. Fixed Design Decisions

These are not open engineering choices for the active implementation:

```text
authority space:
  6D continuous authority
  rho = [rho_dx, rho_dy, rho_dz, rho_droll, rho_dpitch, rho_dyaw]
  rho_i in [0, 1]
  Delta SE_exec = active_task_dims * rho * Delta SE_HSL

training style:
  on-policy authority actor-critic from current rollout

credit unit:
  perturbation event, not arbitrary frame

return:
  K-step executable return with done masking and detached bootstrap when valid

stage boundary:
  HSL loss updates Stage 1 proposal
  authority loss updates Stage 2 actor/critic
  authority loss does not backpropagate through Stage 1 proposal by default
```

## 2.5 Stage Launch Entrypoints

The live task entrypoint is:

```text
FrontRES-Unified-Tracking-Flat-G1-v0
```

Do not revive the old deprecated two-stage gym registrations in
`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`.
Stage selection is done through explicit launch scripts plus Hydra overrides.

Important CLI rule: use `--frontres_stage ...` instead of Hydra deep overrides.
The Hydra root config is structured and rejects root-level overrides such as
`algorithm.xxx` and `experiment_name=...`.  Stage presets are applied inside
`scripts/rsl_rl/train.py` after Hydra has loaded the typed `agent_cfg`.

Stage 1 launch script:

```text
run/run_frontres_stage1_hsl.sh
```

Purpose: train only the Stage 1 Clean-oriented Delta SE proposal.  This is a
proposal training run, not an authority/rho run.

Active Stage 1 overrides:

```text
--experiment_name g1_flat_frontres_stage1_hsl
--frontres_stage stage1_hsl
```

Command shape:

```bash
bash run/run_frontres_stage1_hsl.sh MOTION_PATH [NUM_ENVS] [MAX_ITERS]
```

Example:

```bash
bash run/run_frontres_stage1_hsl.sh /path/to/motions 12000 800
```

Expected Stage 1 checkpoint:

```text
logs/rsl_rl/g1_flat_frontres_stage1_hsl/<run_name>/model_*.pt
```

Stage 2 launch script:

```text
run/run_frontres_stage2_authority.sh
```

Purpose: load the Stage 1 proposal checkpoint, reset optimizer/iteration state,
and train the Stage 2 continuous 6D authority actor-critic.

Stage 2 freezes Stage 1 at the launch-contract level by setting
`algorithm.lambda_supervised=0.0`.  The authority actor-critic loss already uses
the detached proposal path, so Stage 2 should not keep updating the Stage 1 HSL
proposal during the authority run.  If a later experiment wants joint
fine-tuning, it must be named as a separate ablation.

Stage 2 has its own warmup/takeover schedule.  This reuses the old MOSAIC
training idea, but with Stage-2-specific names:

```text
Stage 2.1: Authority Critic Warmup
  iterations: 0-199 by default
  critic_warmup_iterations=200
  algorithm.frontres_authority_actor_warmup_iterations=200
  meaning:
    data/DR stays conservative;
    authority critic learns K-step executable return;
    authority actor loss is disabled.

Stage 2.2: Authority Actor Takeover
  iterations: 200-399 by default
  algorithm.frontres_authority_actor_ramp_iterations=200
  meaning:
    authority actor loss ramps from 0 to 1;
    critic continues learning;
    actor does not suddenly chase an uncalibrated critic.

Stage 2.3: Full Authority Training
  iterations: 400+ by default
  meaning:
    authority actor and critic train at full weight;
    burst curriculum remains active;
    Stage 1 proposal remains fixed.
```

Important boundary:
`critic_warmup_iterations` controls runner-side DR/curriculum behavior.  It is
not enough by itself.  `frontres_authority_actor_warmup_iterations` controls the
authority actor loss and is required to prevent Stage 2 actor updates from
following an untrained authority critic.

Stage 2 must also set `--supervised_warmup_iterations 0`.  The Stage 1 proposal
has already been trained and loaded from checkpoint.  Re-running the old joint
supervised warmup inside Stage 2 would blur the stage boundary.

Active Stage 2 overrides:

```text
--resume_student_checkpoint STAGE1_CHECKPOINT
--is_full_resume False
--experiment_name g1_flat_frontres_stage2_authority
--frontres_stage stage2_authority
```

Command shape:

```bash
bash run/run_frontres_stage2_authority.sh STAGE1_CHECKPOINT MOTION_PATH [NUM_ENVS] [MAX_ITERS]
```

Example:

```bash
bash run/run_frontres_stage2_authority.sh \
  /path/to/stage1/model_800.pt \
  /path/to/motions \
  12000 \
  2000
```

Checkpoint semantics:

- Stage 1 -> Stage 2 must use `--is_full_resume False`.  This loads the Stage 1
  proposal weights and normalizers, but treats Stage 2 as a new optimization
  run.
- Interrupted Stage 2 resume must use a Stage 2 checkpoint with
  `--is_full_resume True`, because it should restore optimizer, iteration,
  authority actor/critic, and training state.

## 3. Module-Level Plan

### Module A: Authority Space Helper

Create:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_space.py
```

Owns:

- continuous 6D authority bounds `[0, 1]`;
- optional active-task-dim masking;
- conversion from raw network output to bounded rho;
- continuous authority diagnostics such as mean/std/min/max and near-zero /
  near-one fractions.

Must not own:

- neural networks;
- K-step return;
- runner storage write;
- loss computation.

First local test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_space.py
```

Test requirements:

- raw network output maps to bounded rho in `[0, 1]`;
- active-task-dim mask zeros forbidden dimensions;
- gradients can flow through the bounded rho mapping;
- diagnostics summarize continuous rho distribution per dimension.

### Module B: Policy / Network Surface

Modify:

```text
source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py
```

Current useful path:

```text
frontres_split_acceptance_head
acceptance_actor(full_obs, detached Delta SE proposal)
```

Add or expose:

```text
authority_actor:
  outputs continuous 6D authority parameters

authority_critic:
  Q_phi(state/history, detached Delta SE proposal, continuous rho)
```

Implementation preference:

- keep Stage 1 proposal actor as current `residual_actor` or split proposal
  branch;
- keep Stage 2 actor separate from Stage 1 proposal path;
- add a distinct authority critic rather than reusing PPO `critic`;
- expose helper methods that are easy for tests and algorithm code to call:

```text
get_authority_rho(obs, proposal)
evaluate_authority_q(obs, proposal, authority_rho)
bound_authority_rho(raw_authority)
```

Must not do:

- route authority critic through the frozen GMT policy;
- let authority loss backpropagate into Stage 1 proposal;
- silently use old sigmoid rho outputs as the active authority action.

Existing test:

```text
source/rsl_rl/rsl_rl/tests/frontres_split_acceptance_architecture.py
```

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_network.py
```

Test requirements:

- actor produces bounded 6D continuous rho, or distribution parameters whose
  sampled rho is bounded to `[0, 1]`;
- active-task-dim mask zeros forbidden dimensions;
- authority critic receives state/proposal/rho;
- authority loss gives gradients to Stage 2 actor/critic and not Stage 1;
- checkpoint save/load round-trips `authority_actor` and `authority_critic`
  when the optional authority actor-critic surface is enabled.

### Module C: Perturbation Event Helper

Create:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_event.py
```

Owns:

- perturbation event id;
- event start flag;
- event active mask;
- burst duration;
- clean/recovery tail duration;
- authority query frame;
- persistent-mode refresh interval.

Does not own:

- environment-side perturbation magnitudes;
- authority network;
- K-step return loss.

Integrates with:

```text
source/rsl_rl/rsl_rl/frontres/frontres_dr_curriculum.py
source/rsl_rl/rsl_rl/frontres/training_schedule.py
source/rsl_rl/rsl_rl/runners/frontres_training_setup.py
source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py
```

Live implementation note:

- IID jump perturbations can be converted from independent per-frame samples
  into single-frame, burst, or persistent temporal events.
- The existing local-root artifact burst is also exposed as an authority event.
  This keeps the authority query boundary aligned with every short corrupted
  reference event that the active FrontRES curriculum can generate.
- The event helper owns event metadata only.  The environment-side perturber
  still owns perturbation magnitudes and sampled artifact values.

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_event.py
```

Test requirements:

- single-frame mode creates one-frame events;
- burst mode creates length-L events;
- persistent mode refreshes only at configured event boundaries;
- event masks and query frames match the design contract.

### Module D: K-Step Authority Return

Create:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_return.py
```

Owns:

- K-step executable return construction;
- done masking;
- optional detached bootstrap;
- event-level return extraction.

Input:

```text
exec_reward[t, env]
done[t, env]
event_start[t, env]
event_id[t, env]
bootstrap_value[t+K, env] or None
gamma
K
```

Output:

```text
authority_return_k[t, env]
authority_mask[t, env]
```

Does not own:

- how executable reward is computed;
- how authority action is sampled;
- algorithm loss.

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_return.py
```

Test requirements:

- exact K-step sum for a hand-built reward sequence;
- done mask stops future accumulation;
- bootstrap is detached and masked when invalid;
- only event-start frames receive authority targets unless the design changes.

### Module E: Storage Contract

Modify:

```text
source/rsl_rl/rsl_rl/storage/rollout_storage.py
```

Add explicit fields:

```text
proposal_delta_se
authority_action
authority_log_prob
authority_rho
authority_return_k
authority_mask
authority_event_id
authority_event_start
```

Rules:

- append new fields rather than inserting into the middle of existing tuple
  semantics when possible;
- if minibatch tuple expands, update `FrontRESUnified.update()` unpacking in the
  same commit;
- keep old `acceptance_target` / `acceptance_mask` only for old ablation path.

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py
```

Test requirements:

- add_transitions stores authority fields;
- mini_batch_generator returns the exact fields expected by algorithm;
- old acceptance fields cannot overwrite authority fields.

### Module F: Runner Integration

Modify:

```text
source/rsl_rl/rsl_rl/runners/on_policy_runner.py
source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py
source/rsl_rl/rsl_rl/runners/frontres_training_setup.py
source/rsl_rl/rsl_rl/runners/frontres_post_step_connector.py
```

Runner owns the live event path:

```text
event start
  -> get Stage 1 proposal
  -> sample or produce continuous 6D rho
  -> apply active_task_dims mask
  -> execute active_task_dims * rho * proposal
  -> collect executable reward stream
  -> compute/write event-level K-step target
  -> write authority fields into storage
```

Rules:

- do not put authority critic loss in runner;
- do not pass `locals()` to logging or helpers;
- do not let `frontres_structured_rho.py` overwrite authority storage fields;
- print a one-line sentinel when authority actor-critic path is active.

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_runner_toy.py
```

Test requirements:

- synthetic event start triggers one authority query;
- same authority is applied across burst event segment;
- storage gets proposal/rho/event/return fields;
- non-event frames do not create duplicate authority targets.

### Module G: Algorithm Update

Modify:

```text
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
```

Add a dedicated authority actor-critic path.

Critic loss:

```text
L_Q = MSE(Q_phi(s, proposal, rho), authority_return_k)
```

Actor loss:

```text
Use authority critic to prefer higher-Q continuous rho actions.
```

The exact actor objective can be implemented as a continuous policy-gradient or
critic-guided deterministic actor update, but it must obey:

- uses the new continuous 6D authority output, not old sampled-rho advantage;
- uses authority fields from storage, not `acceptance_target/mask`;
- does not use old sampled-rho PPO ratio;
- does not use underwrite bonus;
- does not use boundary prior rho pull;
- does not update Stage 1 proposal from authority loss.

New test:

```text
source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
```

Test requirements:

- critic fits hand-built returns;
- actor moves probability toward higher-return authority levels;
- Stage 1 proposal parameters receive no authority-loss gradient;
- old structured-rho loss is inactive.

### Module H: Diagnostics

Modify:

```text
source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py
source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py
```

Required live diagnostics:

```text
authority_space
authority_rho_mean/std/min/max by dim
authority_rho_near_zero / near_one by dim
authority_return_by_rho_bucket
authority_q_pred_by_rho_bucket
authority_actor_loss
authority_critic_loss
proposal_magnitude_by_rho_bucket
temporal_perturb_mode
burst_duration
event_count
authority_event_mask_mean
```

Rules:

- diagnostics receive scalar or detached CPU values only;
- old rho-advantage metrics must be clearly marked as ablation-only or hidden
  when authority actor-critic is active;
- live log must prove event-level authority is active.

## 4. Implementation Order

This order is not a list of extra method ideas.  The new method has only four
design changes:

```text
Design 1: two-stage FrontRES
Design 2: rho authority actor-critic
Design 3: K-step executable return
Design 4: burst/event perturbation curriculum
```

Some engineering steps below are plumbing steps.  They exist only because a
learning system must connect model output, rollout execution, storage, loss,
config, and diagnostics.  If a step does not directly implement one of the four
designs, it is labeled as plumbing and must not introduce a new concept.

### Step 1: Authority Space

Design role:

```text
supports Design 2
```

Why this step exists:

`rho` can no longer mean an arbitrary scalar, grouped coefficient, old
sampled-rho advantage target, or acceptance label.  Before touching the network
or runner, we need one fixed definition of the action that Stage 2 will choose:

```text
rho = [rho_dx, rho_dy, rho_dz, rho_droll, rho_dpitch, rho_dyaw]
rho_i in [0, 1]
Delta SE_exec = active_task_dims * rho * Delta SE_HSL
```

The earlier grouped discrete helper was useful as a prototype, but it is not the
active mainline.  Grouping rho couples unrelated dimensions and can force the
policy to write directions it did not intend to write.

Modules modified:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_space.py
source/rsl_rl/rsl_rl/frontres/__init__.py
source/rsl_rl/rsl_rl/tests/frontres_authority_space.py
```

Owns:

```text
raw authority output -> bounded continuous rho
active-task-dim mask application
continuous rho diagnostics
gradient-through-bound test
```

Must not own:

```text
network architecture
K-step return
runner storage
algorithm loss
burst scheduler
```

Accept when test passes.

Status:

```text
obsolete prototype completed: 2026-06-23
prototype verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_space.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_authority_space.py source/rsl_rl/rsl_rl/tests/frontres_authority_space.py source/rsl_rl/rsl_rl/frontres/__init__.py
active continuous 6D version: done 2026-06-23
active version verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_space.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_authority_space.py source/rsl_rl/rsl_rl/tests/frontres_authority_space.py source/rsl_rl/rsl_rl/frontres/__init__.py
```

### Step 2: K-Step Return

Design role:

```text
implements Design 3
```

Why this step exists:

The old reward/advantage path judged `rho` with a weak or too-local signal.
Authority should be judged by the short future window that belongs to the
perturbation event.  This step turns executable rewards and done masks into the
training target for the authority critic.

Modules to modify:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_return.py
source/rsl_rl/rsl_rl/tests/frontres_authority_return.py
```

Owns:

```text
exec reward + done + event mask -> authority_return_k
optional detached bootstrap rule
exact horizon/done masking
```

Must not own:

```text
rho sampling
policy network
storage tuple
runner rollout
diagnostic formatting
```

Accept when exact toy returns match by hand.

Status:

```text
done: 2026-06-23
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_return.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_authority_return.py source/rsl_rl/rsl_rl/tests/frontres_authority_return.py source/rsl_rl/rsl_rl/frontres/__init__.py
```

### Step 3: Event Scheduler Helper

Design role:

```text
implements Design 4
```

Why this step exists:

Burst perturbation is not just a different DR scale.  It defines the unit of
credit for Stage 2: one perturbation event gets one authority decision and one
K-step return.  This helper should make event boundaries explicit before the
runner uses them.

Modules to modify:

```text
source/rsl_rl/rsl_rl/frontres/frontres_authority_event.py
source/rsl_rl/rsl_rl/tests/frontres_authority_event.py
```

Owns:

```text
single / burst / persistent event masks and query frames
event_id / event_start / event_active masks
authority refresh rule
```

Must not own:

```text
DR scale magnitude
neural network
K-step numeric return
storage write
algorithm loss
```

Accept when event masks match hand-built timelines.

Status:

```text
done: 2026-06-23
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_event.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_authority_event.py source/rsl_rl/rsl_rl/tests/frontres_authority_event.py source/rsl_rl/rsl_rl/frontres/__init__.py
```

### Step 4: Authority Network

Design role:

```text
implements Design 1 and Design 2
```

Why this step exists:

Stage 1 proposes `Delta SE_HSL`.  Stage 2 must see current state/history plus
the detached proposal and choose continuous 6D authority.  The authority critic must
estimate the value of `(state, proposal, rho)`, not the old PPO state value.

Modules to modify:

```text
source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py
source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py
source/rsl_rl/rsl_rl/tests/frontres_authority_network.py
```

Owns:

```text
proposal-conditioned continuous authority actor and authority critic
Stage-1 / Stage-2 gradient boundary
bounded rho with shape [..., 6]
Q(state, proposal, rho)
checkpoint save/load boundary for authority_actor and authority_critic
```

Must not own:

```text
K-step target construction
rollout storage
runner perturbation schedule
old structured-rho advantage loss
```

Accept when gradients respect Stage-1 / Stage-2 boundary and the optional
authority actor-critic modules survive checkpoint save/load.

Status:

```text
done: 2026-06-23
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_network.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py source/rsl_rl/rsl_rl/tests/frontres_authority_network.py
```

### Step 5: Storage

Design role:

```text
plumbing for Designs 1-4
```

Why this step exists:

The runner and algorithm cannot share authority learning data through comments
or local variables.  Storage must preserve exactly the fields needed by the
authority actor-critic update.

Modules to modify:

```text
source/rsl_rl/rsl_rl/storage/rollout_storage.py
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py
```

Owns:

```text
authority fields survive add_transitions -> mini_batch_generator
proposal_delta_se
authority_action / authority_log_prob / authority_rho
authority_return_k / authority_mask
tuple compatibility with existing FrontRESUnified batch parsing
```

Must not own:

```text
how rho is sampled
how K-step return is computed
how the network is structured
diagnostic interpretation
authority actor-critic loss
```

Accept when tuple contract test passes.

Status:

```text
done: 2026-06-23
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_storage_algorithm_loss.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/storage/rollout_storage.py source/rsl_rl/rsl_rl/algorithms/frontres_unified.py source/rsl_rl/rsl_rl/tests/frontres_authority_storage.py
```

### Step 6: Algorithm Loss

Design role:

```text
implements Design 2 and consumes Design 3
```

Why this step exists:

This replaces the failed sampled-rho advantage path.  The algorithm trains an
authority critic against K-step executable return, then updates the authority
actor toward higher-value authority levels.

Modules to modify:

```text
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
```

Owns:

```text
authority critic regression + authority actor update
no sampled-rho PPO advantage
no underwrite bonus
no boundary prior pull
no legacy acceptance BCE on the active path
```

Must not own:

```text
K-step return construction
event scheduler
network feature construction
runner storage write
```

Accept when toy actor moves toward higher-return authority levels.

Status:

```text
done: 2026-06-23
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_storage_algorithm_loss.py
  frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/algorithms/frontres_unified.py source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
```

### Step 7: Runner Integration

Design role:

```text
connects Designs 1-4 into the live rollout path
```

Why this step exists:

Before this step, helpers and tests may be correct but the training loop still
does not use the new method.  This step wires the real sequence:

```text
event starts
  -> Stage 1 proposal
  -> Stage 2 authority action
  -> Delta SE_exec = rho * detached Delta SE_HSL
  -> env step / executable reward
  -> K-step return target
  -> storage
```

Modules to modify:

```text
source/rsl_rl/rsl_rl/runners/on_policy_runner.py
source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py
source/rsl_rl/rsl_rl/runners/frontres_training_setup.py
source/rsl_rl/rsl_rl/runners/frontres_post_step_connector.py
```

Owns:

```text
live event -> proposal -> authority -> execution -> return -> storage
training/inference consistency for authority application
old-branch overwrite prevention
```

Must not own:

```text
new action-space definition
new K-step formula
new network architecture beyond calling it
new algorithm loss
```

Accept when live-path sentinel prints and no old rho-advantage field overwrites
the authority fields.

Status:

```text
done: 2026-06-23
implemented:
  source/rsl_rl/rsl_rl/runners/frontres_rollout_step.py
    - live rollout keeps Stage-1 proposal in action[:6]
    - Stage-2 authority rho replaces action[6:12]
    - transition stores proposal_delta_se, authority_action/rho, authority_mask
  source/rsl_rl/rsl_rl/runners/frontres_post_step_connector.py
    - writes the current executable reward delta as a placeholder
    - after rollout collection, rewrites storage.authority_return_k and
      storage.authority_mask with event-level K-step returns at event-start
      frames
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py
    - IID perturbations can now run as shared single/burst/persistent temporal
      events instead of independent per-frame jumps
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
    - synchronizes burst event state across Projected/Candidate/Noisy/Clean
      split-env branches
  source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
    - disables generic PPO surrogate when authority actor-critic is active
      so old action-log-prob PPO does not train the authority rho head
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_runner_integration.py
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_burst_perturbation.py
pending:
  persistent temporal scheduler validation in live run
  config defaults and old-branch retirement in Step 8
```

### Step 8: Config And Branch Retirement

Design role:

```text
plumbing and safety guard for the active method
```

Why this step exists:

The new design can silently fail if old structured-rho branches still run,
allocate tensors, write storage, or print active diagnostics.  This step makes
the active experiment select the authority actor-critic route and hard-gates old
paths as ablations.

Modules to modify:

```text
source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
source/rsl_rl/rsl_rl/frontres/frontres_structured_rho.py
```

Owns:

```text
new authority actor-critic path is active
old structured-rho advantage path is ablation-only or hard-gated off
startup summary says which path is active
```

Must not own:

```text
new research objective
new loss term
new perturbation type
new rho semantics
```

Accept after branch-retirement search.

Status:

```text
done: 2026-06-23
implemented:
  source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py
    - policy cfg accepts frontres_authority_actor_critic and hidden dims
  source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py
    - algorithm cfg accepts authority actor/critic flags and weights
  source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py
    - active G1 FrontRES policy builds authority actor/critic
    - active algorithm enables authority actor-critic
    - old structured-rho route is disabled with zero weight
  source/rsl_rl/rsl_rl/runners/on_policy_runner.py
    - old rho advantage / alpha payload writes run only when structured-rho
      ablation is explicitly active and authority actor-critic is not active
  source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
    - startup summary prints authority actor-critic as the active objective
    - generic PPO actor weight reports 0 when authority actor-critic owns rho
verified:
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
  frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_runner_integration.py
  frontres/bin/python -m py_compile on touched config/runner/algorithm files
pending:
  Step 9 diagnostics and short-run sentinel
```

### Step 9: Diagnostics And Short Run

Design role:

```text
verification only
```

Why this step exists:

Diagnostics are not part of the method.  They prove the four designs are
actually live before a long run spends GPU time.

Modules to modify:

```text
source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py
source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py
```

Goal:

```text
one short run proves authority actor-critic path is active
```

Do not judge by episode length first.  First judge:

```text
active objective printed at startup
authority level coverage
authority return by level
critic loss finite
actor loss finite
event count nonzero
old structured-rho diagnostics absent or ablation-marked
generic PPO actor weight shown as disabled in authority mode
K-step horizon / temporal mode visible
```

Implemented diagnostics:

```text
source/rsl_rl/rsl_rl/algorithms/frontres_unified.py
  emits detached scalar authority diagnostics:
  authority_actor_critic_enabled
  authority_actor_loss / authority_critic_loss / authority_total_loss
  authority_active_frac
  authority_return_mean
  authority_q_behavior_mean / authority_q_actor_mean
  authority_rho_mean / std / min / max
  authority_rho_near_zero_frac / near_one_frac
  authority_rho_dx/dy/dz/roll/pitch/yaw_mean
  authority_return_low/mid/high_rho_mean
  authority_q_actor_low/mid/high_rho_mean
  authority_proposal_abs_low/mid/high_rho_mean
  authority_k_horizon

source/rsl_rl/rsl_rl/frontres/frontres_diagnostics.py
  prints a dedicated Authority Actor-Critic block when
  authority_actor_critic_enabled=1.
  Old structured-rho optimization diagnostics remain hidden when their lambda is
  zero.
```

Verification:

```text
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_diagnostics.py
frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_authority_algorithm_loss.py
```

## 5. Branch Retirement Audit

Before any real run, search:

```text
structured_joint
rho_advantage
underwrite
repair_bce
frontres_acceptance_preference
state_alpha
Stable Frame
acceptance_target
acceptance_mask
```

For each occurrence, classify it as:

```text
active-new
ablation-only
dead/hard-gated
bug
```

Do not rely on zero weights if the branch still builds tensors, writes storage,
or prints active-looking diagnostics.

## 6. Engineering Acceptance Criteria

The refactor is not ready for a real run until:

- authority-space test passes;
- K-step return test passes;
- event scheduler test passes;
- authority network gradient-boundary test passes;
- storage tuple test passes;
- algorithm toy loss test passes;
- runner live-path sentinel prints;
- old structured-rho branch is retired or ablation-only;
- diagnostics prove authority actor-critic is active.

## 7. Current Status

Completed:

```text
frontres_split_acceptance_architecture.py
  proves detached proposal input and Stage-1 / Stage-2 gradient boundary for the
  existing split acceptance path.
```

Next planned step:

```text
Step 1: implement frontres_authority_space.py and its test.
```
