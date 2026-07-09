# FEMR Impact Rules

Use this file before choosing local tests. The goal is to expand from changed
files to likely affected modules so cross-cutting bugs do not hide.

## Semantic Object Expansion

### `normalizer.py`, `mean`, `std`, obs stats

Potential impact:

```text
MAIN-10 observations.py
MAIN-12 frontres_observation_layout.py
MAIN-19 front_residual_actor_critic.py
MAIN-20 normalizer.py
MAIN-32 frontres_checkpointing.py
MAIN-40 frontres_segment_sequence_eval.py
MAIN-48 frontres_segment_checkpointing.py
MAIN-50 exporter.py
M2C-04 observation layout/stats
M2C-19 checkpoint stats
M2C-21 export/play sinks
```

Recommended tests:

```text
S0 py_compile changed files
S1 frontres_observation_layout_contract.py
S3 frontres_segment_checkpoint_contract.py
S3 export/play stats test if present; otherwise mark missing-test
S4 sequence eval sentinel only if live eval behavior changed
```

### `rollout_storage.py` or transition/batch tuple fields

Potential impact:

```text
MAIN-34 post-step connector
MAIN-41 Transition schema
MAIN-42 add_transitions
MAIN-43 mini_batch_generator
MAIN-44 segment storage
MAIN-46 segment PPO
MAIN-47 FrontRES loss
M2C-14 training payload
M2C-15 batch contract
```

Recommended tests:

```text
S1/S2 frontres_segment_storage_contract.py
S1/S2 frontres_segment_algorithm_contract.py
S2 frontres_segment_all_contract_suite.py
```

### Runner helper changes

Potential impact:

```text
MAIN-30 on_policy_runner.py
MAIN-31 setup
MAIN-33 rollout_step
MAIN-34 post_step_connector
MAIN-35 HSL target
MAIN-36 runtime correction
MAIN-37/40 Stage 3 live loop
MAIN-41/43 storage
MAIN-45/47 algorithm
```

Recommended tests:

```text
S0 py_compile changed files
S2 relevant runner contract
S1/S2 frontres_segment_diagnostics_contract.py when live summary, diagnostics,
  replay candidate counts, or Motion Quality fields change
S2 frontres_segment_stage3_pseudo_suite.py
S4 frontres_segment_live_sentinel_contract.py when live path changed
```

### task-space action distribution, `transition_means`, raw logits, action clamp

Potential impact:

```text
MAIN-19 front_residual_actor_critic.py
MAIN-22 task_space_correction.py
MAIN-32 frontres_checkpointing.py
MAIN-33 frontres_rollout_step.py
MAIN-38 frontres_segment_live_training.py
MAIN-40 frontres_segment_sequence_eval.py
MAIN-46 frontres_segment_ppo.py
MAIN-47 frontres_unified.py
MAIN-48 frontres_segment_checkpointing.py
```

Recommended tests:

```text
S0 py_compile changed files
S1/S2 frontres_segment_algorithm_contract.py
S1/S2 frontres_segment_diagnostics_contract.py
S2 frontres_segment_sequence_eval_contract.py
S3 frontres_stage3_noise_std_migration_contract.py
S3 frontres_segment_checkpoint_contract.py when checkpoint payload/load changes
S4 offline/sequence eval log inspection when a real checkpoint has `transition_means` or `log_prob` explosions
```

### direct Delta SE PPO semantic closure

Potential impact:

```text
MAIN-19 front_residual_actor_critic.py
MAIN-33 frontres_rollout_step.py
MAIN-38 frontres_segment_live_training.py
MAIN-39 frontres_segment_live_update_loop.py
MAIN-44 frontres_segment_storage.py
MAIN-45 ppo.py
MAIN-46 frontres_segment_ppo.py
MAIN-47 frontres_unified.py
```

Recommended tests:

```text
S0 py_compile changed files
S1/S2 frontres_segment_live_probe_ppo_contract.py for executed 6D Delta SE old/new log_prob transform
S1/S2 frontres_segment_storage_contract.py for old_means/old_sigmas persistence through storage
S1/S2 frontres_segment_algorithm_contract.py for valid-mask, gradient behavior, exact clipped surrogate, exact old-stat distribution KL, old-policy detach, row permutation invariance, and full-6D support under single-family action-mask metadata
S2 frontres_segment_live_single_update_contract.py for old-stat pre KL -> pre-step high-KL LR reduction without low-KL LR increase -> optimizer.step -> post-update trust-region KL diagnostic/reject route
S2 frontres_segment_live_update_loop_contract.py for optimizer route
```

High-risk PPO T extensions:

```text
T-clip        exact clipped-surrogate behavior for positive and negative advantage ratio cases.
T-kl-exact    exact old/new distribution KL from old_means/old_sigmas/new mean/sigma.
T-detach      old policy, return, advantage, and old distribution tensors do not receive gradient.
T-permute     row permutation does not change loss, KL, ratio summaries, or update decision.
T-update-order pre-loss KL, pre-step high-KL LR reduction without low-KL LR increase, optimizer step, post-update KL, and optional Segment trust-region rollback order are explicit.
T-cone        direct Delta SE full-support semantics under single-family perturbation metadata; action_mask must not silently shrink PPO log_prob, KL, loss, or gradients unless a later method explicitly changes the design.
T-adv-dominance valid-sample advantage distribution reports whether one or a few samples dominate the actor update.
T-adv-sign-preserve default Segment PPO advantage scaling preserves the sign of positive/negative no-regret evidence; standard mean-centering is only an explicit ablation.
T-bounded-logprob-source bounded Delta SE actions reconstruct log_prob from the same raw policy distribution stats and Jacobian correction.
T-small-sigma-kl-sensitivity exact KL stress shows how small sigma amplifies the same mean shift.
T-post-mean-delta post-update diagnostics report policy mean movement from stored old_means on the same PPO batch.
T-ratio-diagnostic-consistency pre-loss raw log-ratio, pre-loss clamped ratio, post-update raw log-ratio, and post-update clamped ratio are named and displayed as separate timepoints.
T-lr-scale    Segment PPO initial learning_rate, pre-step LR adjustment, post-update KL rejection, and CLI override path are covered together; low pre-step KL must not amplify LR before the post-update trust-region gate.
```

### Segment cache/dataset/sampler changes

Potential impact:

```text
MAIN-26 cache builder
MAIN-27 dataset
MAIN-28 sampler
MAIN-37 live sampler
MAIN-38 live training
```

Recommended tests:

```text
S1/S2 frontres_segment_all_contract_suite.py
S1 frontres_segment_live_sampler_contract.py when sampler changes
S4 live sentinel only if reset/env interaction changed
```

### Reward/executability changes

Potential impact:

```text
MAIN-11 balance.py
MAIN-16 rewards.py
MAIN-17 frontres_segment_reward.py
MAIN-18 frontres_reward_diagnostics.py
MAIN-24 executable floor
M2C-07 executable floor
```

Recommended tests:

```text
S1 frontres_balance_margin_contract.py
S1/S2 frontres_segment_reward_contract.py
S1 frontres_reward_compute.py if applicable
S4 live sentinel only when env reward lifecycle changed
```

## Reporting Rule

Every local test recommendation must report:

```text
changed file -> atlas block -> expanded blocks -> chosen tests -> uncovered gaps
```
