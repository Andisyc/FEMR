# FEMR Semantic Objects

This file records cross-cutting objects whose aliases and lifecycle make simple
keyword search insufficient.

## Observation Stats / Normalizer

Aliases:

```text
normalizer
obs_norm
running mean/std
mean
std
mean_std
state_dict
checkpoint stats
privileged stats
FrontRES prefix
GMT suffix
combined actor payload
```

Owner and lifecycle:

```text
owner: MAIN-20 source/rsl_rl/rsl_rl/modules/normalizer.py
layout owner: MAIN-12 source/rsl_rl/rsl_rl/modules/frontres_observation_layout.py
policy consumer: MAIN-19 source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py
checkpoint: MAIN-32, MAIN-48
eval/export sinks: MAIN-40, MAIN-50
```

Required evidence:

```text
S1 layout split/compose contract
S3 checkpoint save/load contract
S3 export/play/eval sink check
S4 live eval sentinel when live sequence eval path changes
```

Current gap:

```text
Export/play normalizer sink test is not yet inventoried as covered.
```

## Storage Transition Payload

Aliases:

```text
transition
payload
rollout storage
mini batch
batch tuple
frontres target
mask
alpha/rho fields
diagnostics
```

Owner and lifecycle:

```text
source: MAIN-34 frontres_post_step_connector.py
storage schema: MAIN-41 rollout_storage.py :: Transition
write: MAIN-42 rollout_storage.py :: add_transitions
read: MAIN-43 rollout_storage.py :: mini_batch_generator
consumer: MAIN-46 frontres_segment_ppo.py, MAIN-47 frontres_unified.py
```

Required evidence:

```text
S1/S2 storage contract
S1/S2 algorithm contract
S2 aggregate contract suite
```

## Segment Replay Lifecycle

Aliases:

```text
segment
cache
dataset
sampler
reset request
live sampler
live training
sequence eval
preroll
eval window
live summary replay candidates: `sampler_update_replay_candidate_count`
legacy/compat replay candidates: `sampler_replay_candidates`
motion quality positions: `motion_clean_body_pos`, `motion_noisy_body_pos`, `motion_repaired_body_pos`
```

Owner and lifecycle:

```text
cache: MAIN-26
dataset: MAIN-27
sampler: MAIN-28
reset: MAIN-29
live sampler/training/update/eval: MAIN-37/40
```

Required evidence:

```text
S1/S2 aggregate contract suite
S1/S2 diagnostics contract for replay-candidate field alias and missing-position Motion Quality reporting
S2 stage3 pseudo suite
S4 live sentinel when reset/env/live route changes
```

## Task-Space Action Distribution Health

Aliases:

```text
transition_means
transition_sigmas
action_mean
action_std
raw mean
raw logits
raw_policy_action
segment_transition_actions
delta_se_norm
mean_raw_saturated_frac_abs_gt_2
mean_raw_abs_max
actions_log_prob
```

Owner and lifecycle:

```text
policy distribution: MAIN-19 front_residual_actor_critic.py
action mask / env bridge: MAIN-22 task_space_correction.py, MAIN-33 frontres_rollout_step.py
algorithm log-prob consumer: MAIN-47 frontres_unified.py
sequence/live diagnostics: MAIN-38, MAIN-40
checkpoint source: MAIN-32, MAIN-48
```

Required evidence:

```text
S1 diagnostics contract for raw mean saturation health
S2 sequence eval debug contract that prints `action_distribution_health`
S3 checkpoint/load forward-health contract before long training
S4 sequence eval/live sentinel when actual checkpoint quality is questioned
```

Current gap:

```text
S3 checkpoint load -> one policy forward health gate is not yet enforced.
```
