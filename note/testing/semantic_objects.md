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
advantages
returns
```

Owner and lifecycle:

```text
policy distribution: MAIN-19 front_residual_actor_critic.py
action mask / env bridge: MAIN-22 task_space_correction.py, MAIN-33 frontres_rollout_step.py
algorithm log-prob consumer: MAIN-47 frontres_unified.py
PPO loss/update: MAIN-46 frontres_segment_ppo.py
sequence/live diagnostics: MAIN-38, MAIN-40
checkpoint source: MAIN-32, MAIN-48
```

Required evidence:

```text
S1 diagnostics contract for raw mean saturation health
S1/S2 algorithm contract for positive-advantage gradient direction toward stored 6D actions
S2 sequence eval debug contract that prints `action_distribution_health`
S3 checkpoint/load forward-health contract before long training
S4 sequence eval/live sentinel when actual checkpoint quality is questioned
```

Current gap:

```text
S3 checkpoint load -> one policy forward health gate is not yet enforced.
```

## Direct Delta SE PPO Semantic Closure

Aliases:

```text
direct Delta SE HRL
executed Delta SE
sampled repair action
old_mu
old_sigma
old_means
old_sigmas
desired_kl
adaptive LR
learning_rate
initial_lr
frontres_segment_ppo_lr
pre-step LR
post-update KL
PPO trust region
raw_log_ratio
clamped ratio
pre/post ratio diagnostics
raw_action
raw action tail
raw_action_old_mean
per-dim sigma
per-dim mean delta
per-dim log-ratio contribution
tanh Jacobian contribution
frontres_mask
paired repaired-vs-noisy gain
actor_update_mask
```

Owner and lifecycle:

```text
policy distribution: MAIN-19 front_residual_actor_critic.py
rollout action capture: MAIN-33 frontres_rollout_step.py, MAIN-38 frontres_segment_live_training.py
segment storage: MAIN-44 frontres_segment_storage.py
Segment PPO loss/update: MAIN-46 frontres_segment_ppo.py
live update loop: MAIN-39 frontres_segment_live_update_loop.py
legacy/direct PPO reference path: MAIN-45 ppo.py, MAIN-47 frontres_unified.py
sequence/eval diagnostics: MAIN-40 frontres_segment_sequence_eval.py
```

Required evidence:

```text
S1/S2 action/log_prob transform contract for the same executed 6D Delta SE
S1/S2 storage contract preserving old_log_prob, old_means, old_sigmas, returns, advantages, and valid masks
S1/S2 advantage scaling contract preserving no-regret sign under default `scale_only` mode
S1/S2 PPO contract that uses old distribution stats for KL/trust-region diagnostics or explicitly proves the intended alternative
S1/S2 exact clipped-surrogate, old-policy detach, row-permutation invariance, invalid-row isolation, and full-6D support under single-family action-mask metadata contracts
S1/S2 ratio diagnostic consistency contract separating pre-loss raw/clamped ratio from post-update raw/clamped ratio
S2 paired repaired-vs-noisy reward/advantage route contract
S4 live sentinel when real rollout/update is changed
```

Current status:

```text
Segment storage preserves old_means/old_sigmas through FrontRESSegmentPPOBatch. frontres_segment_ppo.py reports both logprob_approx_kl and MOSAIC-style distribution_kl_mean, and uses distribution_kl_mean as approx_kl when old/new distribution stats are available. frontres_segment_algorithm_contract.py now confirms exact distribution KL, exact clipped surrogate behavior, old-policy tensor detach, invalid-row isolation, row-permutation invariance, full-6D PPO support under rp-only action-mask metadata, advantage-dominance diagnostics, small-sigma KL sensitivity, ratio-source decomposition, and scale-only advantage normalization preserving all-positive no-regret signs. run_frontres_segment_single_update defaults Segment PPO advantage normalization to `scale_only`, allows schedule=adaptive pre-step LR reduction when old/new distribution KL is already high, blocks low-pre-KL LR amplification before optimizer.step, recomputes post-update trust-region KL after optimizer.step, reports the post value as the live `ppo.kl`, reports post-update mean_delta from stored old_means, and rolls back adaptive post-KL violations before retrying with a reduced LR. bounded Delta SE log-prob reconstruction is covered by the live single-update contract using raw policy stats plus tanh Jacobian correction, and live probe text now prints ratio-source blocks for raw action tail, per-dim sigma, per-dim mean delta, per-dim log-ratio contribution, and tanh-Jacobian contribution. Step B on 2026-07-09 contract-confirms explicit separation of pre-loss raw-log-ratio, pre-loss clamped-ratio, post-update raw-log-ratio, and post-update clamped-ratio diagnostics, and live text now prints `pre_log_ratio`, `pre_ratio`, `post_log_ratio`, and `post_ratio` without ambiguous `ratio.reported_mean`. The Stage 3 entrypoint and launch contracts now cover explicit `--frontres_segment_ppo_lr` pass-through. This is S1/S2 contract-confirmed; real training quality remains S4/live-log evidence.
```

Current gap:

```text
`action_mask` is preserved through Segment storage and validated by Segment PPO, but direct Delta SE PPO intentionally keeps full-6D repair support. Current contracts prove an rp-only mask does not change loss/gradient, all 6D actor rows can receive gradient, and KL remains full-6D. Do not reinterpret perturbation family as a PPO dimension mask.
```
