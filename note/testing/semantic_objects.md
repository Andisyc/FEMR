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

## Segment Replay State Model

Aliases:

```text
segment_state
unknown
promising
frontier
delayed_regret
solved
hopeless
evidence_count
valid_evidence_count
success_count
fall_count
best_gain
best_short_gain
best_long_gain
last_horizon_k
last_trial_count
last_policy_gain
last_mean_gain
last_success_frac
last_fall_frac
last_oracle_gap
last_confidence
```

Owner and lifecycle:

```text
owner: MAIN-28 frontres_segment_sampler.py
source evidence: MAIN-37 frontres_segment_live_sampler.py
persistence: MAIN-32, MAIN-48 via frontres_segment_sampler_state_dict
diagnostics: MAIN-37/38 sampler stats and replay pool summaries
consumers: multi-trial allocation and horizon curriculum planning
```

Required evidence:

```text
S1 sampler state transition contract for unknown/promising/frontier/delayed_regret/solved/hopeless
S1 state_dict round-trip and legacy solved/hopeless migration contract
S2 live sampler contract proving existing rollout evidence still updates sampler priority
S3 checkpoint contract when new sampler state fields become formal resume gates
S4 live sentinel only when state allocation affects real env rollout horizon or reset behavior
```

Current status:

```text
Step 1 adds explicit sampler-owned segment_state and evidence counters. Step 2 adds fixed-policy multi-trial aggregation for repeated rows of the same segment visit, including trial_count, policy_gain, best_gain, mean_gain, success_frac, fall_frac, oracle_gap, confidence, and last_* persistence fields. Step 3 adds sampler-owned rollout-budget planning and policy-first trial-row expansion. Step 4 adds sampler-owned fixed-row-budget live sampling so expanded policy/search trial rows reach dataset batch construction and live probe without changing PPO loss/update semantics. Step 5 adds the live-probe trial metadata interface so policy/search, source_index, trial_index, and budget_horizon_k reach reset requests, storage priority evidence, summaries, and human-readable trial logs. Step 6 makes PPO eligibility role-gated: policy rows may become PPO-valid rows, while search rows stay priority evidence and are invalid for PPO update. The S1 sampler contract confirms state transitions, legacy migration, multi-trial aggregation, budget allocation, trial-row roles, and fixed env-row-budget sampling. The S2 live sampler, live probe, and live-probe PPO contracts confirm expanded trial rows, role metadata, horizon metadata, sampler update aggregation, reset/storage/summary visibility, and PPO semantic isolation. Real IsaacLab S4 training quality remains unconfirmed.
```

## Segment Multi-Trial Evidence

Aliases:

```text
FrontRESSegmentTrialEvidence
trial_count
valid_trial_count
policy_gain
best_gain
mean_gain
success_frac
fall_frac
oracle_gap
confidence
```

Owner and lifecycle:

```text
owner: MAIN-28 frontres_segment_sampler.py
source evidence: MAIN-37 frontres_segment_live_sampler.py rollout rows
semantic boundary: fixed policy snapshot, no optimizer update between local trials
persistence: sampler state_dict last_* fields for resume/debug continuity
future consumers: replay priority, rollout-budget allocation, horizon curriculum
```

Required evidence:

```text
S1 multi-trial aggregation contract with constructed duplicate segment ids
S1 update contract proving oracle_gap/confidence update sampler state without PPO mutation
S2 live sampler contract proving existing single-rollout evidence still routes correctly
S2 live probe contract proving trial metadata reaches reset/storage/summary without entering PPO batch
S2 live probe PPO-boundary contract proving search rows stay priority evidence and are invalid PPO rows
S3 checkpoint contract proving new last_* state fields do not break resume compatibility
S4 live sentinel only when real multi-trial scheduling changes env rollout behavior
```

Current status:

```text
Step 2 is implemented as an aggregator and state-update boundary. Step 3 adds the sampler-owned budget planner and policy-first trial expansion. Step 4 wires expanded rows into the fake-live sampler/env-probe boundary under a fixed env-row budget. Step 5 makes trial roles and trial budgets visible inside live probe reset/storage/summary. Step 6 confirms search trial rows remain sampler priority evidence and are not PPO-valid rows. It still does not change PPO update timing and does not train from best-trial actions.
```

## Segment Rollout Budget Plan

Aliases:

```text
FrontRESSegmentRolloutBudget
FrontRESSegmentTrialPlan
rollout_trial_count
budget_reason
policy trial
search trial
source_index
trial_index
horizon_k
max_horizon_k
```

Owner and lifecycle:

```text
owner: MAIN-28 frontres_segment_sampler.py
input: selected segment_ids plus sampler-owned segment_state/evidence fields
output: per-segment trial_count/horizon_k and expanded policy-first trial rows
semantic boundary: pure scheduler plan only, no env rollout, no PPO update
future consumer: MAIN-37 live sampler/env loop
```

Required evidence:

```text
S1 budget allocation contract for unknown/promising/frontier/delayed_regret/solved/hopeless
S1 trial expansion contract proving policy row is first for every segment
S1 fixed-row-budget sampling contract proving expanded rows do not mark unexecuted base segments as seen
S2 live sampler contract proving expanded policy/search rows reach dataset batch/probe with role and horizon metadata
S2 live probe contract proving role/index/horizon metadata survives reset request, priority evidence, and printed summary
S2 live probe PPO-boundary contract proving policy/search rows can share a segment id while only the policy row is PPO-valid
S4 live sentinel only after expanded rows are wired into real rollout execution
```

Current status:

```text
Step 3 is S1-confirmed for budget planning and role-ordered trial expansion. Step 4 is S1/S2-confirmed for fixed-row-budget live sampler connectivity: sample_rollout_rows() produces executable policy/search rows, frontres_segment_live_sampler.py attaches frontres_segment_trial_role, frontres_segment_trial_index, and frontres_segment_budget_horizon_k to the batch, and sampler evidence aggregates multiple rows of the same segment. Step 5 is S2-confirmed for live-probe trial metadata visibility across reset/storage/summary. Step 6 is S2-confirmed for PPO-boundary eligibility: repeated policy/search rows can share the same segment id, but only policy rows are valid for PPO. Real IsaacLab S4 rollout execution remains unconfirmed.
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
masked bounded action
unmasked old mean
per-dim sigma
per-dim mean delta
per-dim log-ratio contribution
tanh Jacobian contribution
frontres_mask
paired repaired-vs-noisy gain
actor_update_mask
execution action mask
perturbation family metadata
trial_role
ppo_update_valid_mask
policy trial row
search trial row
ppo_boundary
evidence rows
search evidence-only rows
policy invalid rows
valid_policy_frac
valid_evidence_frac
sampler oracle gap
sampler confidence
delayed_regret diagnostics
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
S1/S2 exact clipped-surrogate, old-policy detach, row-permutation invariance, invalid-row isolation, and full-6D support under single-family perturbation metadata when execution mask is full 6D
S1/S2 ratio diagnostic consistency contract separating pre-loss raw/clamped ratio from post-update raw/clamped ratio
S1/S2 execution-mask projection contract showing how projected or zeroed bounded actions avoid one-dim log-ratio spikes from unprojected raw old/current means when sigma is small
S2 live-probe PPO-boundary contract proving multi-trial `search` rows stay invalid for PPO while `policy` rows remain eligible
S1/S2 diagnostic-boundary contract proving live/probe/update logs expose trial roles, evidence rows, PPO-valid rows, search-only evidence, policy-invalid rows, and sampler oracle quality
S2 paired repaired-vs-noisy reward/advantage route contract
S4 live sentinel when real rollout/update is changed
```

Current status:

```text
Segment storage preserves old_means/old_sigmas through FrontRESSegmentPPOBatch. frontres_rollout_step.py now projects masked bounded Delta SE actions, old log-prob, old_mean, and old_sigma into the same rollout tuple representation before storage. frontres_segment_live_probe.py now writes an execution action mask into Segment storage instead of a hard-coded all-ones mask and gates Segment PPO eligibility by trial role: policy rows can be valid PPO rows, search rows stay priority evidence only. frontres_segment_ppo.py reports both logprob_approx_kl and MOSAIC-style distribution_kl_mean, uses distribution_kl_mean as approx_kl when old/new distribution stats are available, and projects current policy eval on inactive executed dims so log_prob, ratio, and KL compare the same action representation. frontres_segment_algorithm_contract.py now confirms exact distribution KL, exact clipped surrogate behavior, old-policy tensor detach, invalid-row isolation, row-permutation invariance, full-6D PPO support under rp-only perturbation metadata when execution mask is full 6D, advantage-dominance diagnostics, small-sigma KL sensitivity, ratio-source decomposition, execution-mask projection that prevents inactive executed dims from creating sixth-dim ratio spikes, and scale-only advantage normalization preserving all-positive no-regret signs. frontres_segment_live_probe_ppo_contract.py confirms the Step 6 multi-trial PPO boundary: repeated policy/search rows can share the same segment id, priority evidence records both roles, storage/PPO valid_mask keeps only the policy row valid, and PPO valid_count is one. Step 7 diagnostics now make that boundary visible in live/probe/update logs: `frontres_segment_live_probe_contract.py` covers readable `ppo_boundary.*` fields, `frontres_segment_live_sampler_contract.py` covers `sampler.oracle` gap/confidence/delayed-regret output, and update-loop/training pseudo contracts cover the main `trial:` line. run_frontres_segment_single_update defaults Segment PPO advantage normalization to `scale_only`, allows schedule=adaptive pre-step LR reduction when old/new distribution KL is already high, blocks low-pre-KL LR amplification before optimizer.step, recomputes post-update trust-region KL after optimizer.step, reports the post value as the live `ppo.kl`, reports post-update mean_delta from stored old_means, and rolls back adaptive post-KL violations before retrying with a reduced LR. bounded Delta SE log-prob reconstruction is covered by the live single-update contract using raw policy stats plus tanh Jacobian correction, and live probe text now prints ratio-source blocks for raw action tail, per-dim sigma, per-dim mean delta, per-dim log-ratio contribution, and tanh-Jacobian contribution. Step B on 2026-07-09 contract-confirms explicit separation of pre-loss raw-log-ratio, pre-loss clamped-ratio, post-update raw-log-ratio, and post-update clamped-ratio diagnostics, and live text now prints `pre_log_ratio`, `pre_ratio`, `post_log_ratio`, and `post_ratio` without ambiguous `ratio.reported_mean`. The Stage 3 entrypoint and launch contracts now cover explicit `--frontres_segment_ppo_lr` pass-through. This is S1/S2 contract-confirmed; real training quality remains S4/live-log evidence.
```

Current gap:

```text
`action_mask` is preserved through Segment storage and validated by Segment PPO as execution-mask metadata. Direct Delta SE PPO intentionally keeps full-6D repair support when the execution mask is full 6D; perturbation family metadata such as local_rp must not be reinterpreted as a PPO dimension mask. Multi-trial `trial_role` is deliberately resolved before PPO as a row-level valid_mask gate rather than a PPO-batch field. S4 live-log proof is still needed to confirm real rollout/update logs reflect the projected tuple and role-gated PPO semantics.
```
