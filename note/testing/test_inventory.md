# FEMR Test Inventory

This file records reusable tests for all-module coordination. It maps tests to
repo atlas blocks, test tiers, commands, and evidence classes.

## Tier Legend

```text
S0 Static: syntax/import/path/symbol/config/schema checks.
S1 Module Semantic: local deterministic tiny/golden/metamorphic tests.
S2 Offline Connectivity: fake env/runner/batch/storage/model chain.
S3 Persistence / Semantic Object: stats, checkpoint, resume, export, eval.
S4 Live Sentinel: minimal real runtime boundary.
```

## Existing Test Assets

| Test asset | Tier | Covers | Command | Evidence | Notes |
| --- | --- | --- | --- | --- | --- |
| `python -m py_compile <changed python files>` | S0 | Any changed Python file | `python -m py_compile ...` | static-confirmed | Syntax only, not behavior. |
| `note/architecture/auxiliary/atlas_app/check_viewer_import.mjs` | S0 | Atlas data/viewer contract, MAIN owner paths | `node check_viewer_import.mjs` from `note/architecture/auxiliary/atlas_app` | static-confirmed | Validates 01/02 layout contracts and concrete owner paths. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py` | S1/S2/S4 | MAIN-12, MAIN-22/29, MAIN-31/48, MAIN-41/47, MAIN-54 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py` | contract-confirmed | Main Segment Replay aggregate suite. Uses torch-capable python when needed. |
| `source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py` | S1/S3 | MAIN-12, MAIN-20, M2C-04 | `python source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py` | contract-confirmed | Layout and stats boundary anchor. |
| `source/rsl_rl/rsl_rl/tests/frontres_balance_margin_contract.py` | S1 | MAIN-11 | `python source/rsl_rl/rsl_rl/tests/frontres_balance_margin_contract.py` | contract-confirmed | Balance math fixture. |
| `source/rsl_rl/rsl_rl/tests/frontres_balance_obs_cfg_contract.py` | S0/S1 | MAIN-05, MAIN-11 | `python source/rsl_rl/rsl_rl/tests/frontres_balance_obs_cfg_contract.py` | static-confirmed | Config-level balance observation coverage. |
| `source/rsl_rl/rsl_rl/tests/frontres_balance_offline_connectivity_contract.py` | S2 | MAIN-10, MAIN-11, MAIN-36 | `python source/rsl_rl/rsl_rl/tests/frontres_balance_offline_connectivity_contract.py` | connectivity-confirmed | Offline observation/connectivity check. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py` | S3 | MAIN-20, MAIN-32, MAIN-48, M2C-19 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py` | persistence-confirmed | Segment checkpoint contract. |
| `source/rsl_rl/rsl_rl/tests/frontres_stage3_noise_std_migration_contract.py` | S3 | MAIN-19, MAIN-32, MAIN-48 | `python source/rsl_rl/rsl_rl/tests/frontres_stage3_noise_std_migration_contract.py` | persistence-confirmed | Regression for legacy 12D checkpoint std not polluting current 6D Stage 3 policy std. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py` | S2/S4 | MAIN-40, M2C-20 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py` | connectivity-confirmed | Sequence eval owner and live-boundary probes. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py` | S2 | MAIN-37/40, MAIN-53 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py` | connectivity-confirmed | Cheap Stage 3 pseudo path. |
| S<=3 contract pack | S0/S1/S2/S3 | MAIN-01/54 offline and persistence gate | Run the targeted 40-command FrontRES contract pack from the latest control-board sweep | contract-confirmed / persistence-confirmed | Excludes S4 real live sentinel as a gate; includes pseudo/live-named offline contracts where they validate fake hooks or persistence. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_sentinel_contract.py` | S4 | MAIN-38/40, MAIN-54 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sentinel_contract.py` | runtime-confirmed | Minimal live sentinel contract. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py` | S1/S2 | MAIN-41/44, M2C-14/15 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py` | contract-confirmed | Storage field and batch boundary, including old_means/old_sigmas preservation into PPO batch. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py` | S1/S2 | MAIN-46/47, M2C-16/18 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py` | contract-confirmed | Segment algorithm update contract, including positive-advantage gradient direction toward stored 6D Delta SE actions, exact old-stat distribution KL, exact clipped surrogate cases, old-policy tensor detach, invalid-row isolation, row-permutation invariance, and full-6D PPO support under rp-only action-mask metadata. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py` | S1/S2 | MAIN-28, MAIN-37 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py` | contract-confirmed | Live sampler contract and per-sample probes. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py` | S1/S2 | MAIN-19/37/38/40 diagnostics | `python source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py` | contract-confirmed | Regression for live summary `replay_candidates` field alias, Motion Quality missing-position data reporting as `UNCONFIRMED`, Periodic Eval score/motion/action formatting, and raw action-distribution saturation health. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py` | S2/S4 | MAIN-29, MAIN-38 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_reset_hook_contract.py` | connectivity-confirmed | Reset/reference-window hook coverage. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_single_update_contract.py` | S2 | MAIN-39, MAIN-46 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_single_update_contract.py` | connectivity-confirmed | Live single-update contract, including old-stat pre KL, post-update trust-region KL reporting, optimizer parameter delta, and post-KL adaptive LR route. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_update_loop_contract.py` | S2 | MAIN-39, MAIN-46 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_update_loop_contract.py` | connectivity-confirmed | Live update loop wiring. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_live_resume_pseudo_contract.py` | S3 | MAIN-32, MAIN-48 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_live_resume_pseudo_contract.py` | persistence-confirmed | Resume pseudo coverage. |
| `source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py` | S0/S2 | MAIN-01, MAIN-04, MAIN-30 | `python source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py` | connectivity-confirmed | Stage entrypoint contract. |
| `source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py` | S0 | MAIN-01, MAIN-04, MAIN-38 | `python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py` | static-confirmed | Launch command/config contract, including sequence_eval smoke env overrides for sequence count, rollout steps, and max preroll. |

## Inventory Gaps

| Gap | Affected blocks | Needed tier | Suggested next test |
| --- | --- | --- | --- |
| Segment PPO `action_mask` is confirmed not to reduce direct Delta SE PPO support; missing live-only proof remains for real simulator perturbation-family accumulation. | MAIN-19, MAIN-22, MAIN-33, MAIN-39, MAIN-44, MAIN-46 | S4 | Short live sentinel with local_rp curriculum should print full-6D action stats and post-update full-6D gradient/parameter-delta summaries. |
| Export/play normalizer sink inventory is incomplete. | MAIN-03, MAIN-20, MAIN-50, M2C-21 | S3 | Fake checkpoint -> export/play loader -> assert expected normalizer dims and keys. |
| Stage 3 checkpoint health is not yet live-gated before long training. | MAIN-19, MAIN-32, MAIN-38/40 | S3/S4 | Checkpoint load -> one batch policy forward -> fail/warn when raw mean is saturated before starting formal training. |
| Env construction/reset coverage is mostly indirect. | MAIN-07/09, MAIN-29, MAIN-38 | S4 | Minimal live reset sentinel when env lifecycle changes. |
| Base PPO/GMT-only path is not fully mapped in this matrix. | MAIN-21, MAIN-45 | S1/S2 | Existing non-FrontRES tests should be inventoried or added. |
| Static stale-comment/stale-atlas scan is not yet scripted. | All MAIN/M2C | S0 | Add repo-local scan for old dimensions, legacy route names, and stale test references. |
