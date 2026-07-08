# FEMR All-Module Test Control Board

This is the current all-module test matrix for FEMR. It starts from
`note/architecture/architecture/01_repo_architecture.data.json` and should be
updated whenever tests, modules, or impact rules change.

Status values:

```text
covered: current inventory has a suitable test.
covered-but-live-gap: static/offline coverage exists but live-only proof remains.
missing-test: no suitable test is inventoried.
needs-inventory: likely tests exist, but they have not been mapped here yet.
```

## Fast Baseline Commands

```text
python -m py_compile <changed python files>
python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py
python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py
```

Use a torch-capable Python when a contract requires torch. The aggregate suite
already attempts to select one.

## Latest S<=3 Sweep - 2026-07-07

Scope:
- Full-repo test gate up to S3 only: S0 static, S1 module semantics, S2
  offline connectivity, and S3 persistence / semantic-object contracts.
- S4 real live sentinel was intentionally not used as a pass/fail gate for this
  sweep.

Commands passed:
- `frontres/bin/python -m py_compile <changed/untracked python files>`
- `frontres/bin/python -m json.tool note/architecture/architecture/01_repo_architecture.data.json`
- `frontres/bin/python -m json.tool note/architecture/runtime/02_frontres_flow.data.json`
- `/opt/homebrew/bin/node check_viewer_import.mjs`
- S<=3 contract pack: 40 targeted FrontRES/Segment/Balance/Checkpoint
  contract commands.

Evidence:
- S0 Python compile: passed for the current changed/untracked Python set.
- Atlas contract: `roughjs atlas import and data contracts ok; checked 52 repo owner paths`.
- S<=3 contract pack: `total=40 failed=0 elapsed_sec=32.3`.
- Key S3 coverage included cache IO/resume, checkpoint save/load/resume,
  normalizer layout/checkpoint stats, segment storage, and live-resume pseudo
  persistence.

Training gate:
- OK for starting formal training from S0-S3 evidence.
- Remaining non-blocking gaps are S4-only: real simulator physics, long
  training quality, and export/play deployment sink behavior.

## Latest Full Sweep - 2026-07-07
Scope:
- Tested the current dirty FEMR worktree through the Repo Mainline Atlas and
  Method-to-Code Atlas.
- Used `MAIN-*` rows to route changed/risky surfaces to S0/S1/S2/S3/S4 tests.

Commands passed:
- `frontres/bin/python -m py_compile <46 changed python files>`
- `python -m json.tool note/architecture/architecture/01_repo_architecture.data.json`
- `python -m json.tool note/architecture/runtime/02_frontres_flow.data.json`
- `/opt/homebrew/bin/node check_viewer_import.mjs`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_balance_margin_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_balance_obs_cfg_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_balance_offline_connectivity_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sentinel_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py`
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py`

Evidence:
- `frontres_segment_stage3_pseudo_suite.py`: `contract_count=17 failed_count=0`.
- `frontres_segment_all_contract_suite.py`: `contract_count=41 failed_count=0`.
- Atlas contract: `roughjs atlas import and data contracts ok; checked 52 repo owner paths`.

Atlas readability finding:
- The 01 Repo Mainline Atlas is useful for routing broad changed surfaces to
  tests, especially observation/layout/balance, Segment Replay, storage,
  algorithm, checkpoint, and sequence-eval boundaries.
- The remaining unclear rows are the existing missing-test rows: MAIN-09
  termination, MAIN-49 DR sweep eval, MAIN-50 export sink, and base RSL-RL
  inventory for MAIN-21/45.
- Passing this sweep does not prove simulator physics, long training quality,
  or export/play normalizer behavior.
## Matrix

| MAIN | Owner | Module type | Required S | Required T | Current tests | Evidence | Status | Gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MAIN-01 | `scripts/rsl_rl/train.py` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `frontres_stage_entrypoint_contract.py`; `frontres_segment_stage3_launch_command_contract.py` | connectivity-confirmed | covered | Full Hydra/live launch still separate. |
| MAIN-02 | `scripts/rsl_rl/cli_args.py` | Entrypoint / CLI / config | S0 | T-connect | `frontres_segment_stage3_launch_command_contract.py` | static-confirmed | covered | None known. |
| MAIN-03 | `scripts/rsl_rl/play.py` | Checkpoint / resume / export / play | S3, S4 when deployment changes | T-persist, T-order, T-diff, T-live | none mapped | unconfirmed | missing-test | Need play/export normalizer sink inventory. |
| MAIN-04 | `rsl_rl_cfg.py` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `frontres_stage_entrypoint_contract.py`; launch command contract | connectivity-confirmed | covered | None known. |
| MAIN-05 | `tracking/config/` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | `frontres_balance_obs_cfg_contract.py` | static-confirmed | needs-inventory | Broader task config tests not mapped. |
| MAIN-06 | `tracking_env_cfg.py` | Entrypoint / CLI / config | S0, S2 | T-connect, T-oracle | launch/config contracts indirect | note-confirmed | needs-inventory | Need static env config scan. |
| MAIN-07 | `mdp/commands.py` | Env command / reset / lifecycle adapter | S2, S4 when live behavior changes | T-connect, T-oracle, T-live | sequence/reset contracts indirect | note-confirmed | covered-but-live-gap | Live env command behavior is S4. |
| MAIN-08 | `mdp/events.py` | Env command / reset / lifecycle adapter | S2, S4 when live behavior changes | T-connect, T-oracle, T-live | reset hook contracts indirect | connectivity-confirmed | covered-but-live-gap | Real IsaacLab reset remains S4. |
| MAIN-09 | `mdp/terminations.py` | Env command / reset / lifecycle adapter | S1, S2, S4 when live behavior changes | T-value, T-oracle, T-live | none mapped | unconfirmed | missing-test | Need static or tiny termination fixture. |
| MAIN-10 | `mdp/observations.py` | Tensor layout / adapter | S1, S2, S4 when env observation changes | T-shape, T-order, T-mask, T-transform, T-connect, T-live | balance offline connectivity; observation layout contract indirect | connectivity-confirmed | covered-but-live-gap | Full observation group requires env-aware tests. |
| MAIN-11 | `mdp/balance.py` | Reward / metric / evaluator | S1, S2 | T-value, T-meta, T-oracle, T-connect | `frontres_balance_margin_contract.py`; offline connectivity | contract-confirmed | covered | None known. |
| MAIN-12 | `frontres_observation_layout.py` | Tensor layout / adapter | S1, S2, S3 when stats persist | T-shape, T-order, T-mask, T-transform, T-persist | `frontres_observation_layout_contract.py` | contract-confirmed | covered | Add S3 when stats persist. |
| MAIN-13 | `mdp/motion_perturbations.py` | Env command / reset / lifecycle adapter | S1, S2, S4 when env perturb changes | T-value, T-dist, T-oracle, T-live | burst/segment perturbation contracts likely | needs-inventory | needs-inventory | Map exact perturbation tests. |
| MAIN-14 | `frontres_dr_curriculum.py` | Sampler / curriculum / priority | S1, S2 | T-dist, T-meta, T-role | aggregate suite cache/curriculum targets | contract-confirmed | covered | Distribution stress when schedule changes. |
| MAIN-15 | `perturbation_runtime.py` | Env command / reset / lifecycle adapter | S2, S4 when live behavior changes | T-connect, T-oracle, T-live | aggregate suite indirect | connectivity-confirmed | covered-but-live-gap | Runtime env hook is S4. |
| MAIN-16 | `mdp/rewards.py` | Reward / metric / evaluator | S1, S2, S4 when env reward changes | T-value, T-role, T-oracle, T-meta, T-diff, T-live | reward tests indirect | needs-inventory | needs-inventory | Map direct reward contracts. |
| MAIN-17 | `frontres_segment_reward.py` | Reward / metric / evaluator | S1, S2 | T-value, T-role, T-oracle, T-meta, T-diff | `frontres_segment_reward_contract.py`; aggregate suite | contract-confirmed | covered | None known. |
| MAIN-18 | `frontres_reward_diagnostics.py` | Diagnostics / notes / atlas | S0, S1 | T-connect, T-oracle, T-role | reward diagnostics contracts likely | needs-inventory | needs-inventory | Map exact diagnostics tests. |
| MAIN-19 | `front_residual_actor_critic.py` | Algorithm / loss / optimizer | S1, S2 | T-shape, T-order, T-mask, T-grad, T-connect | `frontres_task_space_proposal_only_contract.py`; `frontres_stage3_noise_std_migration_contract.py`; HSL policy tests likely | contract-confirmed | covered | Live policy quality not claimed. |
| MAIN-20 | `normalizer.py` | Checkpoint / resume / export / play | S1, S3, S4 when deployment stats change | T-shape, T-order, T-persist, T-diff, T-live | observation layout contract; checkpoint contract indirect | persistence-confirmed | covered-but-live-gap | Export/play stats sink missing. |
| MAIN-21 | `actor_critic.py` | Algorithm / loss / optimizer | S1, S2 | T-shape, T-grad, T-connect | none mapped | unconfirmed | needs-inventory | Base RSL-RL tests not mapped. |
| MAIN-22 | `task_space_correction.py` | Tensor layout / adapter | S1, S2 | T-shape, T-order, T-mask, T-transform, T-value | proposal-only contract; action cone tests indirect | contract-confirmed | covered | None known. |
| MAIN-23 | `frontres_action_cone.py` | Tensor layout / adapter | S1, S2 | T-shape, T-mask, T-value, T-meta | aggregate suite indirect | contract-confirmed | covered | Add direct cone fixture if action cone changes. |
| MAIN-24 | `frontres_executable_floor.py` | Reward / metric / evaluator | S1, S2 | T-value, T-mask, T-oracle, T-meta | aggregate/executability contracts likely | needs-inventory | needs-inventory | Map direct floor tests. |
| MAIN-25 | `frontres_alpha_rho_bridge.py` | Algorithm / loss / optimizer | S1, S2 | T-shape, T-mask, T-value, T-grad, T-connect | authority/alpha/rho tests likely | needs-inventory | needs-inventory | Map exact alpha/rho tests. |
| MAIN-26 | `frontres_segment_cache_builder.py` | Storage / batch tuple | S1, S2, S3 | T-shape, T-order, T-connect, T-persist | aggregate suite cache_builder | contract-confirmed | covered | None known. |
| MAIN-27 | `frontres_segment_dataset.py` | Storage / batch tuple | S1, S2 | T-shape, T-order, T-role, T-connect | aggregate suite dataset | contract-confirmed | covered | None known. |
| MAIN-28 | `frontres_segment_sampler.py` | Sampler / curriculum / priority | S1, S2 | T-dist, T-meta, T-role | aggregate suite sampler; live sampler contract | contract-confirmed | covered | Distribution stress for sampler policy changes. |
| MAIN-29 | `frontres_segment_reset.py` | Env command / reset / lifecycle adapter | S2, S4 when live behavior changes | T-connect, T-oracle, T-live | aggregate suite reset; reset hook contract | connectivity-confirmed | covered-but-live-gap | Real env reset is S4. |
| MAIN-30 | `on_policy_runner.py` | Runner / orchestration | S2, S4 for live-only boundaries | T-connect, T-oracle, T-live | runner lifecycle/boundary contracts | connectivity-confirmed | covered-but-live-gap | Full training run not claimed. |
| MAIN-31 | `frontres_training_setup.py` | Runner / orchestration | S2 | T-connect, T-oracle | aggregate suite runner targets indirect | connectivity-confirmed | covered | None known. |
| MAIN-32 | `frontres_checkpointing.py` | Checkpoint / resume / export / play | S3 | T-persist, T-order, T-diff | checkpoint contracts; `frontres_stage3_noise_std_migration_contract.py` | persistence-confirmed | covered | Export/play sink separate. |
| MAIN-33 | `frontres_rollout_step.py` | Runner / orchestration | S2, S4 for live-only boundaries | T-connect, T-oracle, T-live | runner/rollout contracts indirect | connectivity-confirmed | covered-but-live-gap | Live rollout is S4. |
| MAIN-34 | `frontres_post_step_connector.py` | Storage / batch tuple | S1, S2 | T-shape, T-order, T-mask, T-connect | storage/algorithm contracts indirect | connectivity-confirmed | covered | None known. |
| MAIN-35 | `frontres_hsl_rollout_target.py` | Tensor layout / adapter | S1, S2 | T-shape, T-mask, T-value, T-transform | HSL acceptance tests likely | needs-inventory | needs-inventory | Map exact HSL target tests. |
| MAIN-36 | `frontres_runtime.py` | Runner / orchestration | S2, S4 for live-only boundaries | T-connect, T-oracle, T-live | balance offline connectivity; runtime contracts likely | connectivity-confirmed | covered-but-live-gap | Deployment sink tests incomplete. |
| MAIN-37 | `frontres_segment_live_sampler.py` | Sampler / curriculum / priority | S1, S2, S4 when live sampling changes | T-dist, T-role, T-connect, T-live | live sampler contract; diagnostics contract; pseudo suite | contract-confirmed | covered | Diagnostics contract verifies live `sampler_update_replay_candidate_count` is the displayed replay candidate field. |
| MAIN-38 | `frontres_segment_live_training.py` | Runner / orchestration | S2, S4 | T-connect, T-oracle, T-live | diagnostics contract; pseudo suite; live sentinel | runtime-confirmed | covered-but-live-gap | Expensive full training not claimed; Motion Quality missing positions now report `UNCONFIRMED`. |
| MAIN-39 | `frontres_segment_live_update_loop.py` | Algorithm / loss / optimizer | S2 | T-connect, T-grad, T-oracle | live update loop contract | connectivity-confirmed | covered | None known. |
| MAIN-40 | `frontres_segment_sequence_eval.py` | Reward / metric / evaluator | S2, S4 | T-role, T-oracle, T-meta, T-diff, T-live | diagnostics contract; sequence eval contract; live sentinel | connectivity-confirmed | covered-but-live-gap | Real long sequence eval is S4; diagnostics contract covers Periodic Eval score/motion/action formatting and missing-data reporting. |
| MAIN-41 | `rollout_storage.py::Transition` | Storage / batch tuple | S1, S2, S3 if persisted | T-shape, T-order, T-mask, T-persist | storage contract; aggregate suite | contract-confirmed | covered | None known. |
| MAIN-42 | `rollout_storage.py::add_transitions` | Storage / batch tuple | S1, S2 | T-shape, T-order, T-mask, T-connect | storage contract | contract-confirmed | covered | None known. |
| MAIN-43 | `rollout_storage.py::mini_batch_generator` | Storage / batch tuple | S1, S2 | T-shape, T-order, T-mask, T-connect | storage + algorithm contracts | contract-confirmed | covered | None known. |
| MAIN-44 | `frontres_segment_storage.py` | Storage / batch tuple | S1, S2, S3 if persisted | T-shape, T-order, T-mask, T-connect, T-persist | segment storage contract | contract-confirmed | covered | None known. |
| MAIN-45 | `ppo.py` | Algorithm / loss / optimizer | S1, S2 | T-mask, T-value, T-grad, T-connect | none mapped | unconfirmed | needs-inventory | Base PPO tests not mapped. |
| MAIN-46 | `frontres_segment_ppo.py` | Algorithm / loss / optimizer | S1, S2 | T-mask, T-value, T-grad, T-connect | segment algorithm contract; update loop contract | contract-confirmed | covered | None known. |
| MAIN-47 | `frontres_unified.py` | Algorithm / loss / optimizer | S1, S2 | T-mask, T-value, T-grad, T-connect | segment algorithm contract; authority/HSL tests likely | contract-confirmed | covered | Map exact sub-loss coverage. |
| MAIN-48 | `frontres_segment_checkpointing.py` | Checkpoint / resume / export / play | S3 | T-persist, T-order, T-diff | checkpoint/resume contracts; `frontres_stage3_noise_std_migration_contract.py` | persistence-confirmed | covered | None known. |
| MAIN-49 | `frontres_dr_sweep_eval.py` | Reward / metric / evaluator | S1, S2 | T-dist, T-oracle, T-diff, T-connect | none mapped | unconfirmed | missing-test | Need sweep eval static/connectivity test. |
| MAIN-50 | `whole_body_tracking/utils/exporter.py` | Checkpoint / resume / export / play | S3, S4 when deployment changes | T-persist, T-order, T-diff, T-live | none mapped | unconfirmed | missing-test | Need normalizer/export sink test. |
| MAIN-51 | `tests/frontres_*_contract.py` | Diagnostics / notes / atlas | S0, S1 | T-connect, T-oracle | this matrix | note-confirmed | covered | Keep inventory updated. |
| MAIN-52 | `tests/frontres_segment_*_contract.py` | Diagnostics / notes / atlas | S0, S1 | T-connect, T-oracle | aggregate suite | contract-confirmed | covered | None known. |
| MAIN-53 | `frontres_segment_stage3_pseudo_suite.py` | Diagnostics / notes / atlas | S0, S2 | T-connect, T-oracle | pseudo suite itself | connectivity-confirmed | covered | None known. |
| MAIN-54 | `frontres_segment_live_sentinel_contract.py` | Diagnostics / notes / atlas | S0, S4 | T-live, T-oracle | live sentinel itself | runtime-confirmed | covered | Real training duration not claimed. |

## Current High-Priority Missing Tests

1. MAIN-20/50: normalizer mean/std through export/play sink.
2. MAIN-49: DR sweep eval static/connectivity coverage.
3. MAIN-09: termination tiny/static fixture.
4. MAIN-21/45: base RSL-RL actor/PPO inventory.

## Maintenance Rule

Every new test should update `test_inventory.md` and the corresponding rows
above. Every cross-cutting bug should update `impact_rules.md` and
`semantic_objects.md`.
