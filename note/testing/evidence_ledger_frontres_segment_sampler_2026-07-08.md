# FrontRES Segment Sampler Evidence Ledger

Date: 2026-07-08

Purpose: replace the vague claim "boundary is clean" with auditable evidence.

## Claims And Evidence

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The mistaken `single_update_override` plumbing is absent from current source/note state. | static-confirmed | Search command: `ctx_search("single_update_override|_frontres_segment_live_last_storage_batch|_attach_sampler_ppo_update_summary|sampler_update_order|before_ppo_update|sampler-before-PPO|sampler update -> PPO|Level Replay-style sampler-before-PPO", source+rsl_rl+note)` returned 0 matches. | No searched literal remnants remain in `source/rsl_rl/rsl_rl` or `note`. | It does not prove semantic absence of every possible equivalent implementation. |
| The public runner wrapper no longer accepts or forwards `single_update_override`. | code-confirmed | `source/rsl_rl/rsl_rl/runners/on_policy_runner.py:687-694` has `run_frontres_segment_live_probe(self, init_at_random_ep_len=True)` forwarding only `init_at_random_ep_len`. | The wrapper interface no longer exposes the removed override. | It does not prove all runtime modes are healthy. |
| The live sampler step uses rollout summary to build sampler evidence, then calls `sampler.update_with_probe(evidence)`. | code-confirmed | `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py:213-232` calls live probe, builds `build_live_sampler_evidence(...)`, then calls `sampler.update_with_probe(evidence)`. | Sampler update input is the evidence object built from sample, rollout summary, horizon, and reset result. | It does not prove policy quality or physics correctness. |
| The retained regression test poisons PPO post-update diagnostics without changing sampler evidence. | contract-confirmed | `source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py:531-564` injects `ppo_post_update_distribution_kl_mean=1.0e9`, post-update ratios, and `ppo_param_delta_l2=1.0e9`, then asserts valid mask, gain, noisy/repaired scores, and update valid count from rollout evidence. | Sampler priority evidence is isolated from those post-update PPO diagnostic fields in this controlled S1/S2 contract. | It does not prove a full S4 training run has no unrelated sampler bug. |
| The test actually executed in the current worktree. | contract-confirmed | Command: `/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`; output included `[probe evidence-ppo-isolation] ... priority_after=0.043500` and `frontres_segment_live_sampler_contract: ok`. | The evidence-isolation test ran and passed now, not only existed in source. | It does not replace live IsaacLab training validation. |
| The concept note no longer claims literal Level Replay call-order alignment. | note-confirmed | `note/frontres_core/contracts/history/sources/segment_replay/references/external_code_reuse_map.md:560-576` says live flow is `segment_reward -> PPO update -> sampler priority update`, then states Level Replay alignment is semantic: rollout-time evidence independent of post-update PPO diagnostics. | The preserved design source distinguishes semantic evidence alignment from literal call-order changes. | It does not prove future edits will preserve this unless tests remain in CI/manual gate. |
| The test matrix records the same semantic boundary. | note-confirmed | `note/testing/test_control_board.md:186` records MAIN-37 as evidence-isolation / policy-update-independent sampler evidence, not sampler-before-PPO order. | The all-module-test inventory now points humans to the intended evidence boundary. | It does not enforce honesty by itself unless reports cite evidence rows. |

## Commands Re-run In This Audit

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py source/rsl_rl/rsl_rl/runners/on_policy_runner.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py

/Users/chengyuxuan/ArtiIntComVis/MOSAIC/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
```

Observed result:

```text
py_compile: exit 0
sampler contract: frontres_segment_live_sampler_contract: ok
required probe observed: [probe evidence-ppo-isolation]
```

## Anti-Black-Box Rule Extracted

Future `all-module-test` reports must not use ungrounded closing phrases such as
"clean", "covered", "OK", "safe", or "fixed" unless every such phrase is backed
by an evidence row with:

1. exact claim;
2. file lines or command;
3. observed output fact;
4. S tier and T kind;
5. explicit limitation.

If a claim lacks one of these fields, report it as `unconfirmed`, not as `clean`.

## E23 / 2026-07-13 Active Legacy-Owner Cleanup

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Detached Segment checkpoint two-head compatibility is removed from the active tree. | static/code-confirmed | Deleted `source/rsl_rl/rsl_rl/runners/frontres_segment_checkpointing.py`, its two-head checkpoint contract, and the obsolete migration test; removed two-head migration from `frontres_checkpointing.py`. | Formal checkpoint ownership is the strict full-6D runner checkpoint path. | It does not prove an external legacy checkpoint remains loadable; that is intentionally retired. |
| Active FrontRES model/config no longer declares acceptance, authority, state-router, structured-rho, or task-confidence fields. | static/code-confirmed | `front_residual_actor_critic.py`, both FrontRES config owners, and stage pseudo contract were scanned and compiled after cleanup. | The active model/config boundary is full-6D Delta SE(3), not a disabled legacy head. | It does not cover the retained original MOSAIC algorithm branch. |
| Task-space sampling and execution now use one raw Gaussian plus one bounded full-6D transform. | contract-confirmed | `frontres_actual_policy_distribution_contract.py` and `frontres_task_space_proposal_only_contract.py` both returned `ok`. | Raw distribution stats, bounded action, inverse-transform log-prob, and six-dimensional shape agree in the controlled contract. | It does not prove IsaacLab runtime behavior. |
| Active task correction no longer routes through oracle/stable/tri-anchor/authority branches. | code-confirmed | `task_space_correction.py` now only applies full-6D position/orientation rows plus contact-consistent root-z projection; `frontres_rollout_step.py` no longer refreshes state-alpha routing. | Command application has one active owner and one action representation. | Remaining generic logger branches are still pending cleanup. |
| The current local contract baseline remains green after cleanup. | contract-confirmed | `python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py` returned `contract_count=40 failed_count=0 total_marker_count=40`. | No covered Segment contract regressed. | This is not S4 live evidence and does not authorize formal training yet. |

## E24 / 2026-07-13 Generic Logger Cleanup

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The generic runner logger no longer contains retired FrontRES display/routing branches. | static/code-confirmed | `frontres_runner_logging.py` was rebuilt around Stage 1, generic PPO, full-6D supervised, and active Gain/geometry fields; a retired-symbol scan returned zero matches. | Logger output cannot reintroduce the removed objective/authority/state-routing vocabulary through this owner. | It does not remove unrelated stale producers in other modules; those require the final repository symbol scan. |
| The logger preserves its callable surface and does not regress covered Segment contracts. | compile/contract-confirmed | `python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_runner_logging.py`; aggregate suite returned `contract_count=40 failed_count=0 total_marker_count=40`; `git diff --check` passed. | The edited logger is syntactically valid and the existing offline contract baseline remains green. | It does not prove real TensorBoard emission or S4 simulator behavior; an isolated import probe was blocked by the local environment's missing `isaaclab` package. |
| The repository-wide retired-symbol scan is not yet clean. | static/code-confirmed | The earlier scan found residual references in `frontres_reward_diagnostics.py`, `on_policy_runner.py`, runtime/config helpers, and legacy-only tests; the reward-diagnostics/connector portion was subsequently retired in E26. | The earlier logger cleanup was bounded correctly. | This row is historical evidence for the pre-E26 state and must not be read as current code status. |

## E25 / 2026-07-13 Dead Runner/Runtime Branch Removal

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The active runner no longer synchronizes or validates retired acceptance/structured/state-router configuration. | code/compile/contract-confirmed | Removed the unused `on_policy_runner.py` sync/validation helpers and their initialization calls; removed the unused executable-floor wrapper. `py_compile` and the Stage 3 entrypoint pseudo contract passed. | The current runner constructor no longer creates a retired configuration route. | It does not prove the standard FrontRES reward-diagnostic route is current design. |
| Inference and HSL target helpers now use the full-6D route only. | code/compile/contract-confirmed | Removed state-router inference probing; removed HSL `hsl_hybrid/task_conf_dim` branching and retained one mixed full-6D target path; reduced schedule objective recognition to `supervised_restore`. Aggregate contract suite remained `40/40`. | These helper owners no longer contain the deleted branch conditions. | It does not authorize deleting `frontres_reward_diagnostics.py`, which remains called by standard rollout code. |
| Standard rollout reward diagnostics was an open boundary before E26. | reachability-confirmed | `on_policy_runner.py` previously called `compute_frontres_training_truth`, `compute_frontres_reward`, and `materialize_frontres_reward_diagnostic_means` for non-segment FrontRES rollout. | The historical reason for the E26 boundary is preserved. | It does not describe the current post-E26 call graph. |

## E26 / 2026-07-14 Stage 2 Reward Boundary Removal

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Stage 2 no longer consumes the generic FrontRES reward/diagnostic connector. | code/compile/contract-confirmed | `on_policy_runner.py` now routes `supervised_restore` to zero environment reward plus the full-6D HSL target; non-supervised FrontRES reaching the old loop raises explicitly. Removed `frontres_post_step_connector.py` and `frontres_reward_diagnostics.py`; removed their package exports and dedicated diagnostics tests. | Stage 2 policy learning has one active signal: HSL supervised full-6D restoration. | It does not prove real Stage 2 HSL target population or checkpoint statistics; those remain runtime boundaries. |
| The deleted reward connector/diagnostics owner has no active source caller. | static/code-confirmed | Source scan returns no `frontres_post_step_connector`, `frontres_reward_diagnostics`, `compute_frontres_reward`, or `materialize_frontres_reward_diagnostic_means` references; current architecture data was updated. | The deleted files are not hidden behind an active import path. | History documents and unrelated legacy helpers still contain historical names by design. |
| The active Segment baseline remains green after Stage 2 boundary removal. | compile/contract-confirmed | `py_compile` on the edited runner/package/reward-window files; aggregate suite returned `contract_count=40 failed_count=0 total_marker_count=40`; `git diff --check` passed. | Covered Segment contracts did not regress. | This is offline evidence, not S4 live training evidence. |

## E27 / 2026-07-14 Runtime Diagnostics and Reward-Window Owner Cleanup

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Stage 2 restore diagnostics now reports one full-6D action/target/write path. | code/compile-confirmed | `runtime_diagnostics.py` retains the runner-facing function and removes `basis_restore`, `hsl_hybrid`, `task_conf_dim`, and acceptance/gate branches; `py_compile` passed. | The diagnostic cannot silently describe a retired confidence/acceptance interface. | It does not populate real Stage 2 runtime values. |
| Segment physics evidence no longer depends on the generic reward-window owner. | reachability/code-confirmed | `_frontres_branch_balance_margin` moved to `frontres_balance.py`; the only production caller in `frontres_segment_live_probe.py` imports that owner; the generic `frontres_reward_window.py` was deleted and package exports were removed. | Balance evidence remains reachable through a module whose ownership matches the active Segment Gain boundary. | It does not prove live ZMP/contact sensor population. |
| The active offline baseline remains green after the owner migration. | compile/contract-confirmed | `py_compile`, `git diff --check`, and `frontres_segment_all_contract_suite.py` returned `contract_count=40 failed_count=0 total_marker_count=40`. | Covered contracts and syntax did not regress. | This is offline evidence, not S4 live evidence or authorization for formal training. |

## E28 / 2026-07-14 Temporal Reference Cache Retirement

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The active full-6D command writer no longer maintains the retired temporal-rejoin cache. | reachability/code-confirmed | `frontres_temporal_continuity_correction` had no caller; the update/invalidate calls were removed from `task_space_correction.py`, `on_policy_runner.py`, and `frontres_dr_sweep_eval.py`; `temporal_reference_cache.py` and package exports were deleted. | Full-6D Delta SE(3) is written directly to the command term without an unused temporal side state. | It does not remove or weaken K-step Segment replay history or the Gain temporal-change regularizer. |
| The retirement is protected by a full-6D regression contract. | contract-confirmed | `frontres_full6_no_active_mask_contract.py` now asserts no temporal-cache import or field in the active writer/package; it returned `frontres_full6_no_active_mask_contract: ok`. | Future active-path edits cannot silently reintroduce this cache boundary without failing the contract. | It does not prove simulator command timing. |
| The complete offline baseline remains green. | compile/contract-confirmed | `py_compile`, `git diff --check`, and `frontres_segment_all_contract_suite.py` returned `contract_count=40 failed_count=0 total_marker_count=40`. | Covered formal contracts did not regress. | This is offline evidence, not S4 live evidence. |

## E29 / 2026-07-14 Compatibility Config and Legacy-Test Retirement

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| Stage presets no longer write retired `task_conf_dim`, acceptance-head, or state-router compatibility fields. | code/contract-confirmed | Removed the six no-op `_set_if_present` calls from `scripts/rsl_rl/train.py`; `frontres_stage_entrypoint_contract.py` now asserts their absence. | Stage 2/3 entrypoint config cannot silently recreate retired policy interfaces. | It does not validate a remote Hydra config outside this repository. |
| Current FEMR config no longer exposes confidence or basis-restore coefficient fields. | static/code-confirmed | Removed unused `supervised_conf_loss_weight` and `supervised_coeff_*` fields from `whole_body_tracking/utils/rsl_rl_cfg.py`; no current FEMR consumer remains. | Current config ownership matches the full-6D actor contract. | The original MOSAIC config/algorithm remains untouched by design. |
| Unconnected 12D/`hsl_hybrid` tests were removed. | reachability-confirmed | Deleted `frontres_advantage_learning.py`, `frontres_region_direct_update_path.py`, and `frontres_update_memory_pipeline.py`; repository search found no callers outside those files. | The reusable test surface no longer advertises retired training paths. | Historical notes and the original MOSAIC branch are not rewritten. |
| Active Stage 2/3 contracts remain green. | compile/contract-confirmed | Correct interpreter runs passed the policy distribution and proposal-only contracts; aggregate suite returned `contract_count=40 failed_count=0 total_marker_count=40`; `git diff --check` passed. | The current full-6D route remains syntactically and contractually intact. | This is offline evidence, not S4 live training evidence. |

## E30 / 2026-07-14 No-Consumer Executable-Floor and DR-Sweep Cleanup

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| The executable-floor owner had no current FEMR production caller. | reachability/code-confirmed | Source scan found only package exports and `tests/test_frontres_executable_floor.py`; the module, test, package exports, and current FEMR config fields were removed. The MOSAIC config was intentionally left untouched. | The retired score-floor/alpha calibration owner cannot enter the active route through current FEMR imports or config. | It does not change the original MOSAIC branch. |
| DR sweep no longer writes retired state-router-alpha diagnostics. | code/contract-confirmed | Removed the `get_state_router_alpha` block and `_frontres_state_alpha_*` assignments from `frontres_dr_sweep_eval.py`; the full-6D retirement contract asserts absence. | Evaluation no longer creates unused alpha state as a side effect. | It does not prove real simulator evaluation quality. |
| The rho replay test was not part of any current caller graph. | reachability-confirmed | `frontres_live_batch_replay.py` had only its own command documentation and definitions; it was deleted as a retired TEST-ONLY path. | The reusable test surface no longer advertises rho replay. | Historical notes remain unchanged. |
| The active offline baseline remains green after cleanup. | compile/contract-confirmed | Targeted retirement contract returned `frontres_full6_no_active_mask_contract: ok`; aggregate suite returned `contract_count=40 failed_count=0 total_marker_count=40`; `git diff --check` passed. | The remaining active Stage 2/3 contracts did not regress. | This is offline evidence, not S4 live evidence. |

## E31 / 2026-07-14 Storage Mask and Actor-Gate Dataflow Audit

| Claim | Evidence type | Authoritative evidence | What it proves | What it does not prove |
| --- | --- | --- | --- | --- |
| `frontres_actor_gate` was a retired gate, not an active split-env parameter. | reachability/code-confirmed | No production writer existed; storage only defaulted it to ones and `FrontRESUnified` multiplied it into the generic PPO actor mask. The field, batch tuple entry, and loss argument were removed. | The active storage/update tuple no longer carries an unused acceptance-style actor gate. | It does not prove the remaining generic PPO implementation is needed. |
| `frontres_mask` remains an active row-role mask. | code-confirmed | `frontres_rollout_step.py` constructs zeros then sets `[:n_train] = 1.0`; `RolloutStorage` stores it; `FrontRESUnified` uses it for advantage/value/actor normalization. | The split-env FrontRES rows are still identified explicitly. | It does not prove the Stage 3 dedicated Segment route consumes this legacy runner mask. |
| Stage 2 currently does not use generic PPO loss for its active objective. | code-confirmed | `FrontRESUnified` sets surrogate/value loss to zero when `frontres_training_objective == "supervised_restore"`; non-supervised standard FrontRES route is explicitly rejected by the runner. | Generic PPO reachability is the next isolated boundary, not an already-proven active training path. | It does not authorize deleting FrontRESUnified generic PPO before a separate route audit. |
| The actor-gate removal did not regress the offline baseline. | compile/contract-confirmed | `py_compile`, targeted full-6D contract, `git diff --check`, and aggregate suite returned `contract_count=40 failed_count=0 total_marker_count=40`. | Storage tuple and current contracts remain consistent. | This is offline evidence, not live Stage 2 or Stage 3 runtime evidence. |
