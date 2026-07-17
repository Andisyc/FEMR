# FEMR Current Test Inventory

Updated: 2026-07-17

## Main Entries

| Test | Tier | Current ownership |
| --- | --- | --- |
| `frontres_segment_all_contract_suite.py` | S0-S3 aggregate | Runs current Segment/observation/checkpoint/PPO/eval/quality-identity contracts. |
| `frontres_policy_quality_manifest_contract.py` | S1 | Immutable Q1 manifest, canonical hash, permutation, schema fail-closed, and checkpoint/sampler isolation. |
| `frontres_policy_quality_q2_report_contract.py` | S1/S2 | Q2 exact 8-motion x 2-seed coverage, per-item zero noise floors and route deltas, inferred shared Repair weight, pre-cost Style+Physics decomposition, failure-owner classification, permutation invariance, identity/role/Gain fail-closed behavior, and separation of technical validity from negative scientific outcomes. |
| `frontres_hsl_rollout_target_contract.py` | S1 | Canonical post-step HSL target math, difficulty/no-op weighting, full-env shape, training transition write, and non-mutating dedicated quality-audit mode. |
| `frontres_policy_quality_hsl_magnitude_audit_contract.py` | S1 | Q2-C supervised component value/gradient separation, near-zero cosine-gradient dominance, and fail-closed checkpoint-lineage schema. |
| `frontres_policy_quality_q2d_contract.py` | S1 | Q2-D sorted action scales, identical state restore, training-state isolation, Gaussian score-gradient sign, canonical Segment PPO cloned one-update mean projection, and source-policy immutability. |
| `frontres_policy_quality_q2d_wiring_contract.py` | S2 | Dedicated Q2-D CLI/runner/evaluator reaches shared reset/observation/action/rollout/Gain/execution owners without installing or modifying the existing quality evaluator control flow. |
| `frontres_policy_quality_state_contract.py` | S1/S2 | Zero-policy preroll plus root/joint/origin/lifecycle/role/command-cache/perturber/RNG capture, restore, hash, and mismatch rejection. |
| `frontres_policy_quality_eval_contract.py` | S2 | Matched zero/frozen-HSL/policy action sources, explicit observation/normalizer identity, canonical owner callbacks, and training-state isolation. |
| `frontres_policy_quality_entrypoint_contract.py` | S0/S2 | Dedicated CLI/MODE/lazy runner dispatch, required-path validation, full-resume rejection, and zero calls into old eval/training branches. |
| `frontres_policy_quality_executor_contract.py` | S1/S2 | Formal manifest iteration, six named owner callbacks, Q1-C route reuse, atomic result artifact, and optimizer/sampler/warmup isolation. |
| `frontres_policy_quality_real_owner_wiring_contract.py` | S2 | Official entry reaches all six owners, preserves training state, enforces Gain layouts, proves the zero-action oracle, validates role identity, and persists exactly K canonical HSL target/weight/alignment steps only on the HSL route. |
| `frontres_segment_stage1_env_hooks_contract.py` | S1/S2 | Canonical index reset and quartet lifecycle, plus source-level proof that one sampled policy perturbation realization is copied to noisy/base while clean is restored unperturbed. |
| `frontres_policy_quality_q1f_input_contract.py` | S1 | Q1-F single-item manifest identity, motion/frame/family/strength/K/seed freeze, comparison signatures, checkpoint-path review surface, and explicit server-hash blocker. |
| `frontres_policy_quality_atlas_contract.py` | S0/S1 | Eight causal quality owners and source links; QUALITY-ID-01 additionally owns B4 reset-to-route role identity instrumentation. |
| `frontres_segment_stage3_pseudo_suite.py` | S2 | Cheap formal Stage 3 route. |
| `frontres_full6_no_active_mask_contract.py` | S0 | Rejects action-mask reintroduction on formal full-6D paths. |
| `frontres_task_space_correction_contract.py` | S1 | Per-row contact-consistent XY scaling and dynamic `dz` lower/upper bounds. |
| `frontres_observation_layout_contract.py` | S1/S3 | 100D prefix + 770D GMT suffix + checkpoint stats. |
| `frontres_balance_obs_cfg_contract.py` | S0/S1 | Balance/ZMP observation config. |
| `frontres_balance_offline_connectivity_contract.py` | S2 | Balance observation reaches FrontRES actor path. |
| `frontres_segment_cache_builder_contract.py` | S1/S2 | Stage 1 cache construction and resume semantics. |
| `frontres_segment_sampler_contract.py` | S1 | Priority, state, trial planning, and 8/16/32/64 horizons. |
| `frontres_segment_live_sampler_contract.py` | S1/S2 | Formal sampling, quartet budgeting, metadata, and evidence isolation. |
| `frontres_segment_live_probe_contract.py` | S1/S2 | Reset, mixed-K rollout, legacy score boundary, and storage write; active route migration is covered by Step 6 tests. |
| `frontres_segment_live_probe_ppo_contract.py` | S1/S2 | Policy/search row eligibility before PPO. |
| `frontres_segment_storage_contract.py` | S1/S2 | Full-6D PPO tuple and per-row K returns. |
| `frontres_segment_algorithm_contract.py` | S1/S2 | PPO formula, KL, ratio, detach, permutation, full-6D gradients. |
| `frontres_actual_policy_distribution_contract.py` | S1 | Actual actor raw Gaussian mean, one bounded transform, inverse/Jacobian log-prob identity. |
| `frontres_segment_live_single_update_contract.py` | S2 | Optimizer order, adaptive LR, post-KL, rollback, diagnostics. |
| `frontres_segment_warmup_contract.py` | S1/S2 | DP-09 phase values and actor/critic gradient boundaries. |
| `frontres_frozen_gmt_contract.py` | S1/S2 | GMT freeze, optimizer exclusion, and bitwise no-update boundary. |
| `frontres_formal_runtime_audit_contract.py` | S1/S2 | Phase B flag, stable AUDIT labels, v002 raw/K/quality/step-sum fields, policy-row Gain-step slicing, formal-owner hook connectivity, and silent-off behavior. |
| `frontres_segment_checkpoint_contract.py` | S3 | Detached helper persistence compatibility tests; not the formal `OnPolicyRunner` owner. |
| `frontres_segment_live_sampler_contract.py` | S3 | Formal `frontres_checkpointing.py` save/load, sampler persistence, Gain identity match/mismatch/missing-resume rejection. |
| `frontres_segment_sequence_eval_contract.py` | S2 | Sequence grouping, reset/preroll/eval boundary, aggregation. |
| `frontres_segment_live_training_pseudo_contract.py` | S2 | Periodic eval fresh sampling and state isolation. |
| `frontres_segment_diagnostics_contract.py` | S1/S2 | Motion quality, K masks, saturation, canonical Gain decomposition, legacy-score isolation, and `UNCONFIRMED`. |
| `frontres_segment_live_sentinel_contract.py` | S4 boundary | Minimal real-runtime contract entry. |

## Gain v002 Tests

| Needed test | Tier | Proves |
| --- | --- | --- |
| `frontres_gain_components_contract.py` | S1 | Hand-computed Style, Physics, Repair components, full-6D cost, mixed-K/done masking, K=1/4/8 survival quality, per-step-to-final aggregation, missing-K fail-closed behavior, Clean no-op, signs, scales, and missing-evidence behavior. |
| `frontres_segment_motion_quality_capture_contract.py` | S1/S2 | Clean/Noisy/Repaired root-quaternion role pairing, origin-safe motion capture, and quartet contact-proxy pairing. |
| `frontres_segment_gain_connectivity_contract.py` | S1/S2 | Paired Gain replaces legacy score in storage rewards, K-step returns, valid policy rows, and PPO batch conversion; missing formal evidence rejects fallback; rollout transaction identity reaches the storage batch and mixed-transaction rows are rejected. |
| `frontres_segment_live_update_loop_contract.py` | S2 | Update-loop diagnostics classify one transaction as `single` and multiple transactions as `aggregate`. |
| Extend `frontres_segment_live_probe_contract.py` | S1/S2 | Clean-target pairing, mixed-K decomposition, generic-reward isolation. |
| Extend `frontres_segment_storage_contract.py` | S1/S2 | Accepted total Gain reaches per-row returns. |
| `frontres_segment_live_sampler_contract.py` | S1/S2 | Formal Gain source/component propagation, missing/non-finite Gain rejection, legacy-score isolation at evidence boundary, priority/state diagnostic isolation, and single-owner connectivity. |
| Extend periodic/sequence eval contracts | S2 | Periodic fresh-batch/state isolation plus sequence item/per-motion/aggregate canonical Gain routing are covered; S4 remains open. |
| Extend diagnostics contract | S1/S2 | Raw values and separate gains print; missing values are `UNCONFIRMED`. |
| Short live Gain sentinel | S4 | Real components are populated, diverse, and non-stale. |

## Remaining Gain Gaps

- Real runtime population of root-orientation, ZMP/support, and contact
  components remains an S4 question; offline capture and Physics Gain wiring
  are covered.
- Executed-action Repair Cost and Clean no-op diagnostics are covered by S1/S2
  tests; paired-K aggregation and storage connector are covered offline.
- Step 6A/6B evidence-owner and sampler decision migration are covered offline
  by E11/E12; periodic eval migration is covered by E15; sequence and final
  cross-consumer offline closure are covered by E16.
- S4 live Gain sentinel is still required.
- Formal runtime identity equality for Cards 15/16/17 and aggregate
classification for Card 22 remain an S4 evidence gap; local identity
propagation and mixed-transaction rejection are covered offline.
- Formal S3 checkpoint contracts pass offline; a real server checkpoint load and
  post-resume eval remain unconfirmed.
- Isolated retired authority-space/event modules and their dedicated tests were
  removed in Step 10C-A; connected generic-runner legacy paths remain pending
  separate migration.
- Step 7 diagnostic routing is offline-covered; raw ZMP/contact decomposition
  and non-stale live population remain unconfirmed.

## Remaining General Gaps

- Export/play normalizer sink coverage is incomplete.
- Full-6D action identity still needs current S4 log proof.
- Real dynamic reset and mixed-K execution remain S4 evidence.
