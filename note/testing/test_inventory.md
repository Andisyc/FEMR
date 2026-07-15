# FEMR Current Test Inventory

Updated: 2026-07-15

## Main Entries

| Test | Tier | Current ownership |
| --- | --- | --- |
| `frontres_segment_all_contract_suite.py` | S0-S3 aggregate | Runs 43 current Segment/observation/checkpoint/PPO/eval contracts. |
| `frontres_segment_stage3_pseudo_suite.py` | S2 | Cheap formal Stage 3 route. |
| `frontres_full6_no_active_mask_contract.py` | S0 | Rejects action-mask reintroduction on formal full-6D paths. |
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
| `frontres_formal_runtime_audit_contract.py` | S1/S2 | Phase B flag, stable AUDIT labels, formal-owner hook connectivity, and silent-off behavior. |
| `frontres_segment_checkpoint_contract.py` | S3 | Detached helper persistence compatibility tests; not the formal `OnPolicyRunner` owner. |
| `frontres_segment_live_sampler_contract.py` | S3 | Formal `frontres_checkpointing.py` save/load, sampler persistence, Gain identity match/mismatch/missing-resume rejection. |
| `frontres_segment_sequence_eval_contract.py` | S2 | Sequence grouping, reset/preroll/eval boundary, aggregation. |
| `frontres_segment_live_training_pseudo_contract.py` | S2 | Periodic eval fresh sampling and state isolation. |
| `frontres_segment_diagnostics_contract.py` | S1/S2 | Motion quality, K masks, saturation, canonical Gain decomposition, legacy-score isolation, and `UNCONFIRMED`. |
| `frontres_segment_live_sentinel_contract.py` | S4 boundary | Minimal real-runtime contract entry. |

## Gain v001 Tests

| Needed test | Tier | Proves |
| --- | --- | --- |
| `frontres_gain_components_contract.py` | S1 | Hand-computed Style, Physics, Repair components, full-6D cost, mixed-K/done masking, Clean no-op, signs, scales, and missing-evidence behavior. |
| `frontres_segment_motion_quality_capture_contract.py` | S1/S2 | Clean/Noisy/Repaired root-quaternion role pairing, origin-safe motion capture, and quartet contact-proxy pairing. |
| `frontres_segment_gain_connectivity_contract.py` | S2 | Paired Gain replaces legacy score in storage rewards, K-step returns, valid policy rows, and PPO batch conversion; missing formal evidence rejects fallback. |
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
