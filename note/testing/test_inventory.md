# FEMR Current Test Inventory

Updated: 2026-07-27

## Main Entries

| Test | Tier | Current ownership |
| --- | --- | --- |
| `frontres_segment_all_contract_suite.py` | S0-S3 aggregate | Runs current Segment/observation/checkpoint/PPO/eval/quality-identity contracts. |
| `frontres_policy_quality_manifest_contract.py` | S1 | Immutable Q1 manifest, canonical hash, permutation, schema fail-closed, and checkpoint/sampler isolation. |
| `frontres_policy_quality_q2_report_contract.py` | S1/S2 | Q2 exact 8-motion x 2-seed coverage, per-item zero noise floors and route deltas, inferred shared Repair weight, pre-cost Style+Physics decomposition, failure-owner classification, permutation invariance, identity/role/Gain fail-closed behavior, and separation of technical validity from negative scientific outcomes. |
| `frontres_hsl_v007_s1_contract.py` | S1/S2/S3 | v007 proposal carrier and formal `(1,2)` `928/158/770` actor-only HSL; strict HSL v1 persistence and pre-mutation rejection; bounded-smoke config/telemetry schema; exact normalized 158D reload plus bounded CUDA/CPU 6D proposal tolerance with small-differential acceptance and large-drift rejection; critic/optimizer/Stage-3 HSL isolation. |
| `frontres_hsl_v007_s2_connectivity_contract.py` | S2 | CPU-only fake local-scenario path: q29 -> normalizer -> Stage-1 actor -> current target, then v015 zero-writer -> actual storage -> batch -> zero HSL loss. |
| `frontres_v015_two_role_reset_contract.py` | Step 2A S1 | Deterministic fake reset: Repair/Noisy-only layout, Clean `x_t` physical reset, immutable artifact/q29/C/K/hash carrier across retry, legacy-role rejection, and Step-2B future-route fail-closed boundary. |
| `frontres_hsl_rollout_target_contract.py` | S1 | Regression: retired Stage-3 rollout-derived HSL label fails before transition write. |
| `frontres_policy_quality_hsl_magnitude_audit_contract.py` | S1 | Q2-C supervised component value/gradient separation, near-zero cosine-gradient dominance, and fail-closed checkpoint-lineage schema. |
| `frontres_policy_quality_q2d_contract.py` | S1 | Q2-D sorted action scales, identical state restore, transaction-complete credit schema, row/finite fail-closed behavior, Gaussian score-gradient sign, canonical Segment PPO cloned one-update mean projection, and source-policy immutability. |
| `frontres_policy_quality_q2d_wiring_contract.py` | S2 | Dedicated Q2-D CLI/runner/evaluator reaches shared reset/observation/action/rollout/Gain/execution owners; the official Stage 3 update exposes the optional credit-result path without modifying existing evaluator control flow. |
| `frontres_segment_live_single_update_contract.py` | S2 | Official Stage 3 storage-to-PPO owner writes one transaction-complete pre-update raw/bounded action, old stats, Gain, return, advantage, valid-mask and segment tuple without mutating storage, then preserves existing PPO/trust behavior. |
| `frontres_policy_quality_state_contract.py` | S1/S2 | Zero-policy preroll plus root/joint/origin/lifecycle/role/command-cache/perturber/RNG capture, restore, hash, and mismatch rejection. |
| `frontres_policy_quality_eval_contract.py` | legacy S2 | Historical matched zero/frozen-HSL/policy evaluator; v007 makes its HSL route fail closed and it is outside the active formal route. |
| `frontres_policy_quality_entrypoint_contract.py` | S0/S2 | Dedicated CLI/MODE/lazy runner dispatch, required-path validation, full-resume rejection, and zero calls into old eval/training branches. |
| `frontres_policy_quality_executor_contract.py` | S1/S2 | Formal manifest iteration, six named owner callbacks, Q1-C route reuse, atomic result artifact, and optimizer/sampler/warmup isolation. |
| `frontres_policy_quality_real_owner_wiring_contract.py` | legacy S2 | Historical six-owner evaluator wiring; its HSL route is now reject-only and is outside v015 H1-S1a scope. |
| `frontres_segment_stage1_env_hooks_contract.py` | S1/S2 | Canonical index reset and quartet lifecycle, plus source-level proof that one sampled policy perturbation realization is copied to noisy/base while clean is restored unperturbed. |
| `frontres_policy_quality_q1f_input_contract.py` | S1 | Q1-F single-item manifest identity, motion/frame/family/strength/K/seed freeze, comparison signatures, checkpoint-path review surface, and explicit server-hash blocker. |
| `frontres_policy_quality_atlas_contract.py` | S0/S1 | Eight causal quality owners and source links; QUALITY-ID-01 additionally owns B4 reset-to-route role identity instrumentation. |
| `frontres_segment_stage3_pseudo_suite.py` | S2 | Cheap formal Stage 3 route. |
| `frontres_full6_no_active_mask_contract.py` | S0 | Rejects action-mask reintroduction on formal full-6D paths. |
| `frontres_task_space_correction_contract.py` | S1 | Per-row contact-consistent XY scaling and dynamic `dz` lower/upper bounds. |
| `frontres_observation_layout_contract.py` | S1/S3 | 100D prefix + 770D GMT suffix + checkpoint stats. |
| `frontres_balance_obs_cfg_contract.py` | S0/S1 | Balance/ZMP observation config. |
| `frontres_balance_offline_connectivity_contract.py` | S2 | Balance observation reaches FrontRES actor path. |
| `frontres_contact_wrench_zmp_contract.py` | S1/S2 | Contact-wrench ZMP adapter and formal one-action-K connector: exact ground-filter identity, pre-reset raw-view lifecycle, variable per-foot contact slots (`C_left=10`, `C_right=3`) padded to a masked common axis without changing evidence, finite golden ZMP, foot/row permutation, missing-evidence fail-closed, and offline formal-route order. Real PhysX contact population after the padding repair remains S4. |
| `frontres_segment_cache_builder_contract.py` | S1/S2 | Stage 1 cache construction and resume semantics. |
| `frontres_segment_sampler_contract.py` | S1 / Step 2-S1a | Priority, state, legacy trial planning, 8/16/32/64 horizons, and a pure multi-Segment all-policy transaction layout with `M_s >= 2`; it does not prove a frozen parameter snapshot or runner/storage connectivity. |
| `frontres_fixed_noisy_segment_lifecycle_contract.py` | S1 | One immutable Noisy sequence per `source_index`, M-row hash reuse, external-mutation isolation, closure/no-rematerialization, K + H coverage, and Clean-payload rejection; it does not prove command/actor routing. |
| `frontres_segment_motion_command_reference_contract.py` | S1/S2 | Canonical 65D tape materialization/install, fixed current reference and K cursor, read-only H context, and rejection of random/static mixed reference fallback. |
| `frontres_fixed_noisy_actor_context_contract.py` | S1/S2 | `[B, |H|*65]` Noisy H context is joined to the real actor input and old actor layouts fail closed. |
| `frontres_segment_live_sampler_contract.py` | S1/S2 | Formal sampling, fixed-tape/hash reuse, S1b real-policy fingerprint binding, and real sealed-metadata delivery into the S2 accumulator and candidate PPO adapter; rejects a foreign snapshot ID and in-place policy mutation. |
| `frontres_segment_live_probe_contract.py` | S1/S2 | Fixed-tape and S1b metadata reach Clean reset/storage/candidate PPO adapter; S2 accepts one complete multi-Segment/M-policy carrier, rejects early/mixed/non-policy input, and proves one actual fixture optimizer step while not invoking the PPO loss. |
| `frontres_segment_live_probe_ppo_contract.py` | S1/S2 | Policy/search row eligibility before PPO. |
| `frontres_segment_storage_contract.py` | S1/S2 | Full-6D PPO tuple and per-row K returns; storage preserves immutable S1b metadata and explicit row identity at the candidate PPO adapter. |
| `frontres_segment_algorithm_contract.py` | S1/S2 | PPO formula, KL, ratio, detach, permutation, full-6D gradients. |
| `frontres_segment_grouped_ppo_contract.py` | Step 3 S1/S2 offline | Hand-computed nested motion/Segment/attempt/valid-step reduction, row permutation, sign-preserving group scale, no-amplification, missing/misaligned/partial metadata failure, static loss isolation, and storage-to-loss metadata delivery. |
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

## Historical v013 Acceptance Surface

These are superseded fixed-65D-tape acceptance owners. They remain reusable
regression assets but cannot prove v015 semantics.

| Acceptance test | Tier | Required proof |
| --- | --- | --- |
| `frontres_segment_motion_command_reference_contract.py` | S1/S2 | Step 1A legacy provenance mismatch plus Step 1B-S2 canonical 65D materialization/install/current/H/K route. |
| `frontres_fixed_noisy_actor_context_contract.py` | S1/S2 | Ordered H layout, actor dimension/normalizer identity, Noisy segment hash, Clean-input exclusion, H/K separation, and legacy-layout fail-closed behavior. |
| `frontres_double_segment_transaction_contract.py` | S1/S2 | Multiple selected Segments, M >= 2 policy-sampled attempts, fixed x_t/fixed Noisy scenario, one pi_old, no optimizer during collection, and exactly one post-transaction step. |
| `frontres_segment_grouped_ppo_contract.py` | S1/S2 | Motion -> Segment -> attempt -> valid-step reduction, grouped scale-only sign preservation, permutation/M/K invariance, and loss-weight isolation. |
| `frontres_v013_formal_route_contract.py` | S0/S2/S3 | Config-to-runner route, legacy first-policy/immediate-update isolation, future-layout checkpoint/resume identity, and required diagnostics. |
| `frontres_v013_live_sentinel_contract.py` | S4 | One real transaction with reset/reference/snapshot hashes, M attempts, one update, grouped mass shares, frozen GMT, and full-6D action identity. |

## v015 Acceptance Surface

| Test | Tier | Required proof |
| --- | --- | --- |
| `frontres_local_scenario_intent_contract.py` | S1 | Root-only perturbation preserves q29, Noisy-provenance H layout, immutable x_t/artifact/I/C/K identity, and Clean-input exclusion. |
| `frontres_v015_two_role_reset_contract.py` | S1 | Repair/Noisy-only reset layout, shared sealed local command carrier and Clean-dynamics-only `x_t` reset. |
| `frontres_v015_one_action_k_contract.py` | S1/S2 | Two scored roles, one t FEMR action/statistics tuple, frozen later FEMR, command-owned Clean C q29/dq29/root, exact K masks, and immutable M-attempt re-arm in a fake reset-to-capture chain. |
| `frontres_intent_physics_gain_contract.py` | S1 complete, `E-FI-10` | Root-invariant q29 intent fidelity, no-op protection, paired Physics/Cost, explicit UNCONFIRMED optionals, and rejection of Clean-global/Repair-vs-Noisy targets. |
| `frontres_v015_gain_consumer_contract.py` | S1 complete, `E-FI-11`/`E-FI-13` | Candidate-only post-t q29/I[t] -> v003 Gain -> one-row return/advantage plus actual K evidence-step count and immutable scenario-keyed priority evidence; invalid rows and v002/legacy fallback fail closed. |
| `frontres_v015_evaluation_isolation_contract.py` | S1 complete, `E-FI-12` | Sealed v003 candidate carrier -> q29 intent/physics/cost/total local-K report; missing valid rows remain NaN rather than zero; legacy v002 evaluators reject v015 before capture; deployment-composition protocol has no return/priority/PPO feedback. |
| `frontres_v015_grouped_candidate_adapter_contract.py` | S1 complete, `E-FI-13` | Sealed v003 carrier -> immutable local transaction metadata -> complete grouped candidate batch; one Repair row, scenario/hash/x_t/q29/K evidence identity, permutation/K-metadata mass invariance, and legacy/partial/mixed fail-closed. |
| `frontres_v015_transaction_route_contract.py` | S2 complete, `E-FI-14`/`E-FI-15` | CPU fake `2 Segment x 2 attempt` sealed transaction reaches one-row v003 grouped PPO and exactly one optimizer update; it rejects legacy/HSL/partial paths and proves the in-flight checkpoint barrier, but not generic checkpoint cadence/resume. |
| `frontres_v015_checkpoint_resume_contract.py` | S3 complete, `E-FI-15` | CPU fake checkpoint save/load preserves exact q29 H/prefix-normalizer/grouped identity, rejects old/mismatched/tampered layouts before mutation, and allows only idle or committed-receipt transaction state; a valid v015 Stage-3 envelope may retain completed-HSL history, but legacy HSL remains reject-only. |
| `frontres_v015_deployment_composition_s1_contract.py` | S1 complete, `E-FI-28` | Structured `.npz` schema, content identity, canonical persistent-corruption protocol hash, per-frame report, no-feedback type boundary, and legacy/mixed-config rejection. |
| `frontres_v015_deployment_carrier_s2a_contract.py` | S1/S2 complete, `E-FI-29` | Sealed request -> immutable command q29/dq29 sequence -> current `[B,58]` plus dense H `[B,H+1,29]`; frame/cursor/order/identity/provenance/row alignment, no clamp/mixed reference, and actor/GMT/training isolation. |
| `frontres_v015_deployment_composition_s2b_contract.py` | S2 complete, `E-FI-30` | Pre-materialized deployment request -> `T-max(H)` current/H frames -> 928D -> one 6D FEMR action -> frozen GMT -> metrics -> atomic JSON; formal runner isolation and unchanged optimizer/sampler/storage/transition fingerprints. |
| `frontres_v015_deployment_live_cli_s4s0_contract.py` | S2 interface-only, `E-FI-31`; implemented-not-runnable at `E-FI-32` | Absolute checkpoint/file/report and CUDA dispatch isolation. It does not provide the missing trained checkpoint, controlled materializer, paired baseline, or S4 readiness. |
| Planned controlled-carrier contract | S1/S2 missing, G4 | Ordinary Clean/reference `.npz` plus fixed protocol materializes once, seals source/protocol/carrier hashes, exposes no metadata to actor, and never resamples across branches. |
| Planned paired composition contract | S1/S2 missing, G6 | Same carrier/reset/GMT identity reaches No-FEMR baseline and FEMR repair branches; report stays no-feedback. |
| Planned bounded deployment-composition sentinel | S4 blocked, G7 | Trained/reloaded v015 checkpoint plus fixed carrier and paired physical per-frame telemetry; blocked behind G1--G6. |
| `frontres_v015_live_sentinel_contract.py` | S4 | One local transaction prints x_t/artifact/I/C/K/action count/hash, grouped mass, and frozen GMT. |

## Historical Gain v002 Tests

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
