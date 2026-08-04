# FEMR Current Test Inventory

Updated: 2026-08-04

## Phase B Instrumentation Closure E-FI-121

- `frontres_formal_runtime_audit_contract.py` independently exercises
  `AUDIT-B01` through `AUDIT-B08`, including invalid later-FEMR action, mixed
  scenario identity and malformed checkpoint curriculum rejection;
- the production checks read existing one-action-K evidence, immutable final
  telemetry and checkpoint payloads. They do not recompute Gain or feed state
  back into training;
- launch, one-action-K, sealed transaction and checkpoint-v9 focused
  regressions pass. Simulator/live facts remain pending.

## DP10 Future Motion Context Offline Readiness E-FI-119

- `frontres_future_intent_actor_context_contract.py` fixes the only valid
  future offsets to `(1,2)` and rejects `(1,3)` under the active layout version;
- TEST-04/05/10/16 focused contracts pass for role alignment, 928/158/770
  authority, direct policy distribution and checkpoint-v9 persistence;
- `frontres_v015_unmocked_observation_connectivity_contract.py` uses the real
  observation reader and proves `870D + 58D -> 928D -> 158D / 770D` with one
  actor call; `frontres_v015_transaction_route_contract.py` independently
  proves the current formal exact-one transaction;
- the final deterministic aggregate reports `49/49`.

This is offline Phase A evidence only. It does not prove simulator, training,
live, policy-quality or deployment behavior.

## DP09 Actor And Critic Warmup Offline Readiness E-FI-118

- `frontres_segment_warmup_contract.py`: formal phase identity is exactly
  `critic_only -> actor_ramp -> joint`, with unchanged linear weights and K/M/DR
  durations.
- `frontres_v015_transaction_route_contract.py`: K16/M3 and K32/M4 reuse one
  Critic and update it; seeded Actor/std parameters and Adam state remain exact
  in critic-only. Actor-ramp identity reaches final transaction telemetry.
- `frontres_v015_checkpoint_resume_contract.py`: checkpoint-v9 stores and
  reloads `actor_ramp`; old `actor_warmup` rejects before mutation.
- `frontres_interface_refactor_contract.py`: typed request accepts
  `actor_ramp` and rejects the retired phase label.
- The 49-target deterministic aggregate passes. This is offline Phase A
  evidence, not simulator/live or policy-quality evidence.

## DP07 Repair Gain Diagnostics Projection E-FI-116

TEST-18 was rerun after extending the existing immutable local report and
formal telemetry serializer. The exact FRS-GAIN-v007 decomposition, fixed
scale/beta identity and semantic phase-ZMP N/A now survive the official fake
transaction row permutation unchanged. Gain is not recomputed, missing
required Contact/Intent evidence rejects, and diagnostics do not mutate
training state. The focused Gain/step1/transaction contracts and the 49-target
aggregate pass. This is offline Phase A evidence, not simulator or quality
evidence.

## TRAIN-v014 Direct Full-6D Closure E-FI-114

The active HSL/Stage-3 action is one finite direct `[B,6]` value. Focused
HSL/action/log-prob/storage/one-action-K/checkpoint contracts and the 49-target
aggregate pass. HSL-v2/checkpoint-v9 are strict; HSL-v1/checkpoint-v8 and
legacy 12D slicing reject before mutation. This remains offline evidence and
does not authorize Phase B or live execution.

## Human-Confirmed TRAIN-v013 Module Test Closure E-FI-110

All 18 confirmed Module Test Cards were executed at S0/S1 plus module-owned
S3. Result: `18 passed / 0 partial / 0 blocked`. The card questions and
independent answers were unchanged. TEST-10 exposed one executable-test
translation error: three assertions expected the superseded
`FRS-TRAIN-v012` error identity while production correctly rejected under
`FRS-TRAIN-v013`. Only those expected identity strings were corrected; the
full HSL fixture then passed.

This closure did not run the 49-target aggregate, Formal Runtime Audit Phase A
or B, simulator, training, live execution, policy-quality evaluation or
deployment composition. The detailed card-to-fact ledger is
`note/testing/frontres_module_test_execution_2026-08-03.md`. The next gate is
human review of Formal Runtime Audit Phase A; module PASS is not connectivity
evidence.

## TRAIN-v013 Module-Test Closure E-FI-109

TRAIN-v013 changed the curriculum and persistence oracle. E-FI-105 remains
valid for 15 unchanged cards; TEST-02 Training Config, TEST-06 Perturbation
Data and TEST-16 Checkpointing were updated from the active Contract and rerun.
Current historical E-FI-109 status is `18 passed / 0 partial / 0 blocked`, with
active aggregate `49/49` and a separate Phase A pass. E-FI-110 supersedes only
the module-card execution evidence; it does not erase or re-execute that
separate Phase A record.

The required revised cases cover explicit per-K DRStageSpec and lower-DR
restart, d_cap four-class 20/30/40/10 sampling with no feedback/resample, and
checkpoint-v8 full state with v7/g_K pre-mutation rejection.

## Historical Module Test Atlas Closure E-FI-105

The 18 human-confirmed active-generation cards were rerun independently of the
broad regression suite. Result: `18 passed / 0 partial / 0 blocked`.

- canonical card status: `note/architecture/testing/05_frontres_module_test_atlas.data.json`;
- claim/test/fact/limitation ledger:
  `note/testing/frontres_module_test_execution_2026-08-03.md`;
- that v012 module readiness no longer admits formal-runtime audit after v013 activation.

The active aggregate is secondary regression evidence and reports 49/49. The
retired v002/composite adaptive sampler contract remains a historical file and
is excluded from active aggregation.

## Main Entries

| Test | Tier | Current ownership |
| --- | --- | --- |
| `frontres_segment_all_contract_suite.py` | active S0-S3 aggregate | Secondary 49-target regression runner. Its passing result does not replace the 18 independently answered Module Test Cards. |
| `frontres_gain_v007_contract.py` | S1 | Clean anchor, fixed scales, K evidence, family aggregation, Recovery-Aware pressure, N/A/fail-closed behavior, baseline reuse, permutation and beta-cost ordering. |
| `frontres_v017_step1_contract.py` | S1/S2 | Sealed Clean/Noisy/M-Repair evidence -> v007 Gain -> grouped scalar PPO-v005 -> atomic local report, plus active legacy-local-evaluator rejection. |
| `frontres_v015_checkpoint_resume_contract.py` | S3 | Real temporary checkpoint-v9 atomic roundtrip with direct-action plus full DR/RNG/receipt state and pre-mutation v8/v7/`g_K`/mixed/partial/tamper rejection. |
| `frontres_policy_quality_manifest_contract.py` | S1 | Immutable Q1 manifest, canonical hash, permutation, schema fail-closed, and checkpoint/sampler isolation. |
| `frontres_policy_quality_q2_report_contract.py` | S1/S2 | Q2 exact 8-motion x 2-seed coverage, per-item zero noise floors and route deltas, inferred shared Repair weight, pre-cost Style+Physics decomposition, failure-owner classification, permutation invariance, identity/role/Gain fail-closed behavior, and separation of technical validity from negative scientific outcomes. |
| `frontres_hsl_v007_s1_contract.py` | S1/S2/S3 | v007 proposal carrier and formal `(1,2)` `928/158/770` actor-only HSL; exact current anti-DR full-6D target preserves both `dz` signs without axis clamp; strict HSL-v2 direct-6D persistence and HSL-v1 pre-mutation rejection; exact normalized 158D reload plus CUDA/CPU numerical tolerance; critic/optimizer/Stage-3 HSL isolation. |
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
| `frontres_task_space_correction_contract.py` | active S1 | Hand-computed zero/dx/droll world-frame current-only correction, future immutability, row permutation, no scaling/clamping, and non-finite pre-mutation rejection. |
| `frontres_observation_layout_contract.py` | S1/S3 | 100D prefix + 770D GMT suffix + checkpoint stats. |
| `frontres_balance_obs_cfg_contract.py` | S0/S1 | Balance/ZMP observation config. |
| `frontres_balance_offline_connectivity_contract.py` | S2 | Balance observation reaches FrontRES actor path. |
| `frontres_contact_wrench_zmp_contract.py` | S1/S2; S4 at E-FI-84; capacity live-confirmed at E-FI-86 | Contact-wrench ZMP adapter and formal one-action-K connector: exact ground-filter identity, pre-reset raw-view lifecycle, 256 raw contacts per foot/env (`2048` at 8 env), exact-saturation fail-closed, variable per-foot contact slots padded to a masked common axis without changing evidence, finite golden ZMP, foot/row permutation, missing-evidence fail-closed, and live-confirmed real PhysX population through iteration 2000. |
| `frontres_segment_cache_builder_contract.py` | S1/S2 | Stage 1 cache construction and resume semantics. |
| `frontres_segment_sampler_contract.py` | S1 / Step 2-S1a | Priority/state compatibility plus the active sealed TRAIN-v014 exact-K/exact-M layout; the active path must not consult the legacy state-driven budget. It does not prove live runner/storage connectivity. |
| `frontres_fixed_noisy_segment_lifecycle_contract.py` | S1 | One immutable Noisy sequence per `source_index`, M-row hash reuse, external-mutation isolation, closure/no-rematerialization, K + H coverage, and Clean-payload rejection; it does not prove command/actor routing. |
| `frontres_segment_motion_command_reference_contract.py` | S1/S2 | Canonical 65D tape materialization/install, fixed current reference and K cursor, read-only H context, and rejection of random/static mixed reference fallback. |
| `frontres_fixed_noisy_actor_context_contract.py` | S1/S2 | `[B, |H|*65]` Noisy H context is joined to the real actor input and old actor layouts fail closed. |
| `frontres_segment_live_sampler_contract.py` | historical v002 S1/S2 | Characterizes retired adaptive search roles, v002 Gain evidence, and composite perturbation. Active v017 correctly rejects this route; it is not part of the active aggregate. |
| `frontres_segment_live_probe_contract.py` | S1/S2 | Characterizes the legacy facade while exercising extracted reset/storage/one-action-K/rollout/reporting owners: fixed-tape and S1b metadata reach Clean reset/storage/candidate PPO adapter; complete multi-Segment/M-policy carrier accepts and early/mixed/non-policy input rejects. |
| `frontres_segment_live_probe_ppo_contract.py` | S1/S2 | Policy/search row eligibility before PPO. |
| `frontres_segment_storage_contract.py` | S1/S2 | Full-6D PPO tuple and per-row K returns; storage preserves immutable S1b metadata and explicit row identity at the candidate PPO adapter. |
| `frontres_segment_algorithm_contract.py` | S1/S2 | PPO formula, KL, ratio, detach, permutation, full-6D gradients. |
| `frontres_segment_grouped_ppo_contract.py` | Step 3 S1/S2 offline | Hand-computed nested motion/Segment/attempt/valid-step reduction, row permutation, sign-preserving group scale, no-amplification, missing/misaligned/partial metadata failure, static loss isolation, and storage-to-loss metadata delivery. |
| `frontres_actual_policy_distribution_contract.py` | S1 | Actual actor raw Gaussian mean/sample, direct 6D action identity, ordinary Normal log-prob and zero action-transform Jacobian. |
| `frontres_segment_live_single_update_contract.py` | S2 | Optimizer order, adaptive LR, post-KL, rollback, diagnostics. |
| `frontres_segment_warmup_contract.py` | S1/S2 | DP-09 phase values and actor/critic gradient boundaries. |
| `frontres_frozen_gmt_contract.py` | S1/S2 | GMT freeze, optimizer exclusion, and bitwise no-update boundary. |
| `frontres_formal_runtime_audit_contract.py` | S1/S2 | Phase B flag, active v017/v007/v005/v014 two-Segment x exact-M voting weights, exact-one update, checkpoint-v9 direct-action identity, formal-owner hook connectivity, active K isolation, invalid identity/mass rejection and silent-off behavior. |
| `frontres_segment_checkpoint_contract.py` | S3 | Detached helper persistence compatibility tests; not the formal `OnPolicyRunner` owner. |
| `frontres_segment_live_sampler_contract.py` | historical v002 S3 | Historical sampler persistence and retired Gain identity characterization; excluded from active v017 aggregation. |
| `frontres_segment_live_training_pseudo_contract.py` | S2 | Training-loop diagnostics/checkpoint behavior and proof that no evaluator is embedded in training. |
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
| `frontres_intent_physics_gain_contract.py` | S1 complete, `E-FI-85` current | Root-invariant q29 Intent objective, independent Contact/phase-ZMP/survival constraints, role-specific loaded-support applicability, no-op protection, explicit UNCONFIRMED optionals, and rejection of Clean-global targets. |
| `frontres_v015_gain_consumer_contract.py` | S1 complete, `E-FI-11`/`E-FI-13` | Candidate-only post-t q29/I[t] -> v003 Gain -> one-row return/advantage plus actual K evidence-step count and immutable scenario-keyed priority evidence; invalid rows and v002/legacy fallback fail closed. |
| `frontres_v015_evaluation_isolation_contract.py` | S1 complete, `E-FI-12` | Sealed v003 candidate carrier -> q29 intent/physics/cost/total local-K report; missing valid rows remain NaN rather than zero; legacy v002 evaluators reject v015 before capture; deployment-composition protocol has no return/priority/PPO feedback. |
| `frontres_v015_grouped_candidate_adapter_contract.py` | S1/S2 current, `E-FI-85` | Sealed v006 carrier -> immutable local transaction metadata -> grouped candidate batch; explicit Repair/Noisy applicability covers all four combinations, paired ZMP is finite iff both apply, PPO consumes Repair only, and legacy/partial/mixed rows fail closed. |
| `frontres_interface_refactor_contract.py` | S1/S2 current, `E-FI-93`/`E-FI-94`/`E-FI-95`/`E-FI-96` | Typed identity/928-158-770/2-Segment x M ports and unique Stage-3 engine; structurally rejects semantic bodies in compatibility facades, production file-path loading and hidden dependency cycles, and verifies dedicated scenario, evidence/storage, projection/diagnostic, transaction, telemetry, persistence and evaluator owners. |
| `frontres_contract_imports.py` | contract infrastructure, `E-FI-93` | Supplies lightweight real package paths to isolated contracts so normal owner imports are tested without importing the simulator-facing runner facade. |
| `frontres_v015_transaction_route_contract.py` | combined S1/S2 regression, current at `E-FI-110` | CPU fake sealed v017 transaction preserves two Segment sources, exact-M Repair rows, scalar GAIN-v007/grouped PPO-v005, one optimizer step and one committed receipt; partial/mixed/legacy/HSL paths reject. E-FI-110 uses only its module assertions and does not treat this combined file as formal-route proof. |
| `frontres_v015_checkpoint_resume_contract.py` | S3 current, `E-FI-114` | Temporary checkpoint-v9 roundtrip preserves TRAIN-v014 direct action, K/M/DR stage, d_cap/progress, q29 H/prefix-normalizer, optimizer/sampler/RNG and committed receipt; v8/v7, g_K, mixed, partial and tampered payloads reject before mutation. |
| `frontres_v015_deployment_composition_s1_contract.py` | S1 complete, `E-FI-28`, `E-FI-87`, `E-FI-98` | Structured Clean-source/Noisy-carrier identity, canonical persistent-corruption hash, same-state Baseline/Repair report, exact-zero Baseline action, row-aligned Contact/phase-ZMP/survival/lean, N/A/applicability fail-closed, no-feedback boundary, and legacy rejection. |
| `frontres_v015_deployment_carrier_s2a_contract.py` | S1/S2 complete, `E-FI-29` | Sealed request -> immutable command q29/dq29 sequence -> current `[B,58]` plus dense H `[B,H+1,29]`; frame/cursor/order/identity/provenance/row alignment, no clamp/mixed reference, and actor/GMT/training isolation. |
| `frontres_v015_deployment_composition_s2b_contract.py` | S2 complete, `E-FI-30`, `E-FI-87`, `E-FI-98` | Canonical route-start/RNG snapshot -> Baseline restore with zero FEMR -> Repair restore with per-frame 6D FEMR -> frozen GMT -> paired atomic JSON through the public runtime Gateway; training fingerprints unchanged. |
| `frontres_v015_deployment_live_cli_s4s0_contract.py` | S2 interface complete, `E-FI-31`, `E-FI-87`; S4 pending | Absolute post-fix checkpoint, Clean source, Noisy carrier, CUDA, frame-0/episode-length config, zero-update dispatch and final quality sentinel are fail-closed; real simulator values remain S4-only. |
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
| `frontres_segment_live_sampler_contract.py` | historical v002 S1/S2 | Historical Gain/source and priority-state characterization; active v017 uses sealed exact-M transaction contracts instead. |
| Extend diagnostics contract | S1/S2 | Raw values and separate gains print; missing values are `UNCONFIRMED`. |
| Short live Gain sentinel | S4 | Real components are populated, diverse, and non-stale. |

## Remaining Gain Gaps

- Real runtime population of root-orientation, ZMP/support, and contact
  components remains an S4 question; offline capture and Physics Gain wiring
  are covered.
- Executed-action Repair Cost and Clean no-op diagnostics are covered by S1/S2
  tests; paired-K aggregation and storage connector are covered offline.
- Step 6A/6B evidence-owner and sampler decision migration are historical v002
  coverage. Their periodic/offline/sequence consumers are retired and cannot
  prove the current three-capability Evaluation boundary.
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
