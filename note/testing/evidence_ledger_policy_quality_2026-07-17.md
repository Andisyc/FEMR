# FrontRES Policy Quality Evidence Ledger

## Q-E1 - Immutable Comparison Manifest Contract

Date: 2026-07-17

Scope: Q1-A comparison identity only. No runner, environment, checkpoint load,
counterfactual rollout, Gain, or policy-quality claim is included.

Evidence:
- Owner: `source/rsl_rl/rsl_rl/frontres/frontres_policy_quality_manifest.py`.
- Focused contract:
  `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_manifest_contract.py`.
- Focused result: `PASS: immutable policy-quality manifest and comparison
  signatures are closed.`
- Aggregate result: `frontres_segment_all_contract_suite.py` exited
  successfully with the new `policy_quality_manifest` core contract included.

Facts:
- `comparison_signature` is deterministic and independent of manifest row
  order and display-only `item_id`.
- Motion, start frame, perturbation descriptor, effective K, seed,
  environment revision, config revision, and evaluator version alter the
  corresponding comparison identity.
- Duplicate semantic rows, missing fields, unexpected fields, and mutable
  manifest containers fail closed.
- Checkpoint and sampler state are rejected from manifest selection identity.
- Checkpoint identity belongs to route metadata and changes only the route
  signature, not the comparison signature.

Limitations:
- Q-E1 is `Q-formula` plus offline `Q-matched` schema evidence. It does not
  prove simulator scoring-state equality, zero/HSL/policy execution, Gain
  ordering, policy efficacy, or generalization.

Next:
- Q1-B must capture and restore the exact scoring-start dynamic state and bind
  its `initial_state_hash` to this comparison identity.

## Q-E2 - Scoring-State Capture And Restore Contract

Date: 2026-07-17

Scope: Q1-B state isolation using a semantically complete fake environment.
No existing evaluator, real simulator, HSL/policy route, or Gain consumer is
included.

Evidence:
- Owner: `source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py`.
- Focused contract:
  `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_state_contract.py`.
- Focused result: `PASS: policy-quality scoring state capture and restore are
  closed offline.`
- Aggregate result: `frontres_segment_all_contract_suite.py` passed `46/46`
  with `failed_count=0`.

Facts:
- The zero-preroll helper accepts only a step callback and emits exact `(N,6)`
  zero actions; no policy object enters the preroll boundary.
- The snapshot binds root state, joint position/velocity, environment origin,
  episode length, explicit role layout, motion/frame state, perturbed reference cache, FrontRES
  correction cache, every per-environment perturber Tensor, and Python/NumPy/
  Torch RNG state to one comparison signature.
- Mutating those fields and RNG state, then restoring, reproduces the complete
  snapshot and `initial_state_hash` exactly.
- Missing command cache, absent perturber state, duplicate env IDs, and
  comparison-signature mismatch fail closed.

Limitations:
- Q-E2 is offline `Q-matched` evidence. It does not prove that Isaac/PhysX
  articulation state and simulator-side buffers reproduce the hash in a real
  run; that remains the bounded Q1-F live identity sentinel.
- It does not execute zero/HSL/policy counterfactuals or prove policy quality.

Next:
- Q1-C may use this owner to restore the same state before isolated zero, HSL,
  and policy action routes.

## Q-E3 - Isolated Counterfactual Route Contract

Date: 2026-07-17

Scope: Q1-C offline orchestration of zero, frozen-HSL, and tested-policy
routes. This is a callback-connectivity contract, not formal runner wiring or
real policy-quality evidence.

Evidence:
- Owner: `source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py`.
- Focused contract:
  `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_eval_contract.py`.
- Focused result: `PASS: isolated zero/HSL/policy counterfactual execution is
  closed offline.`
- Aggregate result: `frontres_segment_all_contract_suite.py` passed `47/47`
  with `failed_count=0`.

Facts:
- Route order is fixed to `zero -> hsl -> policy`; every route restores the
  same comparison signature and initial-state hash before observation/action.
- Zero emits exact `(N,6)` zeros. HSL and policy load only residual-actor and
  observation-normalizer state from their own checkpoint payloads.
- HSL/policy actor and normalizer copies are inference-only, in eval mode, and
  have no trainable parameters. No supervised target appears in this owner.
- Observation dimension, actor-input dimension, normalizer identity, finite
  full-6D output, and task-space bounds fail closed.
- All three routes invoke one shared action/step/Gain/execution hook surface;
  mutating the supplied optimizer/sampler/warmup isolation state rejects the
  evaluation.

Limitations:
- Q-E3 proves that the isolated evaluator can preserve and call one owner
  surface. Q1-D must still wire that surface to the repository's real
  observation, task-space application, rollout, canonical Gain, and execution
  owners and prove old-mode isolation.
- No real checkpoint file or simulator was executed, and no route is claimed
  to outperform another.

Next:
- Q1-D adds the dedicated `policy_quality_eval` dispatch and thin formal-owner
  connectors without changing old evaluator behavior.

## Q-E4 - Dedicated Entrypoint And Old-Mode Isolation

Date: 2026-07-17

Scope: Q1-D CLI, Stage 3 shell MODE, lazy runner connector, immutable request
validation, and static old-mode isolation. No simulator quality execution is
claimed.

Evidence:
- Focused contract:
  `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_entrypoint_contract.py`.
- Focused result: `PASS: dedicated policy-quality entrypoint and old-mode
  isolation are closed offline.`
- Aggregate result: `frontres_segment_all_contract_suite.py` passed `48/48`
  with `failed_count=0` after rebuilding the shifted Runtime Atlas source
  links.
- Runtime Atlas viewer/data contract passed with 62 owner paths.

Facts:
- `policy_quality_eval` has five dedicated CLI fields: mode, manifest, HSL
  checkpoint, tested-policy checkpoint, and result path.
- The Stage 3 shell mode requires all four paths and forces
  `STAGE3_IS_FULL_RESUME=0`, preventing checkpoint sampler/optimizer/warmup
  restoration into evaluation selection.
- The `OnPolicyRunner` connector lazy-imports the quality owner only when its
  dedicated method is called.
- The quality dispatch is ordered before old offline/sequence/live dispatch,
  rejects conflicting modes, calls one quality runner method, closes the env,
  and returns.
- Request validation rejects absent files, identical HSL/policy checkpoints,
  invalid manifests, and absent result directories.
- A missing formal manifest executor fails closed instead of producing an
  empty or misleading quality result.

Limitations:
- Q-E4 proves dispatch and isolation, not formal manifest execution. Q1-E must
  assemble and test the real observation/action/rollout/Gain/execution owner
  callbacks before any live command is permitted.

Next:
- Q1-E begins with formal executor offline preflight, then creates and checks
  the eight-card Quality Audit Atlas.

## Q-E5 - Formal Executor And Eight-Owner Quality Atlas

Date: 2026-07-17

Scope: Q1-E offline owner preflight, human-readable Atlas, and regression
closure. No simulator rollout or policy-quality comparison is claimed.

Observed evidence:

- `install_frontres_policy_quality_manifest_executor()` consumes immutable
  manifest items in order and delegates each item to the Q1-C
  zero/HSL/policy orchestrator.
- The executor requires exactly six named callbacks: reset, observation,
  action, rollout, Gain, and execution. Missing owner surfaces fail closed.
- Optimizer, sampler, and warmup state are compared before and after execution;
  result JSON is written atomically with comparison and manifest signatures.
- `05_policy_quality_audit.data.json` contains exactly eight causal cards:
  ID, DATA, ACTION, GAIN, CREDIT, UPDATE, EXEC, and TRAJECTORY.
- Every card maps a Concept Figure design point to a falsifiable question,
  failure owner, and source-linked B1/B2/B3 reading boundary.
- Focused executor and Atlas contracts pass; viewer/data import passes; Formal
  Runtime Atlas links were regenerated after source comments moved.
- `frontres_segment_all_contract_suite.py` passes `50/50`, with
  `failed_count=0` and `total_marker_count=50`.

Evidence class: S0-S2 `T-connect/T-link/T-schema/T-isolation/T-regression`.

Limit:

- Q-E5 proves that the independent evaluator is structurally ready for a
  bounded live identity sentinel. It does not prove real Isaac state equality,
  HSL/policy usefulness, Gain correctness, or checkpoint improvement.
- Q1-F remains a user-reviewed S4 live gate and has not been started.

## Q-E6 - Official Entry Real-Owner Wiring Correction

Date: 2026-07-17

Scope: correct the overclaim in Q-E5 and close Q1-E Gate B offline. Q1-F live
execution remains excluded.

Contradicted prior claim:

- Q-E5 called Q1-E complete after testing
  `install_frontres_policy_quality_manifest_executor()` only with a manually
  supplied fake owner bundle. The formal runner/entrypoint never installed that
  bundle and would raise `policy-quality formal owner executor is not
  configured`. Q-E5 therefore proved Gate A only (`implemented-only`).

Observed evidence after repair:

- `run_frontres_policy_quality_eval()` now lazily builds and installs the
  production bundle when the dedicated entry has no executor.
- `frontres_policy_quality_formal_owners.py` independently composes the shared
  lower-level reset, observation normalization, task-space action application,
  frozen-GMT rollout, canonical `compute_segment_gain`, and execution-capture
  owners. It does not call periodic, offline, or sequence evaluator control
  flows.
- The formal reset-support helper installs only cache/index-reset support and
  asserts that Segment sampler identity is unchanged.
- S2 official-entry wiring evidence observes reset once; observation, action,
  rollout, and execution nine times across 3 routes x 3 steps; canonical Gain
  three times; optimizer/sampler/warmup signatures are identical before/after.
- The regression test does not manually install an executor or supply a fake
  formal bundle. A semantic fake env replaces only simulator state/sensor
  behavior; production bundle and canonical Gain code execute.
- Focused entrypoint, counterfactual, executor, real-owner wiring, sampler, and
  Atlas contracts pass. Viewer import passes. Python compile passes.
- `frontres_segment_all_contract_suite.py` passes `51/51`, with
  `failed_count=0` and `total_marker_count=51`.

Evidence class: S2 `T-connect/T-oracle/T-isolation/T-regression`.

Result:

- Q1-E wiring state is `integrated-offline`, not `integrated-live`.
- Q1-F remains blocked pending explicit user authorization. Real Isaac reset,
  state-hash equality, physical execution, and policy quality remain
  unconfirmed.

## Q-E7 - Q1-F Single-Item Input Freeze

Date: 2026-07-17

Scope: freeze reviewable Q1-F inputs without launching Isaac or authorizing
live execution.

Observed evidence:

- Manifest: `note/testing/manifests/frontres_policy_quality_q1f_single_v1.json`.
- HSL baseline: actor-update-free `model_200.pt` from the formal actor-sentinel
  lineage after critic warmup.
- Tested policy: complete `model_701.pt` from the same resumed Stage 3 lineage
  after full-weight actor updates.
- Item identity: KIT/572 wave-right02, frame 163, local_rp, DR scale 1.25,
  K=8, seed 42. These values come from the successful formal reset/run evidence.
- Item signature:
  `206e4c1bd7aec5e987049fa9697b755cef826ed093c3683fb7f057f38e29d2eb`.
- Manifest signature:
  `4c7122e5278c2371d2917659e0ac5944ac1dd8579de94cc99811bdf95dd5eee0`.
- Focused manifest-input contract passes. Aggregate suite passes `52/52`,
  with `failed_count=0` and `total_marker_count=52`.

Evidence class: S1 `T-schema/T-value/T-identity/T-persist-preflight`.

Limit and blocker:

- The local workspace contains neither checkpoint. Their server existence and
  SHA-256 hashes remain `UNCONFIRMED`; paths are evidence-backed but not yet
  artifact-verified.
- Q1-F remains blocked until Dr. Cheng reviews the freeze and server preflight
  records both hashes. No live command has been issued.

## Q-E8 - Q1-F Inference-Tensor Restore Defect

Date: 2026-07-17

Scope: first Q1-F live attempt and deterministic restore-owner repair. The run
did not reach matched route execution and is not Q1 evidence.

Observed facts:

- The canonical index reset reached motion KIT/572 frame 163 with four roles;
  policy-only local_rp strength was 1.25 and the other role strengths were zero.
- The first scoring-state restore failed at `_restore_rows()` because an Isaac
  command cache created under `torch.inference_mode()` rejected an out-of-mode
  in-place `index_copy_`.
- The restore owner now preserves tensor object identity and performs the
  in-place row copy inside `torch.inference_mode()`.
- A regression fixture creates an actual PyTorch inference tensor and proves
  indexed restoration outside the caller's inference context.
- Focused state and real-owner wiring contracts pass; `py_compile` passes;
  aggregate suite remains `52/52`; `git diff --check` passes.

Evidence class: S1/S2 plus failed S4 boundary,
`T-state/T-live/T-regression/T-connect`.

Limit:

- Q1-F remains unconfirmed and must rerun the same immutable manifest. No
  comparison-signature, three-route state-hash, Gain, or policy-quality claim
  follows from the failed run.

## Q-E9 - Q1-F Gain Axis Closure And Pair-Sync Reassessment

Date: 2026-07-17

Scope: inspect the successful Q1-F rerun, repair the deterministic canonical
Gain axis defect, and test the first suspected paired-corruption boundary.

Observed evidence:

- Raw log: `policy_quality_q1f_single_v1.txt`; result:
  `policy_quality_q1f_single_v1_result.json`.
- Manifest signature is
  `4c7122e5278c2371d2917659e0ac5944ac1dd8579de94cc99811bdf95dd5eee0`;
  zero, HSL model_200, and policy model_701 share initial-state hash
  `f171fc08e51881ddf30cb9964c87dae636a673184bd082804d5f96afeaedcd1f`.
- The old Q1 evaluator stacked motion frames as `[T,B,...]`, causing eight
  style/total Gain rows for one paired item. The corrected owner uses
  `[B,T,...]` for style/orientation and preserves `[T,B,6]` for repair cost.
  All four canonical components now serialize as one value per item.
- The live reset trace prints perturbation mask/strength only on policy. Code
  inspection shows this is the random-sampling owner, not proof that noisy is
  clean: `refresh_frontres_reference_cache_current_frame()` calls
  `_sync_frontres_pairs(sync_perturbation=True)`, which copies cached perturbed
  position, quaternion, supervised target, and perturbation states from policy
  to noisy/base, then restores clean to raw motion.
- The proposed change that independently enabled the noisy perturbation mask
  was withdrawn before completion because it would add a second random draw
  and violate the single-realization owner.
- Offline regressions now assert canonical Gain axis semantics, a semantic
  zero-action Gain of zero, and the production policy-to-noisy cache copy plus
  clean reset. Focused contracts pass; aggregate suite passes `52/52`.

Evidence class: Q-formula plus S1/S2
`T-shape/T-metamorphic/T-connect/T-regression` and partial Q-matched live
evidence.

Limit and next boundary:

- The real zero route reports Gain `0.007556` although its 192 action scalars
  are exactly zero. This may be paired-environment numerical/terrain-origin
  divergence; the current artifact does not persist policy/noisy local root,
  joint, cached-position, or cached-quaternion deltas needed to classify it.
- Q1-F remains partial. Do not change reset masks, Gain, or PPO. The next
  bounded live fact is one role-aware identity snapshot after reset and before
  rollout, using local coordinates and synced cache deltas.

## Q-E10 - Q1-F Role Identity Snapshot Preflight

Date: 2026-07-17

Scope: instrument the sole remaining Q1-F simulator-only identity boundary
without changing reset, rollout, Gain, PPO, or any existing evaluator.

Implementation and evidence:

- `frontres_policy_quality_formal_owners.py::_role_identity_snapshot()` reads
  only the immutable scoring-start snapshot captured after reset.
- `prepare_item()` emits one `[QUALITY-ID-01 Role Identity]` block before the
  zero/HSL/policy branch and stores the same mapping as `row.role_identity` in
  the result JSON.
- The snapshot separates world root from `root_state_w - env_origins`, then
  records policy/noisy deltas for local root, root quaternion, linear/angular
  velocity, joint position/velocity, cached perturbed position/quaternion. It
  separately records policy/clean cache deltas as the corruption-presence
  oracle.
- The semantic fake sets policy/noisy world origins 20 m apart while keeping
  local state and cache identical. Observed snapshot reports world/origin delta
  20, all local dynamic/cache deltas 0, and nonzero policy/clean cache deltas.
- `QUALITY-ID-01` Atlas card now contains source-linked B4; its contract allows
  four blocks only for this card and keeps the other seven at three blocks.
- Focused real-owner and Atlas contracts pass.

Evidence class: S1/S2 `T-schema/T-role/T-origin/T-cache/T-connect`.

Limit and next:

- This preflight proves schema, coordinate semantics, insertion order, and
  persistence only. Real Isaac values remain S4-unconfirmed.
- Upload the current code and rerun the unchanged immutable Q1-F command once.
  Stop after reading B4; do not tune or start a broader quality bank first.

## Q-E11 - Q1-F Real Role Identity Closure

Date: 2026-07-17

Scope: classify the real simulator policy/noisy role identity and close or
reject Q1 before any broader policy-quality claim.

Raw evidence:

- Log: `policy_quality_q1f_single_v1.txt`, updated 2026-07-17 18:15.
- Result: `policy_quality_q1f_single_v1_result.json`, updated 2026-07-17 18:15.
- Manifest signature:
  `4c7122e5278c2371d2917659e0ac5944ac1dd8579de94cc99811bdf95dd5eee0`.
- Shared zero/HSL/policy initial-state hash:
  `f171fc08e51881ddf30cb9964c87dae636a673184bd082804d5f96afeaedcd1f`.

Runtime facts:

- Policy/noisy world-root and env-origin max deltas are both 40 m. After origin
  removal, local-root max delta is `9.536743e-7` m.
- Root quaternion, linear/angular velocity, joint position/velocity, cached
  perturbed position, and cached perturbed quaternion deltas are all exactly 0.
- Policy/clean cached position delta is 0 and cached quaternion delta is
  `0.061487824`, matching the active local_rp corruption semantics.
- Zero uses exact zero actions and reports Gain `0.007556424`; this is the
  observed eight-step paired-environment noise floor for this item, not a reset
  or cache identity failure.
- HSL Gain is `0.049282383`; Policy Gain is `0.049358435`. Relative to zero,
  HSL excess is `0.041725960` and Policy excess is `0.041802011`.
  Policy-HSL is only `0.000076052`, below the observed zero noise floor.
- Policy differs from HSL in action space (max element delta `0.0140344`, L2
  delta `0.0465331`) but this item cannot resolve a behavioral improvement.

Evidence class: Q-matched S4 plus single-item Q-causal precursor.

Decision:

- Q1 matched comparison identity is closed.
- This item rejects the no-op hypothesis locally because HSL and Policy both
  exceed zero after accounting for the zero control.
- It does not prove that PPO improves HSL, generalizes, or permits long
  training. Q2 must use at least 8 fixed motions and 2 matched seeds, preserving
  zero/HSL/policy routes and reporting per-item zero noise floors.

## Q-E12 - Q2 Offline Reporter Closure

Date: 2026-07-17

Scope: freeze the governed Q2 bank and close deterministic reporting before
requesting another simulator run.

Implementation and evidence:

- Accepted manifest:
  `note/testing/manifests/frontres_policy_quality_q2_bank_v1.json`, containing
  8 fixed motions x seeds 42/43, local_rp, DR scale 1.25, and K=8.
- `frontres_policy_quality_q2_report.py` validates exact manifest/result/item
  signatures, complete 16-row coverage, shared per-route state hashes,
  policy/noisy local-state and cache identity, corruption presence, one stable
  checkpoint per route, and finite scalar route Gain.
- Every item retains `abs(Gain_zero)` as its own noise floor and reports
  HSL-Zero, Policy-Zero, and Policy-HSL before motion or bank aggregation.
- Motion classification requires two distinct seeds. Scientific failure is a
  report verdict; only structural identity/schema corruption raises.
- The focused pseudo-data contract passes, including permutation invariance,
  missing-row/signature/non-scalar/non-finite/role/state rejection, and a
  deliberately negative HSL/Policy outcome that remains technically valid.
- `py_compile` passes for the report owner and contract; the aggregate Segment
  suite passes `53/53` with `failed_count=0`.

Evidence class: S1/S2 `T-schema/T-matched/T-oracle/T-bucket/T-seed/T-permute`.

Boundary:

- No training, PPO, Gain, reset, perturbation mask, or existing evaluator code
  changed. S4 Q2 collection remains pending and long training remains blocked.

## Q-E13 - Q2 Counterfactual Oracle Result

Date: 2026-07-17

Raw evidence:

- Log: `policy_quality_q2_bank_v1.txt`.
- Result: `policy_quality_q2_bank_v1_result.json`.
- Derived report: `policy_quality_q2_bank_v1_report.json`.
- Manifest signature:
  `b80831fe0bd2aa25c98487b863550af7c943d188809b2c1eb534c4163d63ac4b`.

Technical result:

- All 16 manifest items are present. Every item has zero/HSL/policy routes,
  one shared route state hash, matched policy/noisy local dynamics and caches,
  and nonzero policy/clean local_rp corruption.
- The independent Q2 reporter returns `technical_pass=true`; no simulator
  traceback or result-schema corruption was found.

Scientific result:

- HSL-Zero motion classes: positive 1, negative 4, unresolved 1, mixed 2.
- Policy-Zero motion classes: positive 0, negative 4, mixed 4.
- Policy-HSL motion classes: positive 1, negative 1, unresolved 3, mixed 3;
  per-item median is `0.001159566`.
- Therefore `oracle_valid=false`, `policy_useful=false`,
  `ppo_improvement_supported=false`, and `method_review_required=true`.
- Repair Cost is consistently nonzero for HSL/Policy (roughly 0.128-0.160 in
  this bank, weighted by 0.15), and can dominate small Style/Physics benefits.
  It is not the sole cause: several items also show negative Style or Physics
  changes before cost, so the HSL proposal itself is not uniformly executable.
- The walking-run items have unusually large zero noise floors (about 0.164
  and 0.209), making K=8 route differences on that motion poorly resolvable.

Decision:

- Q2 collection is complete but fails its positive-control quality gate.
- The first divergence is HSL versus zero, before PPO. Do not modify PPO and do
  not begin Q3, checkpoint trajectory evaluation, or long training.
- Next bounded step is an offline HSL/Gain learnability audit: separate
  pre-cost Style+Physics ordering from Repair Cost, verify model_200 proposal
  authority against its supervised target, and classify motion/seed failures.

## Q-E14 - Q2-A Gain Learnability Decomposition

Date: 2026-07-17

Scope: use only the matched Q2 result to locate whether HSL failures arise
before or after Repair Cost, without modifying the active Gain formula.

Evidence:

- `frontres_policy_quality_q2_report.py` now reconstructs the effective Repair
  weight from each persisted component tuple and rejects inconsistent rows.
- Observed shared weight: `0.150000006`, matching the active 0.15 contract.
- For each item it records pre-cost `Style+Physics`, route differences, and one
  failure owner relative to that item's zero noise floor.
- HSL-Zero item owners across 16 items: 5 execution degradation before cost,
  1 Repair Cost dominance, 3 insufficient pre-cost margin after cost, 4
  unresolved at zero noise floor, and 3 resolved improvements.
- The walking-run zero route has large negative Physics Gain in both seeds,
  producing zero noise floors about 0.164 and 0.209. This is paired-execution
  sensitivity at K=8, not action regularization because zero Repair Cost is 0.
- Focused golden/metamorphic Q2 reporter contract passes and verifies the
  reconstructed weight and failure-owner classification.

Decision:

- Do not remove, clip, or retune Repair Cost: cost dominance explains only one
  HSL item and is not the first common failure.
- The earliest unresolved owner is HSL proposal-to-execution quality. The next
  step must compare model_200 output to its canonical dynamic supervised target.
- Existing Q2 artifacts cannot answer that question because they do not store
  post-step HSL target/weight. Q2-B therefore remains a separate dedicated
  evaluator instrumentation step followed by one bounded S4 rerun.

## Q-E15 - Q2-B HSL Target Alignment Offline Preflight

Date: 2026-07-17

Scope: expose the canonical dynamic HSL supervised target inside only the
dedicated policy-quality evaluator, without changing training semantics or any
existing evaluator.

Implementation:

- `build_frontres_hsl_rollout_target()` returns one immutable target/weight/
  harm-weight object. Its default `write_transition=True` preserves the formal
  Stage 2 behavior; `False` is a non-mutating audit mode.
- `_RouteCapture` snapshots the applied task-space correction, calls the same
  owner after `env.step` only for the HSL route, and persists K-step target,
  weights, nonzero mask, action-target L2/cosine, and per-dimension sign
  agreement under `execution.hsl_supervision`.
- The dedicated owner computes the canonical formula in non-mutating audit
  mode even when the Stage 2 transition-write flag is disabled. Zero and
  policy routes never receive HSL supervision fields.
- The independent Q2 reporter supports `--require-hsl-supervision` and rejects
  missing, ragged, non-finite, or shape-inconsistent target evidence.
- QUALITY-ACTION-01 Atlas B4 links the canonical target write boundary and
  names HSL target/proposal/executability as the failure owner.

Offline evidence:

- S1 golden target contract passes hand-computed position residual,
  safe/broken/repair/harm weights, projection, full-env zeros, and mutating vs
  non-mutating behavior.
- S2 real-owner wiring contract reaches exactly K target captures on HSL,
  preserves zero/policy isolation, and leaves optimizer/sampler/warmup state
  unchanged.
- Reporter schema/golden/metamorphic contract and Atlas source-link contract
  pass.
- `py_compile` passes for all modified Python owners/tests; the aggregate
  Segment suite passes `54/54` with `failed_count=0`, and `git diff --check`
  passes.

Boundary and next:

- This proves implementation and formal dedicated-route wiring offline. Real
  model_200 action-target alignment remains S4-unconfirmed.
- Run the unchanged Q2 bank once with the new result schema, then invoke the
  reporter with `--require-hsl-supervision`. Do not start Q3 or long training.

## Q-E16 - Q2-B Training-Flag / Audit-Availability Fix

Date: 2026-07-17

Symptom:

- Live log `policy_quality_q2b_hsl_target_v1.txt` stopped before the first
  manifest item with `frontres_hsl_rollout_label_enabled=False`.

Root cause:

- `frontres_hsl_rollout_label_enabled` owns whether Stage 2 writes supervised
  targets into the training transition. Q2-B incorrectly reused it as a gate
  for whether the dedicated evaluator may compute the same formula without
  writing training state.

Fix and regression evidence:

- `build_frontres_hsl_rollout_target()` now separates
  `enforce_training_enable_flag` from `write_transition`; both default to the
  original formal-training behavior.
- Q2-B passes `write_transition=False` and
  `enforce_training_enable_flag=False`. It computes the canonical diagnostic
  target without enabling Stage 2 supervision or mutating transition state.
- The S1 target contract proves flag-off audit computation equals the flag-on
  target and leaves transition state unchanged.
- The S2 real-owner contract now runs with the flag explicitly False and still
  captures exactly K HSL targets while preserving zero/policy and training
  state isolation.

Decision: this was a dedicated evaluator integration defect, not a checkpoint,
HSL formula, or training-config defect. Rerun the same bounded Q2-B command.

## Q-E17 - Q2-B Real HSL Action-Target Alignment

Date: 2026-07-17

Raw evidence:

- Log: `policy_quality_q2b_hsl_target_v1.txt`.
- Result: `policy_quality_q2b_hsl_target_v1_result.json`.
- Derived report: `policy_quality_q2b_hsl_target_v1_report.json`, generated with
  `--require-hsl-supervision`.

Identity and schema facts:

- All 16 manifest items complete without traceback.
- Every item contains exactly K=8 HSL targets, sample/harm weights, nonzero
  masks, action-target L2/cosine, and six-dimensional sign agreement.
- All targets are nonzero and the existing matched state/role/corruption
  identity remains valid.

Observed alignment:

- Item-level mean action norm ranges 0.1326-0.1493, median 0.1429.
- Item-level mean target norm ranges 0.00627-0.1158, median 0.01326.
- Action/target norm ratio ranges 1.29x-23.29x, median 10.65x.
- Item-level mean action-target cosine ranges 0.390-0.990, median 0.910.
- The two wave items with larger targets have ratios 2.97x and 1.29x and both
  improve over zero. Many failed wash-head, parkour, 912, ROM, and catch items
  have ratios around 8x-23x.

Decision:

- Model_200 generally captures the repair direction but does not scale action
  magnitude with the dynamic HSL target. Its action norm is nearly constant
  while target difficulty spans almost twentyfold.
- This explains the Q2 pattern: large-target items can benefit, while small
  targets are over-corrected and lose Style/Physics before Repair Cost.
- The first failed owner is Stage 2 HSL magnitude calibration. Do not modify
  PPO, Gain weights, reset, masks, or action bounds, and do not begin Q3 or long
  training.
- Next: audit the Stage 2 target magnitude distribution, effective
  magnitude/over-loss configuration, and model_200 training diagnostics or
  checkpoint lineage. Retraining is not yet authorized.

## Q-E18 - Q2-C Stage 2 Magnitude And Lineage Audit

Date: 2026-07-17

Evidence:

- Derived artifact: `policy_quality_q2c_hsl_magnitude_audit_v1.json`.
- Owner: `frontres_policy_quality_hsl_magnitude_audit.py` replays the active
  supervised formula on Q2-B action/target/sample/harm tensors without runner,
  optimizer, or environment mutation.
- Contract: `frontres_policy_quality_hsl_magnitude_audit_contract.py`.

Facts:

- The artifact contains 128 valid held-out Q2-B rows. This is not the complete
  Stage 2 training distribution.
- Action/target norm ratio has median `10.669`; target norm median is `0.01167`.
- Weighted loss values are direction-pos `0.01983`, direction-rpy `0.00432`,
  magnitude `0.00462`, over `0.00367`, base-rot `0.00335`, harm `0.00403`.
- Proposal-gradient L2 is direction-pos `1157.31`, direction-rpy `0.02281`,
  magnitude `0.00619`, and over `0.00494`. Direction is `1.04e5x` the combined
  magnitude/over gradient before the shared `clip_grad_norm_(..., 0.5)`.
- The singular owner is the position cosine term when proposal position is
  near zero but the canonical target position is nonzero. It can consume the
  shared clipped gradient budget while providing no magnitude calibration.
- `save_runner()` persists actor/critic/optimizer/iteration and Gain config,
  but not the effective supervised config, training objective, or source
  checkpoint identity. Current runtime weights cannot prove model_200's
  historical training weights.
- No local model_200/model_warmup checkpoint or original Stage 2 log exists in
  this checkout, so checkpoint lineage and the full training target
  distribution remain unconfirmed.

Decision:

- Q2-C2 is complete: Stage 2 has deterministic supervised
  direction-versus-scale gradient competition and can produce an erroneous
  over-amplitude initialization.
- Q2-C1 remains partial. Do not retrain or alter PPO/Gain. The next bounded
  audit is not a Stage 2 repair. It must first test whether Stage 3 gives the
  failed real samples corrective Gain/advantage and moves policy mean toward a
  better magnitude.

## Q-E19 - Governance Correction: HSL Defect Is Not HRL Root-Cause Closure

Date: 2026-07-17

Evidence:

- User review: HSL is a warmup initializer; Stage 3 HRL is expected to correct
  an imperfect proposal through executable Gain.
- Q-E17/Q-E18: model_200 is over-amplitude and Stage 2 contains severe
  direction-versus-scale gradient competition.
- Earlier generic PPO contracts prove update mechanics and parameter change,
  but do not exercise the same Q2 over-amplitude samples end to end.

Facts and limitations:

- Q-E17/Q-E18 explain the bad starting point, not why the final HRL policy
  remains close to HSL.
- The real-sample relation `over-amplitude action -> Gain -> advantage -> policy
  mean correction` remains unconfirmed.
- Therefore Stage 2 must not be recorded as the complete root cause of final
  Stage 3 quality failure, and neither HSL retraining nor PPO/Gain changes are
  authorized yet.

Next:

- Execute bounded Q2-D offline causality: action-scale Gain sweep, credit-sign
  trace, and one controlled update-direction check on identical Q2 evidence.

## Q-E20 - Q2-D Offline Scale/Credit/Update Preflight

Date: 2026-07-17

Scope:

- Implement the independent Q2-D evaluator and controlled-update oracle without
  changing existing online/offline/sequence or zero/HSL/policy evaluator flows.

Evidence:

- `frontres_policy_quality_q2d.py` owns immutable scales, scaled-HSL execution,
  Gaussian score-gradient direction, and clone-only mean projection.
- `frontres_policy_quality_q2d_eval.py` restores the same scoring state for
  `0/0.25/0.5/0.75/1/1.25x` routes and calls the existing formal owner bundle.
- Dedicated `policy_quality_q2d_eval` shell mode, CLI flag, and thin runner
  connector are wired without importing Q2-D from the old evaluator.
- `frontres_policy_quality_q2d_contract.py` and
  `frontres_policy_quality_q2d_wiring_contract.py` pass.

Confirmed:

- Every scale route is sorted, unique, full-6D, state-matched, and training-state
  isolated.
- The old policy-quality evaluator does not import or call the Q2-D evaluator.
- For Gaussian PPO, the local mean direction is controlled by
  `advantage * (raw_action - old_mean) / sigma^2`, not by counterfactual Gain
  ordering alone.
- A canonical `compute_frontres_segment_ppo_loss()` update on a policy clone
  moves mean toward a positive-advantage sampled action and leaves the source
  policy unchanged.

Unconfirmed:

- Existing Q2-B artifacts contain only the executed 1.0x HSL route; alternate
  scales change GMT dynamics, so their Gain cannot be reconstructed offline.
- No persisted failed real Stage 3 PPO batch currently supplies raw actions,
  old means/sigmas, returns, advantages, and valid mask to the new oracle.

Decision:

- Q2-D1/D2 offline preflight is complete. Q2-D3 is the only next gate: one tiny
  matched live scale sweep plus capture of the corresponding real Stage 3
  credit tuple. No PPO/Gain/HSL modification or long training is authorized.

## Q-E21 - Q2-D Canonical Gain Serialization Fix

Date: 2026-07-17

Evidence:

- Live traceback: Q2-D local serializer rejected `FrontRESSegmentGainResult`.
- Fix: Q2-D now reuses the existing formal quality owner's dataclass-aware
  `_json_value` serializer for actions, Gain, and execution payloads.
- Focused Q2-D wiring and real-owner contracts pass; `py_compile` and
  `git diff --check` pass.

Decision: this was a Q2-D result serialization defect only. Canonical Gain
calculation, scaled actions, state restoration, and old evaluator flows are
unchanged. Rerun the same command.

## Q-E22 - Q2-D Scale Sweep Diagnostic Result

Date: 2026-07-17

Evidence: `policy_quality_q2d_scale_v1.txt` and
`policy_quality_q2d_scale_v1_result.json` contain all 16 Q2 items and six
scales. Mean Gain is `-0.00655/-0.00105/-0.00775/-0.00713/-0.01274/-0.02430`
for `0/0.25/0.5/0.75/1/1.25x`; 1.0x is best on only 1/16 items and is worse
than zero on 11/16.

Limit: every route reports `audit_identity_state=UNCONFIRMED` with no
transaction ID. The ordering supports the over-amplitude hypothesis but cannot
close matched Gain identity and must not be joined to a PPO tuple.

## Q-E23 - Q2-D Transaction And Official Credit Wiring Repair

Date: 2026-07-17

Implementation and evidence:

- The dedicated scale owner sets a route-specific transaction ID and batch
  signature after state restore and before canonical Gain computation.
- `run_frontres_segment_single_update()` optionally writes one atomic
  `frontres_policy_quality_q2d_credit_v1` artifact after finalized storage is
  converted to the official PPO batch and before `optimizer.step`.
- The artifact preserves bounded/raw actions, old means/sigmas, canonical Gain
  reward, returns, advantages, valid mask, segment IDs, and the six-dimensional
  Gaussian mean score direction under one complete transaction identity.
- Focused S1/S2 contracts pass and prove incomplete identity fails closed,
  tensor rows align, the official owner installs the capture, and source
  rewards/advantages remain unchanged.

Decision: deterministic wiring is repaired without changing Gain, PPO,
sampler, warmup, or optimizer semantics. Q2-D3 remains partial until bounded
runtime artifacts confirm the identity and real sampled-action/advantage
covariance. Do not repeat broad training.
