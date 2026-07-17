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
