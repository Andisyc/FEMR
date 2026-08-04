# FRS v015 Interface-Oriented Refactor

Status: completed. Phase A outer Stage-3 interface closure completed at
E-FI-93; Phase B live-probe owner extraction completed at E-FI-94.

## Problem

The formal v015 route is runtime-confirmed, but its implementation is still
coupled through `runner: Any`, private runner attributes, dynamic module loads,
and dictionary-shaped transaction/telemetry carriers. The largest active
orchestrator, `frontres_segment_live_probe.py`, mixes reset, rollout, evidence,
transaction, update, checkpoint-barrier, and diagnostics responsibilities.

## Terminal Outcome

Keep the MOSAIC host unchanged while making the FEMR/FrontRES route depend on
explicit ports and immutable public records. Existing MOSAIC hooks remain the
compatibility shell; one FEMR Stage-3 engine becomes the composition owner.

## Frozen MOSAIC Host

The refactor must not modify:

- `scripts/rsl_rl/train.py`;
- `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`;
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py`;
- `source/rsl_rl/rsl_rl/algorithms/ppo.py`;
- `source/rsl_rl/rsl_rl/algorithms/mosaic.py`.

Existing FrontRES wrapper calls and command methods are treated as external
host APIs and are consumed only through FEMR adapters.

## Invariants

- METHOD-v016, GAIN-v006, PPO-v004, and TRAIN-v011 remain unchanged.
- Observation authority remains raw 870D -> q29 tail 58D -> combined 928D ->
  FEMR prefix 158D / frozen GMT suffix 770D.
- One FEMR action produces frozen-GMT K-step evidence.
- Each transaction contains exactly two Segment sources x active M attempts and
  commits exactly one optimizer update.
- Clean continuation remains GMT/Physics-only evidence.
- Checkpoint format remains `frontres-v015-checkpoint-v6` and old payloads are
  rejected before mutation exactly as before.

## File Responsibility Map

Create:

- `rsl_rl/frontres/frontres_interfaces.py`: Protocols, immutable identity/
  observation/commit records, and interface validation only.
- `rsl_rl/runners/frontres_stage3_engine.py`: the only formal Stage-3
  transaction orchestration owner and MOSAIC adapter composition point.
- `rsl_rl/tests/frontres_interface_refactor_contract.py`: semantic fake host,
  interface lifecycle, identity, shape, and exception-cleanup contracts.

Modify:

- `frontres_segment_live_update_loop.py`: compatibility functions delegate to
  `FrontRESStage3Engine` instead of coordinating private live-probe functions.
- `frontres_segment_runner_boundary.py`: expose an explicit run mode and reject
  invalid mixed mode combinations without expanding boolean consumers.
- `frontres_segment_live_training.py`: validate final emitted telemetry against
  the typed transaction/identity view without changing emitted fields.
- `rsl_rl/frontres/__init__.py`: preserve its public API through lazy exports so
  narrow interface imports do not eagerly load simulator dependencies.
- repository Architecture and this plan: record the new ownership boundary.

Phase A compatibility boundary:

- E-FI-93 retained `frontres_segment_live_probe.py` as the concrete backend
  behind the Stage-3 port. That protected runtime behavior but did not close
  the repository-discipline hotspot: transaction, reset, storage,
  one-action-K, rollout/Physics, update and reporting behavior still coexisted
  in one 5840-line module. Phase B supersedes this retention decision without
  changing method or runtime semantics.

## Phase B One-Shot Live-Probe Owner Extraction

Terminal outcome: `frontres_segment_live_probe.py` becomes a thin compatibility
facade and legacy probe entry. Every active responsibility is implemented in
one CCP-coherent owner with public consumer-shaped functions; formal training
and policy-quality consumers no longer import cross-module private functions.

Scope:

- extract immutable runtime records and checkpoint-transaction barrier state;
- extract reset/metadata installation, rollout storage/Gain capture, policy
  adapter/update, observation/one-action-K execution, simulator Physics capture,
  formal transaction assembly/commit and probe reporting into named owners;
- preserve characterized aliases at the facade only where existing tests or
  legacy callers require compatibility;
- migrate active consumers to public owner functions and reject new private
  imports through a static contract;
- refresh Architecture, task canvas, checklist, inventory and evidence after
  the aggregate deterministic suite passes.

Non-scope: METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, tensor formulas,
928/158/770 authority, K/M schedule, checkpoint-v6 payload, simulator behavior,
MOSAIC host files, policy quality, training and live execution.

Owner map:

- runtime records and Unit-of-Work state -> `frontres_segment_runtime_types.py`;
- reset and scenario metadata adapter -> `frontres_segment_live_reset.py`;
- rollout storage and paired Gain capture -> `frontres_segment_live_storage.py`;
- policy adapter and one optimizer update -> `frontres_segment_live_policy.py`;
- observation and one-action-K evidence -> `frontres_segment_one_action_k.py`;
- Contact/ZMP and motion-quality IsaacLab gateway -> `frontres_segment_physics.py`;
- physical K rollout shell -> `frontres_segment_live_rollout.py`;
- formal request/commit adapter -> `frontres_segment_formal_transaction.py`;
- diagnostic formatting and legacy probe report ->
  `frontres_segment_probe_reporting.py`;
- compatibility imports and legacy probe entry only ->
  `frontres_segment_live_probe.py`.

Embedded acceptance:

- characterization contracts for reset, one-action-K, Physics, transaction,
  storage/PPO and policy-quality consumers;
- static dependency assertions that production consumers do not import facade
  private symbols and the facade contains no semantic implementation owner;
- Python compilation, focused contracts, complete FrontRES deterministic suite,
  JSON Architecture parsing, `git diff --check` and frozen-host zero diff;
- `code-review-expert` final gate with all in-scope P0/P1 findings corrected.

Stop condition: extraction would change tensor values, lifecycle order,
checkpoint identity or exact-one update; an acyclic public dependency graph
cannot be formed without changing METHOD/GAIN/PPO/TRAIN semantics; a frozen
MOSAIC host edit becomes necessary. No simulator/training/live run is required
for behavior-preserving module movement.

## One-Shot Completion Unit

Terminal outcome: make the existing Stage-3 seam enforce the accepted identity,
K x M shape, exact-one receipt and final serializer contract itself, while the
MOSAIC host remains byte-identical.

Scope:

- replace `object` transaction boundaries with a consumer-shaped request port
  and immutable request view;
- bind committed receipts to the exact request transaction, frozen-policy and
  K x M row identity;
- make checkpoint-v6 identity mandatory at the final telemetry consumer;
- make `frontres_segment_live_update_loop.py` the explicit composition root for
  the concrete compatibility backend, so `frontres_stage3_engine.py` no longer
  locates or imports `live_probe` owners;
- add an explicit engine-local Unit of Work for idle/collecting/committing state,
  rejection rollback and exception cleanup;
- switch active evaluation consumers away from cross-module private Physics
  accessors, retaining private aliases only as characterized compatibility
  seams where existing tests or the frozen host still require them.

Non-scope: METHOD/GAIN/PPO/TRAIN mathematics, tensors, observations, K/M
schedule, checkpoint format, simulator behavior, policy quality and MOSAIC host
source.

Embedded acceptance: request/receipt/telemetry negative cases, composition-root
connectivity, exact-one/recollection/cleanup, focused formal transaction and
checkpoint contracts, the aggregate deterministic suite, compilation,
`git diff --check`, frozen-host diff inspection and final discipline review.

Stop: method or persistence semantics must change; a second mutable transaction
owner is required; private concrete imports remain inside the engine; exact-M
or exact-one cannot be proven through the final consumer.

## Core Parameter Path

```text
sealed scenario identity
-> 158D actor observation
-> one full-6D action
-> frozen-GMT K evidence
-> v006 Intent/Physics evidence
-> grouped PPO-v004 update
-> exact-one committed receipt
-> checkpoint-v6 and telemetry serializer
```

## Embedded Verification

- core-param contract: scenario/row/K/M/provenance identity crosses the engine;
- secondary contracts: host adapter, mode, missing-field, invalid identity,
  exception cleanup, checkpoint and telemetry views;
- full FrontRES deterministic contract suite;
- Python compilation and `git diff --check`;
- frozen MOSAIC file hashes and zero diff before/after;
- final `code-review-expert` review.

No simulator, training, deployment, or live run is authorized by this closure.
Those remain live-only evidence after the offline refactor.

## Completion Evidence

- one `FrontRESStage3Engine` is resolved per runner and owns the formal
  transaction state machine;
- typed contracts freeze METHOD/GAIN/PPO/TRAIN/checkpoint identity,
  928/158/770 observation authority, exact two-Segment x M layout, exact-one
  committed update and final telemetry identity;
- deterministic contracts cover provider-time optimizer mutation, bounded
  invalid-evidence recollection, exception cleanup, mode mixing, K8/M2,
  K16/M3 and K32/M4 row shapes;
- formal training, sampler, checkpoint, Gain/storage/diagnostics and runtime
  owners use normal package imports, so one Python module identity owns each
  schedule, audit switch and evaluator; active cross-module calls use public
  seams while historical aliases remain compatibility-only;
- isolated contracts share one lightweight package composition helper instead
  of manufacturing inconsistent empty packages or relying on production
  file-path loaders;
- `frontres_segment_all_contract_suite.py` passes 64/64;
- modified Python files compile and `git diff --check` passes;
- the frozen MOSAIC host file diff is empty.

## Phase B Completion Evidence

- `frontres_segment_live_probe.py` is now a 225-line import-only compatibility
  facade. It owns no function, class or mutable transaction implementation.
- Runtime records, logging, policy update, reset, storage/Gain capture,
  one-action-K, formal transaction, Physics, rollout, reporting and the legacy
  probe entry each have one named owner module; every owner remains below 1000
  lines.
- Active training and policy-quality consumers import their public owner
  directly. Only the frozen `on_policy_runner.py` continues through the facade
  for the legacy probe entry.
- A structural contract rejects semantic bodies in the facade, cross-owner
  private imports, dependency cycles, owner-size regression and active-consumer
  facade bypass.
- Characterization tests were migrated to patch the real owner boundary rather
  than facade globals. The complete deterministic suite passes 64/64, Atlas
  source links resolve to the extracted owners, modified Python compiles and
  frozen MOSAIC host files remain untouched.
- No simulator, training, deployment or live run was used. METHOD-v016,
  GAIN-v006, PPO-v004, TRAIN-v011, checkpoint-v6 and 928/158/770 semantics are
  unchanged.

## Stop Conditions

Stop if the implementation requires a MOSAIC-host edit, changes a method or
persistence identity, changes tensor/role semantics, permits a second mutable
transaction owner, or cannot preserve exact-one update and exception cleanup.

## Phase C One-Shot Adjacent-Owner Closure

Status: completed offline at E-FI-95. This phase is behavior-preserving and does
not reopen METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011 or EVAL-v003.

Terminal outcome: the remaining FrontRES runner hotspots become narrow
orchestration/gateway modules. A change to exact-M transaction identity,
formal telemetry, checkpoint identity, deployment composition or policy-quality
state starts in one named owner and reaches consumers through public seams.

Main execution unit:

- extract the frozen-policy/exact-M transaction aggregate from
  `frontres_segment_live_sampler.py` into one transaction owner shared by the
  sampler, runtime records and formal commit path;
- extract formal transaction telemetry and legacy train-time evaluation from
  `frontres_segment_live_training.py`; remove the reverse dependency from
  `frontres_segment_formal_transaction.py` to the training loop;
- isolate read-only quality checkpoint inspection from the mutable
  `frontres_checkpointing.py` save/load gateway without changing checkpoint-v6
  or HSL-v1 payloads; the gateway consumes a public identity surface rather
  than cross-owner private names;
- isolate legacy sequence and policy-quality compatibility behavior from the
  active v015 deployment and held-out evaluators;
- keep compatibility re-exports only where existing callers require them, with
  structural tests proving that active owners do not depend on the facades.

Embedded acceptance:

- characterization and semantic contracts for exact-M sealing, telemetry final
  serialization, HSL-v1/Stage3-v6 reject-before-mutation, deployment carrier,
  held-out state restore and legacy isolation;
- static owner checks for public imports, acyclic dependencies, absence of
  reverse `formal_transaction -> live_training` imports and absence of semantic
  definitions in compatibility shells, including both active/legacy
  policy-quality module load orders;
- modified-file `py_compile`, focused contracts, complete deterministic suite,
  Architecture source-link checks and `git diff --check`;
- final-gate review under FRS-ENG-v001. P0/P1 findings are repaired and
  re-reviewed inside this closure.

Non-scope: simulator, training, live run, policy-quality claims, formula or
tensor changes, checkpoint migration, new plugin/registry/DI framework, and any
MOSAIC host edit.

Stop condition: stop on any changed serialized value, transaction row/order,
checkpoint payload/restore mutation, evaluation report identity, test-observed
exception behavior, import cycle, or required host edit.

Completion evidence:

- exact-M transaction identity/lifecycle now lives in
  `frontres_segment_transaction.py`; sampler evidence/reporting lives in
  `frontres_segment_sampler_reporting.py`;
- committed telemetry lives in `frontres_segment_training_telemetry.py`, while
  the 650-line training owner retains only validation, iteration, save cadence
  and console orchestration;
- `frontres_checkpoint_quality.py` owns read-only strict artifact inspection;
  `frontres_checkpointing.py` remains the only mutable save/load gateway;
- active v015 deployment and held-out evaluators no longer define legacy
  sequence planning or matched-counterfactual actors. Compatibility exports
  resolve to explicit legacy owners;
- E-FI-95 focused and aggregate evidence proves unchanged exact-M/exact-one,
  telemetry rejection, checkpoint-v6/HSL-v1 payloads, evaluation identity and
  exception cleanup. No simulator, training or live run was used.

Hotspot disposition: the remaining files above 1000 lines are accepted deep
modules, not unreviewed facades. E-FI-95 records their single change reason,
containing contracts and expiration trigger. A new responsibility, checkpoint
format/backend, production evaluation entrypoint/report schema, or a 2000-line
crossing reopens the FRS-ENG-v001 hotspot gate.

## Phase D One-Shot Domain And Algorithm Owner Closure

Status: completed offline at E-FI-96. This phase is behavior-preserving and does not reopen
METHOD-v016, GAIN-v006, PPO-v004, TRAIN-v011, EVAL-v003, checkpoint-v6 or the
928/158/770 observation authority.

Terminal outcome: a future Scenario, evidence, projection or Stage-1 dependency
change starts in one FrontRES owner and reaches active consumers through a
narrow public import. Compatibility modules preserve existing imports but own
no duplicate behavior. The frozen MOSAIC host remains unchanged.

Main execution unit:

- split immutable local-scenario and retired fixed-Noisy lifecycle from the
  stateful Segment sampler; keep selection, priority, budget and sampler
  persistence together;
- split one-action-K/paired-Gain evidence, grouped candidate records/adaptation
  and mutable rollout storage into separate owners, leaving the old storage
  module as a characterized compatibility surface;
- split grouped first-order projection and actual Adam commit authority from
  grouped PPO loss, and split actual-update validation, v015 local evaluation
  and generic/legacy report formatting from the diagnostics hotspot;
- replace production `spec_from_file_location` dependency fallbacks with normal
  package imports, and remove the policy-quality and formal-transaction hidden
  dependency cycles by moving shared records/state toward stable owners and
  keeping concrete composition at the runner boundary;
- migrate active FrontRES consumers to the new public owners while retaining
  tested re-exports only for frozen host and legacy contract compatibility.

Core parameter paths:

```text
Segment sample -> sealed local scenario rows -> scenario/hash identity
one-action-K evidence -> paired facts -> v006 return -> grouped candidate batch
grouped advantages/constraints -> projected gradient -> actual Adam delta
owner value -> final diagnostics/report/checkpoint consumer
```

Embedded acceptance:

- meaningful owner fixtures for scenario immutability/provenance, evidence
  pairing/applicability, grouped row permutation/masks and actual-update KKT;
- static dependency checks for normal production imports, public consumer
  imports, acyclic owner dependencies and compatibility modules without
  semantic bodies;
- focused contracts, complete deterministic FrontRES suite, modified-file
  compilation, JSON/source-link checks, `git diff --check` and frozen-host zero
  diff;
- `code-review-expert` construction/final-gate review under FRS-ENG-v001, with
  in-scope P0/P1 findings fixed before closure.

Non-scope: any method/training/reward/optimization/checkpoint identity change,
tensor or row semantic change, simulator/training/live run, policy-quality
claim, new DI framework/plugin registry, and edits to `commands.py`, `train.py`,
`on_policy_runner.py`, `ppo.py` or `mosaic.py`.

Stop condition: stop if extraction changes a serialized value, row order,
scenario/transaction identity, projection result, optimizer mutation,
checkpoint payload, exception behavior or requires a frozen MOSAIC host edit.

Completion evidence:

- scenario planning, active immutable local scenarios and retired fixed-Noisy
  lifecycle now have separate owners; `frontres_segment_sampler.py` retains
  only stateful selection, priority, budget and persistence;
- evidence records, paired one-action-K facts, grouped candidate adaptation and
  mutable rollout storage have separate owners. `frontres_segment_storage.py`
  is a 41-line compatibility surface with no semantic body;
- `frontres_constraint_projection.py` owns grouped first-order projection and
  actual Adam commit/restore authority. `frontres_segment_ppo.py` retains loss
  and grouped reduction; update validation and report/evaluation construction
  likewise have dedicated owners behind a 51-line diagnostics facade;
- production FrontRES modules use normal package imports. Formal transaction
  dispatch uses the runner port, while policy-quality interfaces and scoring
  state no longer create an evaluator/formal-owner import cycle;
- runtime and policy-quality Atlas links point to the extracted owners and
  exact current `# Bn:` boundaries. The aggregate deterministic suite passes
  64/64, modified Python compiles, JSON parses, `git diff --check` passes and
  frozen MOSAIC host files remain unchanged;
- no simulator, training, deployment or live run was used. The active runtime
  cursor remains P5-C; this closure adds no policy-quality claim.

## Phase E Evaluation Legacy-Isolation Closure

Status: completed offline through E-FI-98. E-FI-97 closed legacy/dependency
isolation; E-FI-98 closes the active composition runtime Gateway and the
EVAL-v003 same-state Baseline/Repair path.

Requested behavior: keep active v015 evaluation, historical evaluation, and
training orchestration behavior unchanged while making their dependency and
entrypoint boundaries explicit. Preserved behavior: METHOD-v016, GAIN-v006,
PPO-v004, TRAIN-v011, checkpoint-v6, 928/158/770, one-action-K, exact-M and
exact-one update. No simulator, training, deployment or live run is part of
this phase.

Owner/interface closure:

- `frontres_segment_training_evaluation.py` is the explicit legacy
  periodic/offline/sequence owner and imports its plan from
  `frontres_segment_legacy_sequence_eval.py`; the active deployment evaluator
  no longer re-exports the legacy plan;
- `on_policy_runner.py` connects directly to the legacy evaluation owner;
  `frontres_segment_live_training.py` owns no evaluation functions, formatting
  helpers, or duplicate legacy Gain-name tables;
- `run_frontres_policy_quality_eval()` is v015-only and fails closed outside
  the formal transaction route. Historical matched-route evaluation requires
  the explicitly named `run_frontres_legacy_policy_quality_eval()` entry;
- policy-quality state images and JSON projection are public owner interfaces;
  consumers no longer import cross-module private helpers.

Dependency and state boundary: Evaluation may read sealed runtime/checkpoint
state and write its declared report only. It must not update optimizer,
sampler, transaction, return, priority or warmup state. Active evaluators must
not import a legacy owner through a compatibility re-export.

Closed semantic boundary: EVAL-v003 already requires both branches to consume
the same fixed carrier, initial state and RNG/reset controls. The executor now
captures one canonical route-start image, restores it before each sequential
branch, runs Baseline with exact-zero FEMR action and Repair with per-frame
FEMR, and rejects state-hash drift, Baseline FEMR use or nonzero Baseline
actions.

Closed engineering boundary: `frontres_runtime.py` owns the consumer-shaped
`FrontRESV015DeploymentRuntimeGateway`. It validates and contains runner-private
connectors, simulator step/sensor IO, observation/normalizer authority,
FEMR/GMT calls, route-start snapshot/restore and training-state isolation.
`frontres_segment_sequence_eval.py` now depends on that public Gateway and owns
only request semantics, paired branch orchestration, pure reduction and atomic
reporting. This removes the named `Inappropriate Intimacy` rather than hiding
it behind a pass-through wrapper.

Evidence: modified Python compiles; deterministic carrier S2A, composition
S1/S2B, evaluation-isolation and interface-refactor contracts pass. No
simulator, training, deployment or live run occurred; physical composition
quality remains an S4 claim.
