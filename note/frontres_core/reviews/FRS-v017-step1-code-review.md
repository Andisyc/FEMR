# FRS-v017 Step 1 Code Review

## Construction Review - 2026-08-02

Mode: `code-review-expert::construction_review`

Reviewed boundary:

```text
FrontRESExecutedKTrajectory
-> FrontRESSegmentBaselineEvidence / FrontRESRepairAttemptEvidence
-> FrontRESSealedRecoveryAwareGainBatch
-> FrontRESRecoveryAwareGainInput
-> compute_recovery_aware_gain
```

Accepted behavior: active GAIN-v007 fixed semantic scales, executed Clean/Noisy
baseline reuse, one Repair row per attempt, continuous K aggregation,
smooth-worst family aggregation, Recovery-Aware pressure, full-6D cost, and
fail-closed evidence.

Findings:

- P0: none.
- P1, resolved: trajectory-local ZMP validation lacked expected-support phase
  authority. Applicability validation now occurs only at baseline/attempt seal,
  where both actual Contact and Clean expected support are present.
- P1, resolved: continuous channel carry-forward initially admitted zero values
  before the first valid observation. The owner now carries values only after a
  valid frame has been observed and rejects rows with no valid observation.
- P2: none open.

Discipline result:

- the new frozen Value Objects remove the prior baseline/attempt Data Clump and
  protect one-time baseline identity without adding a runner or Service Layer;
- `frontres_gain.py` remains the sole formula Pinch Point and imports no runner,
  PPO, persistence, simulator, or diagnostic owner;
- the evidence owner exposes explicit tensors rather than a stable open dict;
- source annotations identify the B1 validation, B2 normalized family
  construction, and B3 scalar ordering boundary;
- the hand-calculated deterministic contract passes for Clean anchoring,
  pressure identity, N/A ZMP, malformed payload rejection, row permutation,
  exact-once baselines, and beta cost ordering.

Verdict: `READY_FOR_CONSUMER_MIGRATION`.

The consumer migration and final-gate result are recorded below.

## Final Gate Review - 2026-08-02

Mode: `code-review-expert::final_gate_review`

Reviewed route:

```text
train.py composition
-> Stage3 Unit of Work
-> one Clean + one Noisy baseline / Segment
-> exact-M typed Repair evidence
-> sole v007 Gain owner
-> grouped scalar PPO-v005 / exact-one optimizer
-> telemetry + v004 local report
-> checkpoint-v7 save/resume
```

Findings:

- P0: none.
- P1, resolved: active Stage3 could still dispatch the v002/v006/quartet local
  and policy-quality evaluators. `train.py` and the launcher now reject those
  modes before active execution; v004 sequence composition remains separate.
- P1, resolved: formal materialization returned `clean_reference_t`, but the
  local-scenario schema rejected/omitted it. The schema now validates and
  carries the GMT/Physics-only field without exposing it to the actor.
- P1, resolved: checkpoint/interface/launcher and isolated test stubs retained
  v6/v004 projection identities. They now bind v7/v005 or are removed from the
  active aggregate rather than made falsely compatible.
- P2, accepted removal debt: historical v002/v006 formula and evaluator code
  remains in its existing files for traceability. Active config, transaction,
  loss, optimizer, telemetry, checkpoint and launcher imports cannot reach it.

Discipline dimensions:

- ownership/change reasons: one Gain Pinch Point, one evidence carrier owner,
  one transaction Unit of Work, one PPO update owner and one persistence Gateway;
- public interface/caller knowledge: frozen baseline/attempt/sealed-batch Value
  Objects replace open cross-layer dictionaries and prohibit silent defaults;
- dependency direction: pure Gain imports no runner/PPO/checkpoint code;
  framework state is adapted only within runner/Physics Gateway modules;
- legacy safety: Characterization Tests preserve 928/158/770, full-6D,
  one-action-K, exact-M and exact-one behavior; legacy local evaluators fail closed;
- state/reliability: partial/mixed evidence, resampling, non-finite required
  values, v6/tampered payloads and collecting-state saves reject before commit;
- hotspot delta: the active additions replace formula/update/persistence
  authority; no runner, service, wrapper or second optimizer was added.

Evidence consumed:

- `py_compile` over changed production and focused contract modules;
- 53/53 active aggregate deterministic contracts;
- focused v007, grouped PPO, formal transaction, checkpoint-v7 and formal audit
  contracts;
- `bash -n` for the Stage3 launcher, structural old-import search and
  `git diff --check`.

Unavailable evidence: IsaacLab execution counts, physical Contact/ZMP values,
production checkpoint-v7, policy efficacy and beta calibration. These belong
to separately authorized Step 2 or later experiments.

Verdict: `APPROVE` for Step 1 deterministic closure. No open P0/P1 finding.

## Repository Health Review - 2026-08-02

Mode: `code-review-expert::repository_health_review`

This review re-opened maintainability and lifecycle structure after the Step 1
closure. It does not revoke the recorded 53/53 deterministic result, but it
supersedes the earlier statement that no open P1 code-discipline finding
remained.

### Open P1 Findings

1. **Baseline execution cardinality is narrowed after, not before, collection.**
   `_build_frontres_v015_local_transaction_request()` selects one representative
   row per Segment only after `collect_frontres_v017_no_actor_baseline()` has
   stepped and captured every vectorized role row. The retained evidence has one
   row, but the framework execution and sensor capture still occurred for the
   allocation scaffold. This contradicts the METHOD-v017/GAIN-v007 statement
   that each Segment executes one Clean and one Noisy baseline, and the current
   deterministic tests assert metadata counts rather than the real `env.step`
   and capture cardinality. Named gates: `Effect Sketch`, `Pinch Point`, and
   `Humble Object`.

2. **One Unit of Work has multiple mutable state representations.**
   `FrontRESStage3Engine` owns a lifecycle string while runtime helpers mutate
   `_frontres_v015_checkpoint_transaction_state` as an open dictionary and the
   request builder publishes batch, sample, observation trace, and diagnostics
   through additional runner-private attributes. Abort also reads the command's
   `_frontres_local_scenario_active` field directly, suppresses command lookup
   failures, and can mark checkpoint state idle after command cleanup was
   skipped. Named gates: `Data Clumps`, `Inappropriate Intimacy`, `Aggregate`,
   and `Unit of Work`.

3. **The active runner imports the retired projection through a compatibility
   facade.** `on_policy_runner.py` imports three functions from
   `frontres_segment_live_probe.py`; importing that facade eagerly imports the
   retired v004 constraint projection and a large inventory of private helpers.
   The old projection may not execute, but it is not isolated from the active
   import graph and every retained private export obstructs deletion. Named
   gates: `Common Reuse Principle`, `Middle Man`, `Speculative Generality`, and
   `ADP/SDP`.

### Open P2 Findings

4. **Active and superseded generations share the same semantic-owner files.**
   `frontres_gain.py` contains v007 followed by the superseded v006/v003 owner;
   `frontres_segment_evidence.py` contains v017 records followed by the old v015
   evidence family; `frontres_checkpointing.py` combines temporary quality
   route switching, HSL persistence, Stage-3 persistence, generic runner state,
   and historical Gain configuration. These are confirmed `Divergent Change`
   hotspots, not line-count findings.

5. **Checkpoint loading accepts pickle-capable payloads.** All three active
   checkpoint entrypoints call `torch.load(..., weights_only=False)`. A malicious
   operator-supplied checkpoint can execute arbitrary pickle code on the
   training host. Exploitability requires control of the checkpoint artifact or
   CLI path, but the impact is host-level code execution. The current strict
   post-load identity validation occurs too late to mitigate deserialization.

6. **Active v017 behavior is still exposed through v015 names.** Formal request,
   engine, transaction state, telemetry, checkpoint, and runner attributes all
   retain `v015` names while enforcing METHOD-v017/GAIN-v007/PPO-v005/TRAIN-v012.
   This is a concrete `Shotgun Surgery` and caller-knowledge cost: contract
   evolution requires widespread version edits and makes active versus legacy
   symbols hard to distinguish.

### Removal And Repair Sequence

1. Add a characterization test at the baseline `env.step`/capture Pinch Point,
   then make the baseline Gateway accept and return only the two authoritative
   Segment rows. Do not add another scorer or baseline owner.
2. Replace runner-private transaction dictionaries and flags with one validated
   transaction Aggregate owned by `FrontRESStage3Engine`; expose immutable
   request/receipt projections to checkpoint and telemetry consumers. Make
   command cleanup an idempotent public operation and never inspect command
   private state from the runner layer.
3. Change `OnPolicyRunner` to import the three concrete owners directly. Retain
   the compatibility facade only for characterized historical consumers and
   assign it a deletion condition; verify that the active import graph no longer
   imports `frontres_constraint_projection`.
4. After legacy evaluator migration is separately authorized, delete the
   superseded v006/v003 Gain and v015 evidence families from the active owner
   files instead of moving them into more production modules. Keep checkpoint
   transport in `frontres_checkpointing.py`; move deterministic identity
   validation into the existing `frontres_checkpoint_quality.py` owner where it
   removes a real change reason.
5. Characterize actual checkpoint payload types, switch strict v7/HSL loads to
   `weights_only=True`, and reject unsupported object types before any mutable
   restore.
6. Introduce active semantic names only at the Composition Root and public
   transaction boundary, migrate consumers once, then delete v015 aliases.
   Avoid a permanent pass-through wrapper layer.

### Evidence And Scope

- Repository discipline: active FRS-ENG-v001.
- Static triage: `fuck-u-code 2.2.2`, 235 parsed files; scores were used only to
  select hotspots and assigned no severity by themselves.
- White-box files: active method/training/Gain/PPO contracts; Stage-3 engine,
  formal transaction, one-action-K, Physics capture, runtime types, Gain,
  evidence, checkpointing, compatibility facade, runner imports, and active
  aggregate contracts.
- Consumed evidence: existing 53/53 deterministic Step 1 result and E-FI-101.
  No tests or live run were executed by this review.
- Annotation pass: no source comments changed, so the Code Quality Atlas was not
  regenerated. Existing B-blocks were consumed only as reading aids.

Verdict: `REQUEST_CHANGES` (P0: 0, P1: 3, P2: 3, P3: 0).

## Repository Health P1 Closure - 2026-08-02

Mode: `code-review-expert::final_gate_review` after the authorized one-shot
engineering closure.

Resolved findings:

1. The one-action-K baseline Gateway now receives the two authoritative Segment
   rows before persistent frame construction. Vectorized IsaacLab still advances
   one framework phase over its allocated width, but sensor/dynamic evidence is
   projected at capture and only two rows can enter the Clean/Noisy trajectory.
   The Pinch-Point contract proves `K+1` vectorized steps, `K` captures, two
   output rows, and duplicate representative rejection.
2. `FrontRESStage3Engine` now owns one validated transaction Aggregate covering
   execution phase, checkpoint state, collection sample/batch, observation
   trace, and immutable pre-update diagnostics. Formal code no longer publishes
   these through parallel runner-private attributes, provider injection is an
   explicit argument, and abort uses the command's public idempotent cleanup.
   Failed cleanup leaves persistence `collecting` and therefore checkpoint save
   remains fail-closed.
3. `on_policy_runner.py` imports the formal sentinel, legacy probe, and single
   update from their actual owners. The compatibility facade and retired v004
   projection are absent from the active runner import graph.

Verification:

- focused interface, v017 baseline/Gain, scalar transaction, checkpoint-v7 and
  formal-runtime contracts pass;
- all changed Python modules pass `py_compile`;
- the aggregate deterministic suite reports `contract_count=53 failed_count=0`;
- structural checks reject facade imports, command-private lifecycle reads,
  duplicate transaction carriers, and active projection imports.

Final verdict: `APPROVE` for the P1 closure (P0: 0, P1: 0). The three prior P2
findings remain explicit non-scope and are not silently reclassified as fixed.

## P2 Engineering Plan Review - 2026-08-02

Mode: `code-review-expert::engineering_plan_review`.

Verdict: `READY`.

Repository discipline: active `FRS-ENG-v001`.

Accepted behavior/non-scope: the behavior-preserving P2 closure recorded in the
current v017 Engineering Plan; active method mathematics, transaction
cardinality, observation/action authority, serialized checkpoint-v7/HSL-v1
identity, simulator, training, live execution and Concept Figure are frozen.

Boundary reviewed: active/history Gain and evidence ownership, restricted
checkpoint deserialization before mutation, and active Composition Root/public
transaction naming.

Findings: no P0/P1 plan blocker. `Divergent Change`/CCP admits generation
isolation; the checkpoint-quality Gateway is the existing Pinch Point and
removes three unsafe loader dependencies; `Shotgun Surgery` admits one
version-neutral public-name migration. A wrapper, parallel semantic owner,
checkpoint format upgrade, compatibility padding and new service layer are
rejected.

Proof route: characterization plus structural isolation, malicious-object
pre-mutation rejection, HSL/v7 safe roundtrip, active transaction/telemetry
connectivity, `py_compile`, complete deterministic suite and final-gate review.
Runtime correctness and policy quality remain unconfirmed until the separately
authorized Step 2 sentinel.

## P2 Final Gate Review - 2026-08-02

Mode: `code-review-expert::final_gate_review`.

Verdict: `APPROVE` (P0: 0, P1: 0).

Resolved findings:

1. `Divergent Change` in active/history semantic files is removed without a new
   service layer: v007 Gain and v017 evidence remain active owners, while older
   formulas and carriers are explicit compatibility modules reached only by
   legacy paths.
2. The existing checkpoint-quality Gateway is the single restricted load Pinch
   Point. HSL-v1 and checkpoint-v7 reject unsupported object payloads before
   mutable restore; active checkpoint code contains no `weights_only=False`.
3. Public active boundaries use version-neutral names. Historical `v015` wire
   keys remain intentionally stable checkpoint data, not active code authority;
   no alias facade was retained.

Discipline audit:

- CCP/CRP/ADP/SDP improve: callers know fewer generation facts and dependencies
  point from orchestration to typed active owners or explicit compatibility
  modules;
- no new runner, wrapper, Protocol, service registry, open dictionary or
  mutable lifecycle carrier was introduced;
- hotspot change reasons decrease in Gain, evidence and checkpointing; no
  unrelated responsibility was added;
- method mathematics, 928/158/770 authority, full-6D action, sealed two-Segment
  x exact-M lifecycle, one-action-K, scalar PPO, checkpoint-v7 wire identity and
  HSL behavior are preserved.

Verification: focused S1/S2/S3 contracts pass, changed modules pass
`py_compile`, `git diff --check` passes, and the complete suite reports
`contract_count=53 failed_count=0`.

Residual boundary: real IsaacLab lifecycle, Physics evidence, parameter delta,
production checkpoint and policy effect remain S4 facts for the separately
authorized Step 2 bounded official transaction.

## TRAIN-v013 Construction Review - 2026-08-03

Mode: `code-review-expert::construction_review`.

Reviewed boundary:

```text
explicit ten-field per-K DRStageSpec
-> frontres_segment_warmup deterministic K/M/DR identity
-> sealed two-Segment x exact-M transaction
-> immutable local_rp class/strength materialization
-> grouped exact-one commit
-> checkpoint-v8 / read-only telemetry
```

Findings:

- P0: none.
- P1, resolved: `frontres_segment_live_sampler.py` imported the retired
  episode-length/frontier curriculum at module load even though the formal
  branch bypassed its functions. The import now occurs only after the explicit
  non-formal branch boundary. Formal TRAIN-v013 construction therefore does not
  load or depend on the old controller.
- P1, resolved: the literal 2.381 ceiling combined with Broken support
  `(d_cap, min(1.10*d_cap, ceiling)]` collapses when `d_cap=ceiling`. The owner
  now treats 2.381 as the outer Broken-tail ceiling and caps the terminal Hard
  `d_cap` at `2.381/1.10`, preserving all four classes without an online rule.

Discipline result:

- CCP keeps K/M/DR arithmetic and deterministic sampling in the existing
  `frontres_segment_warmup.py` owner;
- transaction, sampler, checkpoint Gateway and telemetry consume immutable
  projections and do not recreate schedule decisions;
- no new runner, wrapper, Service Layer, registry, mutable cross-layer dict or
  second curriculum owner was added;
- failure and partial transactions cannot advance the only progress input,
  `current_learning_iteration`.

Verdict: `READY_FOR_FINAL_GATE`, P0=0/P1=0.

## TRAIN-v013 Final Gate Review - 2026-08-03

Mode: `code-review-expert::final_gate_review`.

Formal route reviewed:

```text
train.py explicit config
-> OnPolicyRunner.learn_frontres_segment_live
-> TRAIN-v013 stage identity
-> formal transaction / one-action-K / grouped PPO
-> committed-only iteration
-> checkpoint-v8 + telemetry
```

Evidence consumed:

- affected warmup, four-class, transaction, checkpoint, formal-audit and
  launcher contracts pass;
- all 18 Module Test cards pass and the active aggregate reports 49/49;
- changed Python owners pass `py_compile`;
- Module Test Atlas and generated Code Quality Architecture checks pass;
- final `git diff --check` passes.

Final findings:

- P0: none.
- P1: none open.
- the historical five-field v011 helper and non-formal adaptive sampler branch
  remain compatibility code only. Formal configuration requires all ten v013
  fields and cannot select either fallback.
- simulator population, real Actor/std/Critic deltas and production checkpoint
  materialization remain Step 2 S4 facts, not inferred offline.

Verdict: `APPROVE` for TRAIN-v013 Step 1 / 2 offline closure.

## K-step Phase A Legacy-Isolation Final Gate - 2026-08-03

Mode: `code-review-expert::final_gate_review`.

Reviewed boundary:

```text
TRAIN-v013 resolved K/M
-> existing sampler transaction planner
-> exact-M Repair rows with one shared K
-> one-action-K evidence and checkpoint-v8 consumers
```

Findings:

- P0: none.
- P1: none.
- the sealed formal branch now bypasses the retired state-driven K/M planner
  instead of calling it and overwriting its output. The legacy branch remains
  characterized for compatibility and its audit IDs are explicitly legacy;
- the public transaction plan and caller knowledge are unchanged. No wrapper,
  Protocol, Service Layer, mutable owner or cross-layer dictionary was added;
- FRS-ENG-v001 ownership, CCP/CRP, Unit-of-Work and fail-closed boundaries are
  preserved. The change removes one legacy dependency from the active effect
  path and adds no method, Gain, PPO, checkpoint or simulator responsibility.

Evidence consumed: focused sampler, audit, interface, warmup, transaction,
one-action-K and checkpoint-v8 deterministic contracts; `py_compile`, JSON
parse and targeted `git diff --check`.

Unconfirmed boundary: real K-frame execution, Actor/std freeze and production
checkpoint-v8 values remain Formal Runtime Audit Phase B facts.

Verdict: `APPROVE` for the DP03 Phase A correction (P0: 0, P1: 0).
