# FRS-EVAL-v004 v018 Compatibility One-Shot Engineering Plan

```yaml
plan_id: FRS-EVAL-v004-v018-compatibility-one-shot
status: active
effective_date: 2026-08-10
contract: FRS-EVAL-v004
method: FRS-METHOD-v019
training: FRS-TRAIN-v018
optimization: FRS-PPO-v007
checkpoint: frontres-v018-checkpoint-v13
closure_mode: engineering
```

## Terminal Outcome

The existing inference-only Held-out Policy Quality entrypoint accepts one
strict TRAIN-v018 checkpoint-v13, executes the unchanged K16 held-out question
with exact M4 Repair attempts, and writes an atomic report containing explicit
Segment-level Critic calibration rows. No training or policy-quality claim is
made by the offline engineering closure.

## Engineering Boundary Record

Requested behavior:

- migrate the active EVAL-v004 identity from METHOD-v017 / TRAIN-v015 /
  PPO-v005 / checkpoint-v10 / M3 to METHOD-v019 / TRAIN-v018 / PPO-v007 /
  checkpoint-v13 / M4;
- evaluate the checkpoint's 449D state-value Critic with its exact saved
  privileged-observation normalizer;
- report, per held-out Segment, the shared raw value, exact-M mean `G_total`
  target and raw error.

Preserved behavior:

- the held-out bank remains the same eight K16 Segments and the same local-rp
  corruption parameters;
- Clean/Noisy execute once per Segment and are reused across M Repairs;
- Actor remains the checkpoint's direct full-6D Actor and GMT remains frozen;
- GAIN-v007, `beta`, simulator behavior, action coordinates and local report
  fields are unchanged;
- Evaluation remains inference-only and atomically restores every installed
  module/normalizer state on success and exception.

Forbidden scope:

- no Actor, Gain, PPO loss, optimizer, training schedule, sampler, curriculum,
  simulator, MOSAIC or deployment-composition modification;
- no compatibility padding, partial checkpoint load, v10-to-v13 migration,
  evaluation feedback or multi-checkpoint orchestration;
- no simulator or server execution in this engineering unit.

Owners and public boundaries:

| Semantic object | Owner | Public boundary | Consumer |
| --- | --- | --- | --- |
| Active held-out identity | `FrontRESV018PolicyQualityManifest` | strict JSON parse/round-trip | active request builder |
| Checkpoint inference installation | `frontres_quality_route_actor()` | reversible context manager | held-out evaluator |
| Critic calibration projection | EVAL-v004 evaluator | immutable Segment rows | atomic JSON report |
| CLI admission | Stage-3 shell launcher | `policy_quality_eval` mode | `train.py` composition root |

Public input/output:

```text
HSL-v2 scaffold + v018 manifest + checkpoint-v13 + result path
-> strict immutable request
-> temporary Actor/Critic/prefix+privileged normalizer installation
-> four read-only K16/M4 transactions
-> frontres-v018-policy-quality-report-v1 JSON
```

Dependency direction:

- manifest and checkpoint inspectors own identity validation;
- the checkpoint Gateway owns temporary state installation/restoration;
- the evaluator consumes immutable identities and collection reports;
- Gain, PPO, training and simulator owners do not import Evaluation.

State and persistence boundary:

- checkpoint bytes are read before mutation and validated fail-closed;
- Actor, fixed std, Critic, prefix normalizer state, privileged Critic
  normalizer state and module modes are snapshotted and restored;
- optimizer, value-loss normalizer, sampler, transaction, receipt, curriculum,
  warmup and iteration remain byte-equivalent across Evaluation;
- result JSON is the only admitted write and is atomic.

Legacy rule:

- the v017 K16/M3 manifest and checkpoint-v10 reports become historical and
  are not accepted by the active request builder;
- the explicit generic legacy evaluator remains isolated and is not selected
  by `policy_quality_eval`.

Named engineering gates:

- Characterization Test: current request rejects v018 because it expects
  v017/v015/PPO-v005/checkpoint-v10.
- Effect Sketch: manifest -> request -> checkpoint Gateway -> 449D normalized
  Critic input -> policy values -> Segment calibration rows -> atomic report.
- Pinch Point: `frontres_quality_route_actor()` for checkpoint state and the
  active request builder for contract identity.
- Seam/Enabling Point: deterministic checkpoint mappings and fake external
  collection adapters selected only by tests; production composition remains
  `run_stage3.sh -> train.py -> OnPolicyRunner`.
- CCP/CRP: identity stays in the manifest/checkpoint owners; no new wrapper or
  service is introduced.

## Module Test Cards

### TEST-19 Active Manifest And Request

- Requested: accept exactly v018/checkpoint-v13/K16/M4 with HSL-v2 retained as
  a separate v017/v014 Actor scaffold.
- Input/output: active JSON plus sealed checkpoint identity -> immutable request.
- Oracle: exact literal IDs, dimensions and M4; v017 manifest, v10 policy,
  wrong Critic identity or HSL/policy identity mixing rejects before mutation.
- S/C/T: S1, C1/C3/C4, identity/fail-closed/round-trip.
- Stop: any fallback or mixed identity is accepted.

### TEST-20 Reversible Critic Installation

- Requested: install checkpoint Actor, 449D Critic and exact saved
  privileged-observation normalizer only inside the policy context.
- Input/output: asymmetric module and normalizer states -> installed checkpoint
  outputs in-scope, exact source restoration after success and exception.
- Oracle: independently constructed state dictionaries and fingerprints.
- S/C/T: S1/S3, C1/C3/C5, persistence/restoration/shape.
- Stop: optimizer changes, normalizer omission, partial load or restoration drift.

### TEST-21 Segment Calibration Projection

- Requested: four M4 attempts from one Segment share one value; target is their
  arithmetic mean `G_total`; report error is `value - target` in raw units.
- Input/output: asymmetric two-Segment M4 values/gains and source IDs -> two rows.
- Oracle: hand-computed means; row permutation within a Segment is invariant,
  mixed values, invalid rows or wrong group cardinality reject.
- S/C/T: S1, C1/C3/C4, aggregation/order/fail-closed.
- Stop: attempt-level target, value scaling or cross-Segment mixing appears.

### TEST-22 Official Read-Only Route

- Requested: the production launcher selects `policy_quality_eval` at 16 envs,
  reaches the active evaluator once and emits one atomic v018 report with zero
  training-state mutation.
- Input/output: preflight/official offline composition -> effective command and
  identity/effect receipts.
- Oracle: exact mode flags, env width, M4 report counts, checkpoint-v13 and
  zero-write state hashes.
- S/C/T: S2/R1 plus R2 checkpoint-normalizer restoration.
- Stop: training branch, 12-env/M3 identity, non-atomic output or state write.

The user's 2026-08-10 request confirms these four cards and authorizes local,
reversible implementation and offline verification. It does not authorize
server synchronization, simulator evaluation or training.

## Execution Sequence

1. Write TEST-19 through TEST-22 and observe the active v017/M3/normalizer
   omissions fail for the intended reasons.
2. Migrate the immutable manifest and request/report identities to v018 while
   preserving the held-out Segment bank.
3. Restore the checkpoint-v13 Critic observation normalizer in the existing
   checkpoint Gateway and add the deterministic Segment calibration projection.
4. Change only the active launcher admission from 12 rows/M3 to 16 rows/M4.
5. Run focused module tests, checkpoint persistence tests, entrypoint preflight,
   aggregate regression, py_compile, final code review and official offline
   route audit.
6. Synchronize Contract/Inspector/registry/checklist/evidence records and emit
   one uploadable server test command, without executing it.

## Stop Conditions

Stop before simulator/server work if any identity, 449D normalizer, zero-write,
calibration grouping, atomic-report, entrypoint or persistence test fails. Stop
and return to design authority if supporting v018 would require changing Actor,
Gain, PPO, GMT, simulator or the held-out scientific question.

## 2026-08-10 Active-Name Cleanup Amendment

This behavior-preserving cleanup removes repurposed historical v015/v017 names
from the active v018 Evaluation public dependency surface without moving logic
or introducing an abstraction.

- Requested behavior: active checkpoint format, held-out K16/M4 batch materialization,
  recovery-aware collection, inference mode and state guards use semantic names.
- Preserved behavior: checkpoint bytes/schema, strict identities, K16/M4 row layout,
  449D Critic coordinates, raw calibration arithmetic, JSON output, state restoration
  and the explicit legacy evaluator remain unchanged.
- Owners: checkpoint format remains owned by the checkpoint inspector; batch
  materialization remains owned by the live sampler; inference/state/report helpers
  remain private to the evaluator.
- Public boundary: rename existing symbols and all in-repository consumers directly;
  do not add aliases, wrappers, modules, registries or fallback paths.
- Dependency/state boundary: no owner moves and no runtime, checkpoint, optimizer,
  sampler, transaction, curriculum or simulator state changes.
- Evidence: run the existing active evaluator, checkpoint restoration, local-scenario
  batch and aggregate contract tests before and after the rename, then py_compile and
  verify that active Evaluation no longer imports or exposes the repurposed names.
- Stop: any behavioral test delta, remaining repurposed historical name on the active
  Evaluation public surface, or need to move logic returns this cleanup to review
  rather than expanding scope.
