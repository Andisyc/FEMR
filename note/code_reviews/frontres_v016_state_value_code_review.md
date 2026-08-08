# TRAIN-v016 Future-Conditioned State-Value Code Review

Date: 2026-08-08
Review mode: construction review, removal review, security review, research-ML review, final gate review
Verdict: `APPROVE PHASE B LIVE`; long-training decision pending

## Reviewed Boundary

The change keeps the Actor at 158D, GMT at 770D, the direct full-6D action,
FRS-GAIN-v007, fixed split learning rates, warmup schedule and K/M/DR campaign.
It changes only the training-only state-value path:

```text
Critic state  = current privileged 289D + sealed Noisy future q29 58D = 347D
Critic target = exact-M arithmetic mean G_total per transaction source
Actor signal  = each attempt's own return minus the shared old state value
gradient      = independent Actor/std and Critic clip(0.5), one two-group Adam step
checkpoint    = checkpoint-v11; checkpoint-v10 cannot resume
```

## Findings Closed During Construction

- P1: value targets originally keyed the motion-local `segment_id`. They now
  key transaction `source_index`, so two source motions may reuse the same
  local Segment id without merging their targets.
- P1: checkpoint-v10 compatibility was initially too close to mutable resume.
  Historical v10 is now admitted only by strict read-only quality inspection;
  active restore requires the complete v11 identity before mutation.
- Test-fixture drift: old pseudo-runtime stubs replaced `rsl_rl.modules` with a
  non-package and could not import the observation owner. The fixtures now
  preserve package semantics and load the real owner. The formal transaction
  fixture also uses separate toy Actor input and exact 347D Critic state.
- Test-translation drift: two indentation errors and stale v017/v005/v015
  expectations were corrected. No numeric method oracle was relaxed.

No P0, P1 or P2 finding remains in the reviewed boundary.

## Discipline Review

| Gate | Result |
| --- | --- |
| Semantic owner | Observation composition, Segment target/clipping, transaction commit, diagnostics and checkpoint validation remain with their existing owners. |
| Dependency direction | Simulator/runner data flows into deterministic owners; PPO and checkpoint code do not read simulator private state. |
| Transaction boundary | Collection performs zero updates; one complete transaction installs two clipped gradient families and commits one Adam step. |
| Persistence | v11 validates 347D Critic, two optimizer groups, normalizers, curriculum and receipt before restore; save remains atomic. |
| Information boundary | Critic receives only current privileged state and the same sealed Noisy future q29 tail; no action, Clean future, evaluator outcome, Gain component or K is appended. |
| Removal debt | No second optimizer, wrapper, service, alternate target path or v10 migration was introduced. Historical v005/v10 code remains characterization/read-only only. |

## Security And ML Review

- Checkpoint reads retain mapping/type validation and `weights_only=True` at
  the shared loader; no dynamic evaluation, shell construction or unsafe
  deserialization path was added.
- All new state/target/gradient facts fail closed on malformed shape, mixed
  Segment identity, missing exact-M coverage or non-finite values.
- The exact-M mean changes only the value regression target. It does not replace
  per-attempt Actor advantages, winner-select attempts or add action information
  to `V(s)`.
- Actor/std and Critic parameters are partitioned by the explicit Critic module;
  the partitions are disjoint, clipped separately and still stepped by one
  existing optimizer.

## Verification

- Four confirmed v016 Module Test Cards pass with independent tensor/state
  oracles.
- Focused entrypoint, HSL, grouped PPO, formal transaction, read-only quality,
  checkpoint and telemetry regressions pass.
- `frontres_segment_stage3_pseudo_suite.py`: 13/13 passed.
- `frontres_segment_all_contract_suite.py`: 52/52 passed.
- Python compilation passed for all touched production owners and active tests.

This evidence proves offline module behavior and deterministic connectivity
only. It does not prove IsaacLab execution, Critic calibration quality, policy
improvement, checkpoint-v11 server reload or long-training safety.

## Phase B Probe Construction Review

The confirmed Phase B edit stays inside the existing formal-audit projection
and checkpoint Gateway. `AUDIT-B01..B07` consume immutable route/transaction
facts; `AUDIT-B08` can emit only after atomic replacement, restricted
checkpoint reload and the existing strict v11 validator. The readback snapshots
iteration, loaded-path identity and Actor/Critic fingerprints, and fails if the
validator mutates the live runner. With the formal flag disabled, no readback or
fingerprinting runs.

The audit source contains no training step, backward call, simulator step,
state-dict restore or checkpoint write. The shared loader retains
`weights_only=True`. Target and clipping probes are assertions only and never
feed the optimizer or diagnostics. Focused contracts, 13/13 pseudo contracts,
52/52 aggregate contracts, Python compilation, Atlas checks and diff checks
pass. No P0, P1 or P2 finding remains in the Phase B boundary.

## Phase B Live Final Gate Review

Commit `b74efd7` fixes the only final-gateway contradiction: the strict
normalizer validator had rejected legitimate constant dimensions even though
`EmpiricalNormalization.forward()` stabilizes zero standard deviation with
`eps`. The owner now admits `var=std=0` while still rejecting negative,
non-finite or inconsistent `std.square() != var` states. The production
serializer contract covers positive zero-variance round-trip and negative
tampering cases; the full 52-contract suite passes.

The bounded official transaction emitted B01-B08 exactly once with no
traceback. B07 reports one optimizer step and the expected critic-only role
deltas. B08 reports strict v11 readback with `runner_mutated=0`; independent
artifact inspection confirms a `[1,347]` Critic normalizer, 16 zero-variance
dimensions and consistent moments. No P0/P1/P2 remains. This review approves
the engineering/runtime boundary, not long-run policy quality.
