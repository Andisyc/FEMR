# FRS-EVAL-v010 Clean Calibration R1 Offline Closure Plan

## Boundary

- Requested behavior: one deterministic local transaction must traverse the v010 manifest, fixed-K/M4 materializer, sealed Clean reset, repeated K-step raw producer, typed adapter, and final receipt.
- Preserved behavior: FRS-GAIN-v009 remains the active training Gain; the route stays read-only and invokes no Noisy, Repair, scalar Gain, Replay, optimizer update, checkpoint write, or training loop.
- Semantic owner: the existing clean-calibration gateway remains the use-case owner; existing Stage-1 materialization, reset, K-step trajectory, telemetry reducer, adapter, and transaction aggregate remain their sole owners.
- Public input/output: strict `FRS-EVAL-v010-clean-calibration-v001` manifest to validated JSON calibration receipt; invalid identity, missing evidence, mutation, RNG drift, or incomplete cleanup fails closed with no result.
- Dependency direction: the independent test imports production owners; production imports no test code. Only cache materialization and simulator/sensor effects may be deterministic fakes.
- State boundary: `frontres_readonly_collection_scope(route="clean_calibration")` is the Unit of Work; Actor, optimizer, replay, normalizers, sampler, curriculum, iteration, RNG, and result-file atomicity are unchanged.
- Legacy behavior: the EVAL-v004/v006, active v009 training, and existing local-scenario characterization tests remain unchanged.

## Effect Sketch And Pinch Point

```text
official manifest connector
  -> fixed-K/M4 materializer
  -> read-only transaction aggregate
  -> sealed clean_baseline reset
  -> repeated K-step raw trajectory
  -> Clean measurement and hard-event evidence
  -> typed collection adapter
  -> validated receipt and result file
```

The Pinch Point is `collect_frontres_clean_calibration_from_manifest`: every successful official collection must cross it, while every failure after materialization must close the same sealed batch and emit no partial result.

## Test Design

1. Characterize the strict route and current device/identity fix.
2. Add one independent full-chain pseudo-transaction using the real manifest connector, fixed-K/M4 materializer, reset owner, read-only aggregate, typed collection, adapter, and receipt writer.
3. Replace only external cache payload construction and simulator/sensor observations with deterministic fakes selected in the test process.
4. Require two distinct repeat seeds, identical Scenario/K/cache/GMT identity, complete hard-event evidence, restored RNG/state, exact close IDs, and one final receipt.
5. RED cases: mixed plan devices; request/plan K mismatch; duplicate/missing repeat; hard-event failure; state mutation; RNG drift; post-materialization exception; duplicate result path; any Noisy/Repair/Gain/optimizer call.
6. Re-run module semantics, materializer/reset characterization, route/static checks, relevant aggregate suite, `py_compile`, shell syntax, and `git diff --check`.

## Evidence Boundary And Stop Condition

- The full-chain fake is eligible for `OFFLINE-CHAIN`; it is `R1 OFFICIAL-OFFLINE` only if the unchanged production entrypoint and composition root execute and only the named external adapter is replaced.
- Local absence of IsaacLab is not permission to relabel a direct connector test as R1.
- Stop before server execution until all offline-fixable failures are closed, the final maintainability review has no P0/P1, and formal audit names exactly one remaining live-only fact.

