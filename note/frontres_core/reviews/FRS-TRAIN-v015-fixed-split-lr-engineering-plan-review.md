# FRS-TRAIN-v015 Fixed Split-LR Engineering Plan Review

Date: 2026-08-07

```text
Engineering Plan Review
Mode: engineering_plan_review
Verdict: READY
Discipline: active (FRS-ENG-v001)
```

## Accepted Behavior And Non-Scope

The reviewed behavior is one existing `FrontRESTrackedAdam` with two named,
disjoint parameter groups: Actor `3e-6` and scalar Critic `1e-5`, under a fixed
schedule. The current frozen GMT, fixed task-space std, full-6D Actor, scalar
`G_total` Critic, phase weights, grouped PPO loss, exact-one transaction and
K/M/DR curriculum remain unchanged.

MOSAIC changes, a second optimizer, old Stage-3 optimizer migration, long
training, policy-quality claims and deployment are excluded.

The semantic decision is human-confirmed in the current DP09 Inspector
proposal. TRAIN-v015/checkpoint-v10 activation must occur first inside the
authorized execution unit and before production code mutation.

## Boundary Reviewed

```text
explicit fixed Actor/Critic LR config
-> FrontRESUnified owns named optimizer partition
-> existing grouped PPO installs role-specific gradients
-> one tracked Adam commit
-> committed telemetry
-> strict checkpoint-v10 save/reload
```

The plan has one engineering step because all internal changes share the same
optimizer identity, rollback boundary and terminal outcome. Long training is
correctly retained as the only material-cost boundary.

## Findings

P0: none.

P1: none.

P2: none after plan revision. The initial draft omitted an explicit Step Map
and human-confirmable Module Test Cards; both are now included in the reviewed
plan.

P3: none.

## Discipline Findings

- Ownership and change reasons: READY. Optimizer partition remains solely in
  `FrontRESUnified`; checkpointing and telemetry consume immutable facts.
- Public interface and caller knowledge: READY. Two finite LR values replace
  one ambiguous Stage-3 LR without exposing policy internals.
- Dependency direction: READY. CLI/config points toward the algorithm owner;
  persistence and diagnostics do not repartition parameters.
- Legacy-change safety: READY. The one-group route is characterized; constructor
  and checkpoint envelope are valid Pinch Points; v9 is rejected rather than
  migrated.
- Component placement: READY under CCP/CRP/ADP/SDP. Existing owners change for
  their current reasons; the plan adds no shallow module or cross-layer dict.
- Pattern admission: the existing transaction Unit of Work and checkpoint
  Gateway remain justified. No new wrapper, service, repository, registry or
  hierarchy is admitted.
- State and reliability: READY. Fresh construction, phase updates, rollback,
  exact-one count, save/reload and pre-mutation negative cases are explicit.
- Removal/hotspot delta: READY. Active shared/adaptive Stage-3 authority is
  retired; the unused duplicate config class is not promoted into a second
  owner.
- Research-code discipline: READY. Gradient roles, fixed std, frozen GMT,
  transaction identity and policy-quality evidence limits remain explicit.

## Flow And Proof Review

Every produced fact has a consumer. The four Module Test Cards independently
cover partition, phase commit, persistence and final composition/serialization.
The proof route includes S1 owner behavior, S2 official connectivity, negative
legacy rejection, S3 persistence and a bounded S4 critic-only transaction.

The final serialized telemetry check correctly follows producer -> committed
transaction -> serializer. A constructor-only assertion is not accepted as
consumer closure.

## External Blockers And Residual Facts

- Execution still requires explicit authorization because this review is
  planning-only.
- The bounded official transaction requires server/GPU/HSL access. Its absence
  limits closure to offline-ready but does not invalidate the local plan.
- READY does not prove code implementation, formal-route connectivity,
  checkpoint correctness, runtime correctness or policy quality.
- The LR values remain a user-selected bounded campaign intervention; policy
  improvement is falsifiable only after a fresh Stage-3 run.

## Required Execution Reviews

Within the same authorized one-shot unit:

1. run `construction_review` after the optimizer/config/checkpoint owner
   boundary is coherent and locally verified;
2. run focused `formal-runtime-audit` because checkpoint identity and official
   optimizer configuration change;
3. run `final_gate_review` on the complete diff and evidence;
4. fix and re-review any in-scope P0/P1 before the bounded live sentinel.
