# Engineering Plan Review: TRAIN-v013 Migration

Mode: `engineering_plan_review`

Verdict: `READY`

Discipline: active — `FRS-ENG-v001`.

## Accepted Behavior / Non-Scope

The review accepts TRAIN-v013's fixed outer K/M schedule, per-K lower-to-higher
DR curriculum, four-class 20/30/40/10 sampling relative to explicit `d_cap`,
committed transition restart, same-Critic recalibration, exact-M transaction and
checkpoint-v8 identity. It preserves METHOD-v017, GAIN-v007, PPO-v005,
EVAL-v004, HSL-v1, actor/GMT observation and action authority. Simulator,
training, live, policy-quality and deployment are outside Step 1.

## Boundary Reviewed

The reviewed boundary is:

```text
explicit config
-> existing frontres_segment_warmup schedule owner
-> immutable stage identity
-> existing transaction Aggregate and perturbation materializer
-> grouped exact-one commit
-> existing checkpoint Gateway and read-only telemetry
```

The plan consumes current white-box facts from `frontres_interfaces.py`,
`frontres_segment_warmup.py`, `train.py`,
`frontres_segment_formal_transaction.py`, the existing perturbation owner,
`frontres_checkpointing.py` and training telemetry. Reachability is not inferred
from these reads.

## Findings

- P0: none.
- P1: none after rebase. The former authority conflict between active v012
  frozen `g_K` and the confirmed per-K DR restart is resolved by contract
  versioning rather than hidden inside the plan.
- P2: exact first-campaign starting distributions, advance-rule IDs and advance
  update counts remain empirical launch parameters. This does not block the
  generic owner/interface migration because the plan requires them to be
  explicit, versioned and fail-closed. It blocks only a later live command until
  the user confirms concrete values.
- P3: serialized checkpoint naming retains the established `frontres-v015-*`
  wire prefix. The plan correctly treats this as compatibility identity, not an
  active public-code version label.

Named discipline gates applied:

- `Divergent Change` / CCP: schedule resolution remains in the existing
  K-stage owner; checkpointing gains persistence fields but no schedule policy.
- `Shotgun Surgery` / Pinch Point: one immutable resolved identity supplies
  transaction, sampler, telemetry and persistence instead of repeated local
  inference.
- `Feature Envy` / Dependency Rule: no runner-private traversal or simulator
  state enters the deterministic schedule owner.
- Characterization Test / Seam / Enabling Point: old K/M arithmetic,
  critic-only freeze and pre-mutation rejection are preserved through hand-built
  stage specs selected at the existing Composition Root.
- CCP/CRP/ADP/SDP: consumers receive only the resolved immutable identity;
  dependencies point away from deterministic policy toward runner/IO adapters.

## Flow / Lifecycle Gaps

None in the plan. Creation, seal, commit, abort, retry, progress advance,
checkpoint save and resume are explicit. Partial or mixed transaction state
cannot advance curriculum. Diagnostics remain read-only.

## Pattern Admissions Accepted / Rejected

- accepted: reuse the existing transaction Aggregate and checkpoint Gateway;
- accepted: refine the existing immutable K-stage identity because it removes
  repeated primitive schedule facts from callers;
- rejected: new Service Layer, manager, registry, Protocol, wrapper, parallel
  curriculum class or online controller; none removes a named dependency.

## Proof-Route Gaps

No planning gap. Step 1 includes owner, negative, official offline connectivity,
persistence, module regression, construction review, formal-runtime Phase A and
final review. S4 remains deliberately outside the plan unit because it consumes
GPU/simulator authority.

## External Method / Formal / Runtime Blockers

- the three affected Module Test Cards require human review before test/source
  execution;
- current code is still v012/checkpoint-v7 and therefore contract-mismatched;
- formal connectivity, checkpoint-v8 behavior and live DR values remain
  unconfirmed until their owning evidence gates execute.

These are execution/evidence gates, not reasons to revise the owner plan.

## Required Plan Revisions

None. The plan is executable after the affected Test Cards are human-confirmed.

## Residual Facts Unconfirmed Until Implementation

- whether the existing stage identity can carry every v013 field without an
  admitted in-place schema refinement;
- whether all formal transaction/checkpoint consumers can remove v012/g_K
  assumptions without a new semantic choice;
- checkpoint-v8 atomic roundtrip and v7 pre-mutation rejection;
- official-route absence of the retired episode-length/frontier controller;
- real simulator distribution and policy quality.

`READY` proves only that the plan has a maintainable owner/interface/lifecycle
route. It does not prove implementation, formal integration or runtime quality.
