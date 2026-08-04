# FRS-ENG-v001 Change Discipline Checklist

Use this checklist for each non-trivial FrontRES/FEMR change. It is a reusable
gate, not a task-status ledger. Mark omitted rows `N/A` with a reason.

## Applicability

- [ ] The change is classified as non-trivial or explicitly `not-applicable`.
- [ ] Frozen MOSAIC/vendor/generated/history paths are excluded unless separately authorized.
- [ ] Required method/training/reward/optimization/evaluation contracts are named.

## Change Contract

- [ ] Requested behavior is stated.
- [ ] Preserved behavior is stated.
- [ ] One semantic owner and all public consumers are named.
- [ ] Public input/output shape, identity, provenance, and failure behavior are stated.
- [ ] Dependency direction and forbidden dependencies are stated.
- [ ] State, mutation, transaction, and persistence boundary are stated.
- [ ] Legacy path and its delete/isolate condition are stated.
- [ ] Tests, evidence tier, rollback, and stop condition are stated.
- [ ] Hotspot responsibility/dependency delta is stated.

## Construction

- [ ] `Divergent Change` and `Shotgun Surgery` were checked against the actual co-change surface.
- [ ] `Feature Envy`, `Inappropriate Intimacy`, `Message Chains`, and `Middle Man` were checked at owner boundaries.
- [ ] Repeated primitives were evaluated as `Data Clumps`/Value Objects; no open `dict` substitute was used.
- [ ] Every new abstraction passed the `Speculative Generality` rejection test.
- [ ] The interface removes a named dependency or centralizes a named invariant.
- [ ] No duplicate semantic owner, cache, resolver, lifecycle flag, or mutation path is added.
- [ ] Entrypoint/runner remains orchestration-only.
- [ ] Deterministic domain logic is isolated from simulator/filesystem/process control where practical.
- [ ] No cross-owner `_private`, unjustified `Any`, open stable `dict`, or dynamic-import dependency is added.
- [ ] No silent fallback, padding, clamping, zero-fill, or identity synthesis is added.
- [ ] Public records are immutable or mutation authority is explicit.
- [ ] The module remains deep; parameter tracing does not expand after splitting.
- [ ] Files above 1,000/2,000 lines obey the hotspot review/exception rule.

## Component And Pattern Admission

- [ ] CCP: colocated code changes for the same reason and at the same time.
- [ ] CRP: each consumer depends only on the public surface it actually uses.
- [ ] ADP: the production dependency graph remains acyclic without hidden local/dynamic import cycles.
- [ ] SDP/Dependency Rule: dependencies point toward stable FrontRES policy, not volatile framework details.
- [ ] Humble Object: framework/IO shell is minimal and decision logic is deterministic/testable.
- [ ] Composition Root: real/fake dependencies are selected once at a declared outer entrypoint.
- [ ] Service Layer/Gateway/Repository/Aggregate/Unit of Work was added only after its admission condition was demonstrated.
- [ ] Message Bus/CQRS/plugin/factory/hierarchy was rejected unless an actual asynchronous or replaceable requirement exists.

## Migration And Legacy

- [ ] A Characterization Test records actual behavior, failures, extremes, and relevant invariants.
- [ ] An Effect Sketch identifies affected calls, mutations, persistence, and diagnostics.
- [ ] The selected Pinch Point covers the effects without bypass routes.
- [ ] The Seam and its Enabling Point are both named.
- [ ] The smallest useful dependency-breaking technique is selected explicitly.
- [ ] Any Sprout/Wrap path has an integration and retirement condition.
- [ ] Scratch Refactoring output was discarded and not merged as production structure.
- [ ] One coherent responsibility is moved at a time.
- [ ] All in-scope consumers use the new owner.
- [ ] Superseded path is deleted or explicitly isolated with a retirement condition.
- [ ] Negative tests prove the legacy/private route cannot re-enter.

## Evidence

- [ ] Characterization evidence exists when legacy behavior is moved.
- [ ] S1 owner contract covers shape, identity, invariants, and fail-closed cases.
- [ ] S2 connectivity uses a real consumer rather than an import-only test.
- [ ] Negative boundary tests reject forbidden dependencies and mixed identity.
- [ ] S3 persistence proves pre-mutation rejection and exact restore when state is touched.
- [ ] S4 runtime is used only for facts unavailable offline.
- [ ] Units above 15 lines, four branch points, or four parameters were inspected without mechanical splitting.
- [ ] `python -m py_compile` or the repository-equivalent static check ran after Python edits when practical.

## Review And Closeout

- [ ] `construction_review` checked the first coherent owner/interface boundary.
- [ ] `final_gate_review` checked the complete diff and evidence.
- [ ] P0/P1 findings are closed.
- [ ] P2 findings are fixed or have an explicit owner and retirement condition.
- [ ] Step End Report lists owner/interface, dependency delta, legacy status, tests, unconfirmed facts, and hotspot debt.
- [ ] Architecture was updated if runtime ownership/interface changed, or explicitly marked unnecessary.
- [ ] Concept Figure was changed only if method semantics changed under user authority.
