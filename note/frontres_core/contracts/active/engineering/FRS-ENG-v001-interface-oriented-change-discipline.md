# FRS-ENG-v001: Interface-Oriented Change Discipline

```yaml
contract_id: FRS-ENG-v001
status: active
effective_date: 2026-07-29
updated_date: 2026-07-29
scope: repository-wide FrontRES/FEMR engineering changes, reviews, refactors, tests, persistence, evaluation, and deployment connectors
supersedes: none
authority_owner: workflow-governance
review_consumer: code-review-expert
```

## 1. Purpose And Authority

This contract prevents new FrontRES work from recreating large orchestration
modules, duplicate semantic owners, and wrapper-only abstractions. It governs
how code is changed; it does not change METHOD, GAIN, PPO, TRAIN, EVAL, the
Concept Figure, or the frozen MOSAIC host.

The active method/training/reward/optimization/evaluation contracts remain the
authority for behavior. This contract is the authority for repository-local
ownership, interfaces, dependency direction, change sequencing, and engineering
evidence. `workflow-governance` owns versioning and activation.
`code-review-expert` discovers and enforces this contract but cannot rewrite it
during review.

## 2. Applicability

The discipline is mandatory for every non-trivial change to FrontRES/FEMR
production code, tests, configuration, checkpointing, diagnostics, evaluation,
or deployment composition. A change is non-trivial when it changes an owner,
public shape, provenance, mutation, transaction, persistence, formal route, or
cross-module dependency.

Pure wording fixes, generated files, frozen third-party/vendor code, disposable
one-off scripts, and read-only archives may be marked `not-applicable`. The
exemption and reason must be explicit. MOSAIC-owned behavior is frozen unless
the user separately authorizes a host change.

## 3. Iron Law

A completed change must reduce or cap the number of facts a caller must know.

- Every semantic fact has one owner.
- A caller depends on a narrow public contract, not the owner's private state.
- An abstraction is justified only when it removes a named dependency,
  centralizes a named invariant, or stabilizes a named volatile boundary.
- A wrapper that leaves every old dependency and responsibility in place is not
  a completed refactor.
- Interface-oriented does not mean class-oriented. Prefer an immutable record,
  a small function, or a structural `Protocol` over an inheritance hierarchy.

## 4. Required Change Contract

Before editing a non-trivial boundary, record these facts in the current plan,
task, or implementation report. Do not create a new document for every small
change when an active plan already owns them.

1. Requested behavior and behavior that must remain unchanged.
2. The single semantic owner and its public consumers.
3. Public input/output record, tensor shape, identity, and failure behavior.
4. Dependency direction and forbidden dependencies.
5. State, mutation, transaction, and persistence boundary.
6. Legacy path to characterize, isolate, retire, or deliberately retain.
7. Focused tests and the strongest evidence tier required.
8. Rollback or stop condition.
9. Hotspot delta: which responsibilities and dependencies are added or removed.

If owner or method semantics are unresolved, stop before implementation. Do not
use an interface layer to conceal an unresolved design decision.

## 5. Ownership And Dependency Rules

### 5.1 One owner per fact

Configuration resolution, scenario identity, observation authority, transaction
commit, optimization, checkpoint identity, and evaluation evidence each have
one formal owner. Other modules may consume an immutable projection; they may
not recompute, mutate, cache independently, or silently repair the fact.

### 5.2 Dependency direction

Domain decisions point toward stable FrontRES contracts. Simulator, filesystem,
CLI, checkpoint transport, and reporting details point inward through narrow
adapters. High-level policy must not depend on a low-level dependency's private
attributes or concrete lifecycle.

The following are forbidden across an owner boundary unless a named, tested
compatibility adapter contains them:

- `_private` attribute access;
- `Any` used to erase a stable public contract;
- stable cross-layer payloads represented as open-ended `dict` objects;
- dynamic file imports used as ordinary dependency injection;
- passing simulator objects into deterministic Gain, PPO, schedule, or identity
  logic;
- silent fallback, padding, clamping, zero-fill, or identity synthesis.

Compatibility adapters must be named, isolated, characterized, and assigned a
retirement condition. They do not become new semantic owners.

### 5.3 Deep modules, narrow interfaces

Keep a module deep: its interface is smaller and more stable than its internal
implementation. Split by responsibility and change reason, not by arbitrary
line count or a preference for short functions. A split that makes callers
thread more parameters or understand more lifecycle details is a regression.

### 5.4 Functional core and imperative shell

Where practical, deterministic transformation, validation, scheduling,
identity, reduction, and decision logic form a functional core. IsaacLab,
filesystem, logging, CLI, and process control form the imperative shell.
Entrypoints and runners coordinate owners; they do not absorb domain rules.

### 5.5 Transaction and persistence boundary

Sealed transactions and checkpoint commits are Unit-of-Work boundaries. Partial
state, mixed identity, or uncommitted receipts must not escape. Save/resume must
restore the exact owner state defined by the active Training contract and fail
closed before mutation on incompatible identity.

## 6. Book-Derived Operational Gates

These gates are the working content extracted from the engineering sources.
They are not reading recommendations. A non-trivial review must name the gate
that explains the problem and the corresponding admissible transformation.

### 6.1 Refactoring change-smell gate

Use Fowler's change smells to locate the ownership error before proposing a
module split:

| Named smell | FEMR diagnostic | Required response |
| --- | --- | --- |
| `Divergent Change` | One module changes for unrelated reasons such as simulator capture, transaction commit, PPO math, checkpoint IO, and report formatting. | Separate by change reason under existing semantic owners; do not split into arbitrary helpers. |
| `Shotgun Surgery` | One semantic change requires coordinated edits across runner, sampler, diagnostics, checkpointing, and tests because no owner exposes the fact. | Move the fact to one owner and expose one projection before adding more consumers. |
| `Feature Envy` / `Inappropriate Intimacy` | A function mostly reads another owner's fields, especially `_private` state. | Move the decision to that owner or request a narrow immutable projection. |
| `Data Clumps` / `Primitive Obsession` | The same shape, identity, hash, cursor, mask, or receipt fields travel together as loose primitives. | Introduce a small immutable Value Object with validation; never hide the clump in an open `dict`. |
| `Message Chains` | Code traverses runner -> env -> manager -> term -> private payload to obtain a fact. | Add a Gateway/owner accessor at the first stable boundary and terminate the chain. |
| `Middle Man` | A wrapper forwards arguments and results without validation, translation, lifecycle, or dependency isolation. | Inline/delete it, or give it a real adapter responsibility and consumer contract. |
| `Speculative Generality` | A Protocol, base class, registry, factory, or plugin point has no present volatile boundary or real alternative consumer. | Reject or defer it; tests alone do not count as the second production need. |

Refactoring follows `small change -> compile/static check -> focused test`, but
the small internal rhythm does not create a separate user approval for every
helper. Behavior remains fixed unless another active contract authorizes a
semantic change.

### 6.2 Working Effectively with Legacy Code gate

Use Feathers' techniques when existing code cannot be tested or moved safely:

1. **Characterization Test**: record what the current route actually does, not
   what it ought to do. Cover representative input, failure behavior, extreme
   values, and invariants relevant to the change.
2. **Effect Sketch**: trace the proposed change through calls, mutations, files,
   optimizer/checkpoint state, and diagnostics. Stop at the smallest set of
   potentially affected facts.
3. **Pinch Point**: test at the narrowest shared point through which those
   effects pass. A convenient high-level test is not a pinch point if multiple
   unobserved routes bypass it.
4. **Seam and Enabling Point**: name both where behavior can be substituted and
   where real versus fake behavior is selected. A seam without an explicit
   enabling point is hidden monkeypatching, not a stable interface.
5. **Sprout Method/Class**: use only to place new tested behavior beside an
   untestable host when movement is currently unsafe. Record the integration and
   retirement condition; a sprout is not permission to leave duplicate owners.
6. **Wrap Method/Class**: use only when all relevant calls must be intercepted to
   add validation, telemetry, or lifecycle behavior. The wrapper must own that
   added responsibility and must not become a permanent pass-through.
7. **Extract Interface**, **Parameterize Method/Constructor**, and **Adapt
   Parameter**: use only to break a named hard dependency. The resulting
   interface contains exactly what the consumer uses.
8. **Scratch Refactoring**: use as disposable exploration to understand a
   tangle. Never merge its endpoint directly; discard it and repeat the chosen
   transformation with characterization evidence.

### 6.3 Component cohesion and dependency gate

Apply the Clean Architecture component principles to package/module placement:

- **Common Closure Principle (CCP)**: group code that changes for the same
  reason and at the same time. A `frontres_*` prefix is not sufficient cohesion.
- **Common Reuse Principle (CRP)**: do not force a consumer to depend on methods,
  data, or lifecycle it does not use. A Protocol is consumer-shaped, not an
  inventory of everything an implementation can do.
- **Acyclic Dependencies Principle (ADP)**: the production import/dependency
  graph is acyclic. Local imports, dynamic imports, and registries may not hide
  a cycle; dependency composition belongs at the outer boundary.
- **Stable Dependencies Principle (SDP)**: dependencies point toward the more
  stable contract/owner. Volatile IsaacLab, CLI, filesystem, and reporting code
  depend on stable FrontRES policy, identity, and transaction contracts, not the
  reverse.

Package movement is admitted only when it improves CCP/CRP without violating
ADP/SDP. File-size reduction alone is not evidence of improved components.

### 6.4 Boundary and testability gate

- **Dependency Rule**: source dependencies point toward higher-level FrontRES
  policy. Boundary records contain simple tensors/scalars/identities, never
  IsaacLab manager or sensor objects.
- **Humble Object**: keep simulator reads, CLI parsing, filesystem writes, and
  framework callbacks minimal. Extract computation and validation into a
  deterministic owner that can be tested without the framework.
- **Screaming Architecture**: top-level FrontRES structure and public names must
  reveal scenario, transaction, Gain/PPO, checkpoint, and evaluation use cases,
  not generic `manager`, `utils`, or framework vocabulary.
- **Main Component / Composition Root**: create and connect concrete adapters in
  `train.py`, runner setup, or another declared outer entrypoint. Domain owners
  receive dependencies explicitly and do not locate or dynamically import them.

### 6.5 Python and enterprise pattern admission

Architecture Patterns with Python and EAA patterns are conditional tools:

| Pattern | Admit only when | FEMR meaning |
| --- | --- | --- |
| `Service Layer` | Two or more entrypoints need the same use-case orchestration, or orchestration is duplicated in adapters. | A thin Stage-3/evaluation use-case function coordinates owners; it contains no Gain/PPO/simulator math. |
| `Unit of Work` | Multiple state changes must succeed or fail atomically. | Sealed transaction and checkpoint commit use explicit commit, rollback/fail-closed by default, and one mutation exit. |
| `Aggregate` | A cluster has invariants that must be protected as one consistency boundary. | Scenario/transaction identity and attempts are mutated only through the aggregate owner; consumers receive sealed projections. |
| `Gateway` | Code accesses an external system/resource with volatile API. | IsaacLab sensors/managers, filesystem, checkpoint transport, and external evaluation carriers sit behind narrow gateways. |
| `Repository` | Persistent data access genuinely needs a replaceable collection-like abstraction. | Cache/checkpoint artifact lookup may qualify; ordinary in-memory tensor access does not. |
| `Value Object` | Equality is defined by complete immutable value, not object identity. | Shape/layout identity, scenario identity, receipt, and evidence projections use validated immutable records. |
| `Composition Root` | Concrete real/fake adapters must be selected. | Manual dependency injection occurs once at the outer entrypoint; no DI framework or service locator. |

`Message Bus`, domain events, CQRS, plugin registries, factories, and class
hierarchies are not default FEMR architecture. Admit one only when an actual
many-handler/asynchronous/replaceable requirement exists and a simpler direct
call cannot preserve the same boundary.

### 6.6 Maintainability metric translation

Building Maintainable Software proposes concrete limits: 15-line units, at
most four branch points (McCabe 5), and at most four parameters. FEMR uses them
as investigation prompts, not absolute rules:

- Over 15 lines: ask whether one unit contains separable decisions. Keep a
  longer coherent calculation when extraction would scatter tensors or hide the
  mathematical sequence.
- Over four branch points: identify whether independent policies are mixed.
  Prefer guard clauses or named policy functions; do not replace explicit logic
  with a speculative hierarchy.
- Over four parameters: check for a Data Clump or missing Value Object. Do not
  lower the count by passing `dict`, `Any`, or a whole runner.
- `Write Code Once`: remove duplicated knowledge/invariants, not merely similar
  syntax that represents different owners.
- `Separate Concerns` and `Loose Coupling`: evaluate change reason, incoming
  interface surface, and outgoing dependency direction, not file count alone.
- `Balanced Components` and `Keep Codebase Small`: delete obsolete paths and
  avoid micro-modules; neither one giant component nor one-file-per-helper is
  acceptable by default.
- `Automate Tests`: retain focused owner/consumer/negative/persistence tests as
  reusable regression assets rather than disposable migration probes.

## 7. Hotspot Policy

Metrics trigger inspection; they do not prove bad design.

- Large files, high branch/parameter counts, broad fan-in/fan-out, frequent
  co-change, and static hotspot scores trigger the named gates in Section 6.
- A file above 1,000 lines or with more than three suspected change reasons gets
  a mandatory `Divergent Change` and `Pinch Point` audit before adding behavior.
- A file above 2,000 lines may still be a coherent deep module, but new behavior
  requires proof that CCP/CRP remain intact and Shotgun Surgery is not growing.
- Repeated edits across runner, sampler, diagnostics, checkpointing, and tests
  for one fact trigger `Shotgun Surgery` even when every file is short.

An exception must name why keeping the hotspot is safer, what tests contain the
risk, and when the exception expires. Line count alone never requires a rewrite.

## 8. Safe Refactor Sequence

Refactoring preserves externally accepted behavior unless an active method
contract explicitly changes it.

1. Write a Characterization Test for the accepted route.
2. Draw the Effect Sketch and select the Pinch Point.
3. Name the Seam and Enabling Point.
4. Freeze owner, public contract, dependency direction, and forbidden paths.
5. Apply the smallest named transformation that breaks the dependency.
6. Move one CCP-coherent responsibility behind that seam.
7. Switch all in-scope consumers and prove CRP-shaped connectivity.
8. Delete or explicitly isolate the superseded Sprout/Wrap/legacy path.
9. Re-run negative tests and an ADP/SDP dependency review.
10. Review the final dependency and responsibility delta, not only file count.

Avoid rewrite-all migrations. Small, test-backed transformations are preferred,
but the user-facing unit should remain coherent rather than being split into
avoidable approval gates.

## 9. Evidence Matrix

Every non-trivial change selects the applicable rows and states why omitted rows
do not apply.

| Evidence | Required proof |
| --- | --- |
| Characterization | Existing accepted behavior, invariants, Seam, and Enabling Point are observable before movement. |
| Effect boundary | Effect Sketch and Pinch Point justify the selected test and movement boundary. |
| S1 owner contract | Public shape, identity, invariants, deterministic behavior, and fail-closed cases. |
| S2 connectivity | Real consumer reaches the single owner without fallback, private access, or duplicate recomputation. |
| Negative boundary | Forbidden legacy/private/mixed route is rejected rather than silently tolerated. |
| S3 persistence | Checkpoint/save/resume preserves exact identity and mutation atomicity when state is touched. |
| S4 runtime | Used only when deterministic evidence cannot establish the relevant simulator or formal runtime fact. |

Tests assert behavior and authority, not merely importability or successful
startup. New public interfaces require at least one real consumer test. A legacy
refactor requires characterization before deletion. A persistence change
requires pre-mutation rejection of incompatible identity.

## 10. Review And Closure

`construction_review` applies at the first coherent owner/interface boundary.
`final_gate_review` applies before closing a non-trivial unit. P0/P1 findings
block closure. P2 findings are fixed in scope or recorded with a concrete owner
and retirement condition; P3 is optional.

The Step End Report must state:

- owner and public interface after the change;
- dependencies/responsibilities removed and added;
- legacy path deleted or intentionally isolated;
- tests and evidence tiers actually exercised;
- unconfirmed runtime or policy-quality facts;
- hotspot delta and remaining debt.

Do not call a refactor complete when the old hotspot retains the behavior and a
new wrapper merely forwards into it.

## 11. Forbidden Failure Modes

- Applying design patterns because a book or checklist names them.
- Reporting only a generic `SOLID` violation when a named change smell,
  component principle, seam, or pattern-admission decision is available.
- Creating `Manager`, `Utils`, `Common`, or service dumping grounds.
- Wide interfaces whose consumers use only a small subset.
- A second cache, identity resolver, lifecycle flag, or mutation owner for an
  existing semantic fact.
- Hiding dependency problems behind `Any`, open dictionaries, dynamic imports,
  or private attribute access.
- Moving code into many shallow modules while increasing parameter tracing.
- Changing MOSAIC host behavior for FrontRES convenience without explicit
  authorization.
- Claiming completion from static structure without consumer, negative, and
  persistence evidence where applicable.

## 12. Source Adaptation

This discipline synthesizes the local engineering reading set documented in
`note/software_engineering/frontres_engineering_discipline_sources.md`.
Books are advisory sources, not repository authority. The detailed matrix maps
each named framework to its FEMR trigger, transformation, and rejection case.
When sources conflict, this contract favors behavior preservation, deep modules,
traceable parameters, explicit owners, and the smallest boundary that removes a
real dependency.

## 13. Version And Stop Conditions

Version this contract when repository-wide ownership rules, dependency
direction, mandatory evidence, hotspot policy, or MOSAIC/FrontRES authority
changes. Wording, links, and evidence references may update `updated_date`
without a version bump.

Stop implementation and return to governance when:

- two modules claim the same semantic owner;
- an interface cannot be defined without method ambiguity;
- the change requires actor privilege, method, Gain/PPO, observation, or
  checkpoint semantics not authorized by active contracts;
- the only implementation requires changing frozen MOSAIC host behavior;
- the legacy route cannot be characterized or isolated safely;
- a hotspot would gain a new semantic responsibility without an explicit
  exception;
- required consumer, negative, transaction, or persistence evidence cannot be
  constructed.

No Concept Figure or Architecture Atlas update is required for this activation:
the contract governs change discipline and does not alter current method blocks
or runtime ownership. Future implementation changes must update Architecture
when their owner or runtime interface actually changes.
