# FrontRES Engineering Discipline Source Matrix

This note records how general software-design sources were adapted into
`FRS-ENG-v001`. It is advisory provenance, not an active contract. The active
contract wins whenever wording here conflicts with it.

## Sources And Local Adaptation

| Source | Extracted principle | FEMR adaptation | Deliberate limit |
| --- | --- | --- | --- |
| Martin Fowler, *Refactoring: Improving the Design of Existing Code* | `Divergent Change`, `Shotgun Surgery`, `Feature Envy`, `Data Clumps`, `Primitive Obsession`, `Message Chains`, `Middle Man`, `Speculative Generality`; small behavior-preserving refactor rhythm. | Diagnose the ownership failure by change surface before extracting modules; use Value Objects and owner projections instead of loose identity primitives or private traversal. | A smell selects investigation and transformation; it does not automatically require a class or split. |
| Michael Feathers, *Working Effectively with Legacy Code* | `Characterization Test`, `Effect Sketch`, `Pinch Point`, `Seam`, `Enabling Point`, `Sprout Method/Class`, `Wrap Method/Class`, `Extract Interface`, `Parameterize Method/Constructor`, `Adapt Parameter`, `Scratch Refactoring`. | Protect existing formal routes, locate the narrow test boundary, break one hard simulator/runner dependency, and give every transitional seam a retirement condition. | Scratch output is disposable; Sprout/Wrap cannot become permanent duplicate ownership. |
| Harry Percival and Bob Gregory, *Architecture Patterns with Python* | Repository, Service Layer, Unit of Work, Aggregate, explicit dependencies, manual DI, Composition Root, high-gear service tests. | Use-case orchestration is thin; sealed transaction/checkpoint is explicit-commit UoW; scenario/transaction is the consistency boundary; real/fake adapters are selected at the outer entrypoint. | Repository, Message Bus, events, CQRS, and DI frameworks are conditional, not a package template. |
| Martin Fowler, *Patterns of Enterprise Application Architecture* | Service Layer, Unit of Work, Gateway, Value Object, Separated Interface, Plugin. | Gateways contain IsaacLab/filesystem/checkpoint volatility; Value Objects carry immutable identity; Service Layer exists only for shared use-case orchestration. | Web/database patterns are not copied into robotics training by analogy alone. |
| Robert C. Martin, *Clean Architecture* | CCP, CRP, ADP, SDP, Dependency Rule, Humble Object, Screaming Architecture, Main Component. | Group by change reason, shape consumer interfaces, prohibit cycles, point inward to stable FrontRES contracts, keep framework shells humble, compose dependencies at `main`. | Avoid ceremonial concentric layers and abstract factories around stable functions. |
| Joost Visser et al., *Building Maintainable Software* | 15-line units, at most four branch points/McCabe 5, at most four parameters, Write Code Once, Separate Concerns, Loose Coupling, Balanced Components, Keep Codebase Small, Automate Tests. | Use numerical rules as probes for mixed decisions, Data Clumps, and coupling; retain focused regression assets and delete obsolete routes. | Thresholds never override mathematical continuity, deep-module value, or parameter traceability. |
| *Clean Code* and pattern catalogs | Clear names, cohesive behavior, and pattern vocabulary can improve local readability. | Use names and patterns only when they expose intent or isolate a real variation. | Short-function dogma and pattern-for-pattern's-sake are rejected when they harm parameter traceability or module depth. |

## Local Source Files

- `/Users/chengyuxuan/ArtiIntComVis/Clean-Code-Collection-Books/重构-改善既有代码的设计Refactoring Improving the Design of Existing Code.pdf`
- `/Users/chengyuxuan/ArtiIntComVis/Clean-Code-Collection-Books/Working Effectively with Legacy Code.pdf`
- `/Users/chengyuxuan/ArtiIntComVis/Clean-Code-Collection-Books/Clean Architecture A Craftsman's Guide to Software Structure and Design.pdf`
- `/Users/chengyuxuan/ArtiIntComVis/Clean-Code-Collection-Books/OReilly.Building.Maintainable.Software.Java.Edition.2016.1.pdf`
- `note/software_engineering/code_design_reading.html`

Online primary references named in the reading note:

- https://www.cosmicpython.com/book/preface.html
- https://martinfowler.com/books/refactoring.html
- https://martinfowler.com/books/eaa.html

## Conflict Resolution

1. Active FEMR contracts and user authority outrank all books.
2. Preserve behavior before moving ownership.
3. Prefer a deep module and narrow interface over many shallow helpers.
4. Prefer a concrete immutable record or function over a hierarchy when both
   express the same boundary.
5. Add a pattern only when it removes a named dependency, stabilizes a volatile
   boundary, or makes a transaction explicit.
6. Treat line counts and generic quality scores as prompts for white-box review,
   not verdicts.

## Book-Derived Review Decision Table

| Observed symptom | Named question | Admissible next move | Reject |
| --- | --- | --- | --- |
| One large file changes for unrelated tasks | `Divergent Change` + CCP | Move one same-reason responsibility after characterization. | Splitting by arbitrary helper size. |
| One fact changes across many files | `Shotgun Surgery` + Pinch Point | Establish one owner/projection at the shared effect funnel. | Adding another compatibility flag in every consumer. |
| Caller traverses foreign internals | `Feature Envy`, `Message Chains`, Dependency Rule | Owner accessor or Gateway returning a simple immutable record. | `_private`, whole-runner parameters, or `Any`. |
| Several identifiers always travel together | `Data Clumps`, Value Object | Validated frozen record with complete equality/hash. | Open `dict` or positional tuple with hidden layout. |
| Existing route is hard to test | Seam + Enabling Point + Humble Object | Separate deterministic decision from minimal framework shell. | Global monkeypatch with no production selection owner. |
| Proposed refactor has broad uncertain effects | Effect Sketch + Pinch Point | Test the narrow shared effect boundary, then move one responsibility. | Broad end-to-end startup test as the only proof. |
| New wrapper or protocol is proposed | Middle Man + Speculative Generality + CRP | Admit only with validation/translation/lifecycle or real volatile alternatives. | Pass-through wrapper or inventory-shaped interface. |
| Multiple writes must commit together | Unit of Work + Aggregate | Explicit commit, rollback/fail-closed default, one consistency owner. | Partial save, hidden mutation, or multiple commit exits. |
| External framework API leaks inward | SDP + Gateway + Humble Object | Adapter at the boundary, simple data inward. | Simulator objects inside Gain/PPO/identity logic. |
| Dependency cycle appears | ADP + Composition Root | Invert one edge and compose concrete dependencies at the outer entrypoint. | Local/dynamic imports that merely hide the cycle. |
