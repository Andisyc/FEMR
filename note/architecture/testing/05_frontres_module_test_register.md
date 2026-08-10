# FrontRES Module Test Atlas

Status: TRAIN-v020 outer Scenario Replay cards confirmed and executed offline; Phase A passed.

Current result: `22 passed / 0 partial / 0 blocked`; the repository-wide
contract suite is `57 passed / 0 failed`. TEST-13--18 cover the confirmed
FRS-GAIN-v008 raw evidence boundary, per-attempt fixed symlog, M4 utility mean,
shared Actor/Critic utility, historical checkpoint-v14 evaluation and diagnostics.
TEST-19--22 cover stable Scenario identity, selection/priority, exact-one replay
commit and checkpoint-v15 persistence. The official bounded simulator transaction
remains the live gate before TRAIN-v020 long training.

Interactive page: `../05_frontres_module_test_atlas.html`

## Purpose

This Atlas is the human control surface for module/function correctness. It
does not prove formal-route connectivity or policy quality.

It also contains a compact `Formal Runtime Audit` stage-reading card. It explains
how the already-correct
modules are audited in two steps: Phase A checks method/code alignment, then
Phase B checks only the remaining live-dependent facts on the official route.
Module-semantic failures return to the existing Module Test Cards; policy
efficacy starts only after the formal route closes.

Each card is derived from the active contracts and the canonical module
registry. Its primary human view contains only one plain-language design rule
and a table with three columns:

```text
伪样本 | 正确结果 | 证明什么
```

The correct result must be worked out before the production module is run. It
may be a number, ordering, count, unchanged-state fact, or explicit rejection.
Generic responsibility, interface, evidence-tier, and lifecycle fields are
secondary engineering metadata; they cannot replace the concrete test cases.
A card must be confirmed before tests are written or run.

## Lifecycle

```text
proposed -> human-confirmed -> test-implemented -> executed -> evidence-linked
```

After confirmation, a failed test cannot change the card. The contradiction
must be classified as accepted-design error, production-code error, or test
translation error. Contract, public-interface, responsibility, or oracle
changes make affected evidence stale.

## Current Scope

The Atlas contains the original 18 formal runtime module families plus four
METHOD-v021 outer replay cards. Supporting GMT environment
reward and test-infrastructure modules remain outside the primary method test
surface.

## Execution Summary

| Status | Cards |
| --- | --- |
| passed | TEST-01--22 |
| not-run | none |
| partial | none |
| blocked | none |

The current detailed oracle and execution evidence is linked by
`../../frontres_core/plans/FRS-TRAIN-v020-outer-scenario-replay-one-shot-engineering-plan.md`;
the v017 value-scale ledger
remains historical evidence only. Formal runtime and policy-quality claims are
still separate gates.
