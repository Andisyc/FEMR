# FrontRES Module Test Atlas

Status: all 18 human-confirmed active cards pass after the TRAIN-v015 split-LR refresh.

Current result: `18 passed / 0 partial / 0 blocked`. The user-confirmed v015
optimizer/config/persistence questions are integrated into TEST-01, TEST-02,
TEST-15 and TEST-16 without adding a duplicate module family. E-FI-135 records
their execution and the 50/50 affected suite. The new campaign's bounded live
transaction is runtime-confirmed by E-FI-136; policy quality remains a separate
evidence gate.

Interactive page: `../05_frontres_module_test_atlas.html`

## Purpose

This Atlas is the human control surface for module/function correctness. It
does not prove formal-route connectivity or policy quality.

It also contains a compact `Formal Runtime Audit` stage-reading card. That card
does not add a nineteenth module test. It explains how the already-correct
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

The first Atlas contains the 18 formal runtime module families from
`architecture/01_repo_architecture.data.json`. Supporting GMT environment
reward and test-infrastructure modules remain outside the primary method test
surface.

## Execution Summary

| Status | Cards |
| --- | --- |
| passed | TEST-01--18 |
| partial | none |
| blocked | none |

The detailed claim, test, observed fact, and limitation ledger is
`../../testing/frontres_module_test_execution_2026-08-03.md`.
