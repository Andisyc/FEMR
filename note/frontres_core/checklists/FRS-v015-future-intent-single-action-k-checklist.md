# FRS-v015 One-Shot Acceptance Checklist

Status: active, volatile acceptance surface. Updated: 2026-07-22.

Plan: `../plans/FRS-v015-future-intent-single-action-k-engineering-plan.md`

Rows below are acceptance assertions, not execution steps. Evidence tiers and
source owners remain recorded in the ledger but do not create extra approval
gates.

| Unit | Acceptance assertion | Status | Evidence / trigger |
| --- | --- | --- | --- |
| Foundation | v015 semantics, q29 carrier, 928/158/770 authority, two roles, one-action K, v003 Gain, grouped exact-one update, persistence, and formal connectors exist | completed | `E-FI-0--E-FI-61` |
| Governance | one-shot planning compression is active; Q1--Q6 are not mandatory prerequisites | completed | `E-FI-62--E-FI-63` |
| G5-E0 route | official Stage3 config reaches explicit HSL-v1 initialization and sealed grouped transaction without legacy fallback | ready | inspect and repair inside G5-E0 |
| G5-E0 action | policy action is finite `[B,6]`; valid-row mask and scenario/transaction identities are present | ready | bounded smoke telemetry |
| G5-E0 evidence | v003 Gain components, return, advantage, gradient, group mass, and exact-one counters are finite and coherent | ready | bounded smoke telemetry |
| G5-E0 persistence | complete transaction produces a committed Stage3-v015 checkpoint | ready | actual save sentinel/path |
| G5-E0 closure | no Traceback, NaN/Inf, partial transaction, Clean actor input, later FEMR action, or unresolved visible bug remains after one internal repair cycle | ready | focused checks plus bounded log |
| Runtime audit | smallest official owner/shape/checkpoint audit only if a visible runtime contradiction survives local repair | dormant conditional | `formal-runtime-audit` trigger |
| Quality audit | smallest causal quality probe only if trained metrics are no-op, regressing, harmful, or contradictory | dormant conditional | `policy-quality-audit`; prior `E-FI-58--E-FI-61` reusable |
| X1 experiments | formal training budget, seeds, checkpoint trajectory, paired composition, and paper artifacts | pending costly boundary | authorized only after G5-E0 passes |

## Pass Rule

A clean G5-E0 bounded smoke closes engineering. It does not need to prove final
policy quality or paper-level statistical significance. Those belong to X1.

## Fail Rule

An ordinary assertion failure stays inside G5-E0 for diagnosis and repair. Stop
only for a method choice, new costly/destructive authority, an unresolved
official-route contradiction after one repair cycle, or metrics that activate a
conditional quality audit.
