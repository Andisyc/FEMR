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
| G5-E0 route | official Stage3 config reaches explicit HSL-v1 initialization and sealed grouped transaction without legacy fallback | runtime-confirmed | `E-FI-64`; 928/158/770, 2 Segments x 2 attempts, K=8, grouped exact-one |
| G5-E0 action | policy action is finite `[B,6]`; valid-row mask and scenario/transaction identities are present | runtime-confirmed | `E-FI-64`; abs mean `0.00409`, max `0.01545`, four valid identified rows |
| G5-E0 evidence | v003 Gain components, return, advantage, gradient, group mass, and exact-one counters are finite and coherent | stopped: harmful/contradictory | `E-FI-64`; harm `0.75`, all advantages negative, Physics `0/4`, gradient active and clipped |
| G5-E0 persistence | complete transaction produces a committed Stage3-v015 checkpoint | runtime-confirmed | `E-FI-64`; `model_1.pt`, exact-one receipt and save sentinel |
| G5-E0 closure | no Traceback, NaN/Inf, partial transaction, Clean actor input, later FEMR action, or unresolved visible bug remains after one internal repair cycle | partial: quality stop | route/persistence pass; learned metrics activate conditional quality audit before X1 |
| Runtime audit | smallest official owner/shape/checkpoint audit only if a visible runtime contradiction survives local repair | dormant conditional | `formal-runtime-audit` trigger |
| Quality audit | smallest causal quality probe only if trained metrics are no-op, regressing, harmful, or contradictory | active conditional | inspect Physics evidence and Gain->return->advantage causality from `E-FI-64`; prior `E-FI-58--E-FI-61` reusable |
| Physics evidence closure | one-action-K seals paired survival/ZMP/contact with a common valid mask; missing evidence fails closed; raw/scaled credit diagnostics are row-aligned | contract-confirmed; live pending | `E-FI-65`; unequal/tie/missing/mask/permutation and formal transaction contracts pass |
| X1 experiments | formal training budget, seeds, checkpoint trajectory, paired composition, and paper artifacts | blocked | requires resolution of the `E-FI-64` Physics/harm contradiction |

## Pass Rule

A clean G5-E0 bounded smoke closes engineering. It does not need to prove final
policy quality or paper-level statistical significance. Those belong to X1.

## Fail Rule

An ordinary assertion failure stays inside G5-E0 for diagnosis and repair. Stop
only for a method choice, new costly/destructive authority, an unresolved
official-route contradiction after one repair cycle, or metrics that activate a
conditional quality audit.
