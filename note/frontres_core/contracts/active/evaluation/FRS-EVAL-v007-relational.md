---
contract_id: FRS-EVAL-v007
status: active-pre-training
effective_date: 2026-08-17
updated_date: 2026-08-17
supersedes: FRS-EVAL-v006-for-relational-route
scope: Read-only relational evidence interpretation and preference-edge diagnostics
---
# Relational Evaluation

Evaluation reports the per-Repair level, relation, evidence validity, edge
count, and Actor-credit incidence. It never converts `INCOMPARABLE` to a
numeric rank and never performs an optimizer or Replay mutation. Missing raw
Physics evidence is `TELEMETRY-GAP` or `INVALID`, not a proxy score.
