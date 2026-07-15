---
contract_id: FRS-METHOD-v006
status: superseded
effective_date: 2026-06-19
updated_date: 2026-07-13
supersedes: FRS-METHOD-v005
superseded_by: FRS-METHOD-v007
scope: Conditional HRL over Clean-oriented repair authority
---

# Conditional HRL Repair Authority

This version identified one central variable:

```text
repair authority = how much of a Clean-oriented repair should be written under
the current state?
```

Clean provided direction, Noisy/GMT provided the no-op baseline, rollout
comparison supplied no-regret evidence, and rho represented authority. Sample
classification became a conservative boundary prior rather than direct
advantage or a hard supervised rho target.

June 21 refinements introduced region-direct authority loss, underwrite evidence,
and logit-level rho repair loss. These changed optimization strength but not the
method variable.

The version was superseded because endpoint evidence could reliably identify
accept/reject, but not an exact optimal continuous rho.

Source: restricted compendium lines 907-1499.
