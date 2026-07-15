---
contract_id: FRS-METHOD-v007
status: superseded
effective_date: 2026-06-23
updated_date: 2026-07-13
supersedes: FRS-METHOD-v006
superseded_by: FRS-METHOD-v008
scope: Endpoint-supervised proposal-conditioned acceptance
---

# Proposal-Conditioned Acceptance

This version compressed continuous authority into a two-stage acceptance
problem:

```text
Stage 1 -> detached Clean-oriented Delta SE proposal
Stage 2 -> acceptance conditioned on state/history and that proposal
evidence  -> full-write Candidate versus no-write Noisy/GMT endpoints
```

Intermediate output values meant soft acceptance confidence, not proof of an
optimal continuous repair fraction. This matched the available endpoint
evidence and explicitly avoided Stable Frame search or a ray critic.

It was superseded when the method restored rho as a continuous action learned
through an authority critic and K-step return.

Source: restricted compendium lines 756-906.
