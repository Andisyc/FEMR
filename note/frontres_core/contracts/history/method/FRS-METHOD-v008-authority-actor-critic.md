---
contract_id: FRS-METHOD-v008
status: stopped
effective_date: 2026-06-23
updated_date: 2026-07-13
supersedes: FRS-METHOD-v007
superseded_by: FRS-METHOD-v009
scope: Proposal-conditioned continuous authority actor-critic
---

# Authority Actor-Critic

This version restored rho as a continuous Stage-2 action:

```text
d_t   = detached HSL proposal
rho_t = authority actor(state, d_t)
Q(state, d_t, rho_t) -> K-step executable value
Delta SE_exec = rho_t * d_t
```

The critic was anchored by Noisy/no-write, Candidate/full-write, and behavior
rho returns. Actor updates were gated until critic endpoint ordering agreed with
rollout evidence. The design also introduced event-level authority, K-step
returns, and burst perturbation curriculum requirements.

The research direction was stopped because the experiment could not reliably
observe, optimize, and diagnose the full continuous authority surface. This
idea is not an active ablation roadmap and should not be revived or used to
interpret current code unless the user explicitly reopens the research
direction. Any residual code is historical technical debt, not retained method
authority.

Source: restricted compendium lines 284-755.
