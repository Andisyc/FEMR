---
contract_id: FRS-METHOD-v004
status: rejected
effective_date: 2026-06-11
updated_date: 2026-07-13
supersedes: FRS-METHOD-v003
superseded_by: FRS-METHOD-v005
scope: Joint alpha-rho rollout credit assignment
---

# Structured Joint Alpha-Rho RL

This version kept alpha and rho as separate conceptual heads but trained their
joint executed action from Projected-vs-Noisy rollout advantage:

```text
joint action = (alpha, rho)
L_joint = -A_projected * log pi(alpha, rho | state)
```

It rejected detached pseudo-label ownership because Projected behavior was
caused jointly by alpha and rho.

The branch was rejected after alpha PPO credit re-entangled state routing with
repair retention. The successor returned alpha to supervised state
classification and restricted HRL credit to rho.

Source: restricted compendium lines 3780-3928.
