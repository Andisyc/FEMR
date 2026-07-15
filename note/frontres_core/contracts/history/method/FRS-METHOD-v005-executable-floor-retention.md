---
contract_id: FRS-METHOD-v005
status: superseded
effective_date: 2026-06-12
updated_date: 2026-07-13
supersedes: FRS-METHOD-v004
superseded_by: FRS-METHOD-v006
scope: Executable-floor state router and constrained repair retention
---

# Executable-Floor Router And Repair Retention

This version separated the two learning questions:

```text
alpha -> supervised classifier: is Noisy/GMT leaving the executable floor?
rho   -> HRL policy: how much HSL Repair can remain executable?
```

Noisy, Stable, and Repair still parameterized the projected reference, but
alpha no longer received PPO credit. A unified executable floor, calibrated
from GMT frontier evidence, supplied the alpha label and rho floor penalty.

This design repaired alpha/rho credit ownership but still depended on a
hand-structured Stable Frame and constrained rho geometry. Conditional HRL
subsequently compressed the method around repair authority and no-regret
evidence.

Source: restricted compendium lines 1718-1870, with related June 12 fixes and
oracle/split-advantage experiments later in the compendium.
