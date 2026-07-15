---
contract_id: FRS-METHOD-v001
status: superseded
effective_date: "before 2026-06-09; exact date unconfirmed"
updated_date: 2026-07-13
supersedes: none
superseded_by: FRS-METHOD-v002
scope: FrontRES HSL proposal and per-axis rho acceptance
---

# HSL Proposal With Per-Axis Rho Acceptance

FrontRES used one shared policy with a six-dimensional HSL repair proposal and
a six-dimensional dynamics-aware acceptance vector:

```text
Delta SE_HSL = proposal direction and magnitude
rho[6]       = per-axis write authority
Delta SE_exec = rho * Delta SE_HSL
```

HSL owned geometric restoration. PPO/HRL owned only rho and was forbidden from
changing proposal direction. A state-router alpha and Stable Frame route were
considered as auxiliary structure.

This version established proposal/authority separation, but rho had to express
no-op, repair strength, state risk, and fallback behavior at once. The later
Stable-to-Repair version changed rho's coordinate system.

Source: restricted compendium lines 1543-1717 and the 2026-06-09 frontier/HRL
discussion at lines 2940-3107.
