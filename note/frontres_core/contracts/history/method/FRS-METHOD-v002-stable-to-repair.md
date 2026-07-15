---
contract_id: FRS-METHOD-v002
status: superseded
effective_date: 2026-06-10
updated_date: 2026-07-13
supersedes: FRS-METHOD-v001
superseded_by: FRS-METHOD-v003
scope: Stable-to-Repair authority parameterization
---

# Stable-to-Repair Rho Parameterization

This version changed rho from Noisy-to-Repair scaling into interpolation
between a deterministic Stable Frame and the HSL Repair endpoint:

```text
rho = 0 -> Stable Frame
rho = 1 -> HSL Repair
Projected = Stable + rho * (Repair - Stable)
```

The goal was to give low authority a physically executable meaning instead of
plain no-op. HSL still owned Repair; HRL owned only interpolation authority.

The design was superseded because forcing every low-rho sample toward Stable
damaged states where Noisy/GMT was already executable. Tri-Anchor restored Noisy
as an explicit fallback.

Source: restricted compendium lines 3205-3339.
