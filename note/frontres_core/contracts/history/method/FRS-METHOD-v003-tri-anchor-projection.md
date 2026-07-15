---
contract_id: FRS-METHOD-v003
status: superseded
effective_date: 2026-06-10
updated_date: 2026-07-13
supersedes: FRS-METHOD-v002
superseded_by: FRS-METHOD-v004
scope: Noisy-Stable-Repair tri-anchor projection
---

# Tri-Anchor Projection

This version introduced three semantic anchors:

```text
N = Noisy/no-op
S = Stable Frame
R = HSL Repair
P = rho * R + (1-rho) * ((1-alpha) * N + alpha * S)
```

Rho represented repair retention. Alpha selected whether rejected repair mass
retreated toward Noisy or Stable. Grouped executable evidence later trained the
six rho dimensions through planar, attitude, and vertical groups without
changing the 6D output surface.

The design was superseded when rollout credit for the jointly executed alpha/rho
projection was moved from detached pseudo-targets to a joint policy-gradient
formulation.

Source: restricted compendium lines 3340-3590, with grouped/current-action
refinements through line 3779.
