---
contract_id: FRS-TRAIN-v006
status: superseded
effective_date: 2026-07-19
updated_date: 2026-07-20
supersedes: FRS-TRAIN-v005
superseded_by: FRS-TRAIN-v007
scope: HSL interface continuity and formal Stage-3 routing for future-intent local repair, one policy action per attempt, and frozen-FEMR Clean-continuation K evidence
---

# Future-Intent Single-Action K Training Contract

## Historical Status

This archived v006 contract left HSL target authority unresolved pending Gate H0.
H0-A resolved that decision in `FRS-TRAIN-v007`. This file is historical and
must not authorize HSL, rollout labels, checkpoint migration, or training.

## Recorded v006 Semantics

The active actor/evaluator split was:

```text
actor at t:
  current Noisy root/anchor artifact + future 29DoF intent

GMT at t:
  Noisy root/anchor artifact, optionally written by Delta SE(3)_t

GMT at t+1 ... t+K:
  common full Clean continuation, with FEMR frozen
```

HSL and Stage 3 were required to ultimately consume the same deployable actor
interface:

```text
existing robot, balance, and tracking observation
+ current Noisy root/anchor reference error
+ ordered future internal-intent window I[t:t+H]
-> one full-6D actor distribution
```

The future carrier was root-invariant articulated motion, not a legacy
`[H,65]` raw reference prefix. v006 did not redefine the existing HSL target;
it required an audit before HSL reactivation. That unresolved target boundary is
the sole semantic reason v007 supersedes this contract.

For each selected scenario, v006 required a sealed `x_t`, root artifact,
intent window, Clean continuation, and K horizon. Reset restored dynamics only.
Each attempt had one policy tuple and one action at t; FEMR remained frozen
during K-step GMT execution.

The formal route was required to verify layout/provenance, freeze `pi_old`,
collect M attempts from common `x_t`/artifact/I/C, aggregate one-action K
evidence, run grouped v003 PPO, take exactly one optimizer step, and update
priority from evidence only. Noisy and Repair were the only scored roles; Clean
continuation was evaluator-only.

## Historical Acceptance State

| Gate | Required proof | State at supersession |
| --- | --- | --- |
| S0 | registry, Concept Figure, plan, and legacy isolation | documentation-complete |
| S1 | root-only perturbation / q29 invariant and actor H provenance | partially implemented outside HSL |
| S2 | two-role reset -> one action -> frozen FEMR -> Clean GMT continuation | not started |
| S3 | intent/physics Gain storage and consumer connectivity | not started |
| S4 | transaction/grouped-PPO formal route and persistence | blocked |
| S5 | bounded live identity sentinel | user-gated |

