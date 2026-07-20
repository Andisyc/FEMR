---
contract_id: FRS-GAIN-v002
status: superseded
effective_date: 2026-07-16
updated_date: 2026-07-16
supersedes: FRS-GAIN-v001
superseded_by: FRS-GAIN-v003
scope: Stage 3 Segment Replay paired gain and evaluation decomposition
---

# Style-Physics Repair Gain Contract

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | `Pairing And Time` |
| `FRS-DP-07` | Repair Gain | `Q-01` | `Core Decision` |

## Core Decision

FrontRES gain has two primary paired improvements and one ordinary repair
regularizer:

```text
style_gain   = style_quality(Repaired | Clean) - style_quality(Noisy | Clean)
physics_gain = physics_quality(Repaired)       - physics_quality(Noisy)
repair_cost  = magnitude_and_temporal_cost(Delta SE(3))

gain_total = w_style * style_gain
           + w_physics * physics_gain
           - w_repair * repair_cost
```

No epsilon-style budget, calibration artifact, learned gate, or extra authority
variable belongs to this design. Small style degradation may be offset by a
larger physical-executability improvement through the explicit scalar weights.

## Style Gain

Style asks whether the robot execution still follows the original Clean motion.
The comparison target is immutable Clean, never the FrontRES-written reference.

Required raw components:

- body MPJPE;
- root-orientation geodesic error;
- body/root velocity error;
- acceleration error.

Each component is converted to a bounded or normalized quality with one named
scale. The Repaired and Noisy branches must use identical definitions and
scales. The initial implementation may stage components, but every omitted
component must remain visible as `UNCONFIRMED`, not silently treated as zero.

## Physics Gain

Physics asks whether frozen GMT can execute the reference more reliably.

Required raw components:

- success/fall;
- raw survival steps, reported only as episode/survival length;
- survival quality = `survival_steps / effective_horizon_K`;
- physics survival Gain = `repaired_survival_quality - noisy_survival_quality`;
- ZMP/support margin;
- contact consistency.

Penetration and floating may be added only when their runtime measurements are
reliable and tested. They are not implicit requirements of v001.

## Repair Regularizer

Repair cost prevents large or oscillatory residuals from becoming a shortcut:

- full-6D Delta SE(3) magnitude;
- temporal change of Delta SE(3);
- correction on Clean or near-Clean references when such rows are available.

This is a regularizer, not a third notion of task success.

## Pairing And Time

Noisy and Repaired must share motion identity, start state, perturbation, and
effective horizon K. Raw survival steps remain a report-only quantity. Final
paired Gain uses `survival_steps / effective_horizon_K`; per-step training uses
the current alive increment divided by that same K, so summing the per-step
survival component reproduces the final K-normalized survival Gain. Done and
survival semantics must be identical between training and evaluation.

## Training And Evaluation Alignment

Training and evaluation must share the same component functions, units,
normalization scales, signs, and K-step aggregation. Evaluation additionally
prints raw survival steps separately from repaired/noisy survival quality and
their paired Gain; `gain_total` alone is never sufficient evidence of method
quality. A 120-step sequence survival report remains a long-sequence metric,
not the denominator of a short Segment Gain.

## Single Active Gain Owner

`frontres_gain.py` is the only active owner of paired Style/Physics/Repair
Gain calculation. PPO reward, Segment Replay evidence and priority input,
training diagnostics, periodic evaluation, and sequence evaluation must all
consume the same `gain_total` and component decomposition.

The former family-specific executability score is a legacy compatibility path.
It must not remain active as a PPO reward, sampler score, sampler difficulty
heuristic, diagnostic replacement, or evaluation metric. It may remain in the
repository only as explicitly marked legacy code with an isolation or
retirement test.

The headline interpretation is:

```text
physical executability improvement
while preserving original-motion style as measured explicitly
and avoiding unnecessary repair magnitude or oscillation
```

## Forbidden Inputs

- full environment reward;
- teleoperation or command-following reward;
- unrelated tracking/task reward;
- reward terms whose raw metric is not printed in evaluation;
- a score computed against the modified Repaired reference instead of Clean;
- perturbation-family action masks;
- online or batch-dependent epsilon/style tolerance.

## Required Diagnostics

```text
style.mpjpe_noisy / repaired / gain
style.root_ori_noisy / repaired / gain
style.vel_noisy / repaired / gain
style.acc_noisy / repaired / gain
physics.success/fall/survival noisy / repaired
physics.survival_steps_raw noisy / repaired (report-only)
physics.survival_quality noisy / repaired
physics.survival_gain normalized paired difference
physics.zmp_margin_noisy / repaired / gain
physics.contact_noisy / repaired / gain
repair.delta_se_norm
repair.delta_se_temporal_change
gain.style / gain.physics / cost.repair / gain.total
```

## Acceptance

1. S1 component tests prove sign, units, normalization, pairing, and K masks.
2. S2 connectivity proves the formal Segment live route, PPO return, sampler
   evidence, periodic eval, and sequence eval consume the same decomposition.
3. S4 live evidence proves all components are populated and non-stale.

Until the code passes these gates, any active consumer of the former RP-only
Segment score remains a code-confirmed implementation mismatch, not an
alternative active design.
