---
contract_id: FRS-GAIN-v004
status: active
effective_date: 2026-07-22
updated_date: 2026-07-22
supersedes: FRS-GAIN-v003
scope: Stage 3 paired local-repair Gain with expected support-mode preservation, contact-phase-conditioned ZMP, non-compensatory Physics admissibility, root-invariant Intent, and full-6D repair cost
---

# Support-Mode Physics Admissibility Gain Contract

## Design Delta

`FRS-GAIN-v003` combined Intent improvement, an available-mean Physics
improvement, and repair cost additively. That allows a sufficiently large
Intent term to compensate for a physical violation. It also allows survival to
hide a known shortcut: FEMR can create sustained lateral lean, force frozen GMT
to take unplanned compensating steps, and remain alive while visibly changing
the reference action's support semantics.

`FRS-GAIN-v004` changes the ordering rule, not the policy interface. Physics is
first an admissibility condition. Within the admissible set, the scalar Gain
then prefers better Intent realization and smaller full-6D repair. There is
still one full-6D `Delta SE(3)` actor, one scalar Gain, and one scalar Critic.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Paired Evidence And Time |
| `FRS-DP-07` | Repair Gain | `Q-01` | Non-Compensatory Core Decision |

Expected Contact and Contact phase are evidence inside `Q-PAIR/Q-01`. They are
not new top-level method modules or actor-side prediction tasks.

## Unchanged Method Boundary

The actor continues to consume only the deployable v015 interface:

```text
current Noisy root artifact + current robot/tracking state
+ deployment/Noisy q29 future Intent
-> one full-6D Delta SE(3) action at t
```

The following remain unchanged and frozen by their active contracts:

- Clean `x_t` is a repeatable dynamics reset only;
- `H` is actor-visible deployment/Noisy q29 Intent context;
- `K` is the one-action executable-evidence horizon;
- FEMR is frozen after the action at `t`;
- the sealed multi-Segment x M transaction and grouped PPO reduction;
- proposal-only HSL authority and checkpoint identity.

No Contact, support phase, Clean continuation, future root/global state, noise
label, perturbation time, or Physics admissibility bit may enter the actor
observation or supervised target.

## Paired Evidence And Time

Noisy and Repair roles must share the same sealed scenario identity, `x_t`,
current root artifact, deployment/Noisy q29 Intent, full Clean continuation,
`K`, frozen GMT, and valid-step clock. Clean is not a scored role.

The full Clean continuation is permitted only for GMT execution and the Physics
evaluator. While materializing its same ordered Clean frame identities,
`MultiMotionCommand` already reads reference body kinematics. The materializer
uses those same Clean-frame foot kinematics to deterministically derive and
seal the expected left/right support sequence over K:

```text
11 = double support
10 = left support
01 = right support
00 = flight
```

The serialized `[K,65]` GMT command is not falsely treated as if it directly
contained foot contact. The derived `[K,2]` support carrier is immutable
scenario evidence, covered by the scenario identity/hash, and reused without
resampling or mutation across all attempts.

Actual Contact comes from the existing IsaacLab `contact_forces`
`ContactSensor`. Robot foot height is not authoritative actual-contact
evidence. The existing `contact_state` schema may carry the detached immutable
sensor state, but missing production population must fail closed rather than
fall back to a height proxy.

## Expected Support-Mode Preservation

Contact comparison is phase-aware alignment, not framewise equality:

- a reference-planned support switch or step is legal;
- a small configured early/late alignment tolerance is legal;
- an extra lift-off, touchdown, or contact switch induced by Repair is a
  violation when the reference does not plan it;
- a missed planned contact, persistent dragging contact, or switch outside the
  tolerance is a violation;
- tolerance and recovery-window identities are fixed evaluator configuration,
  recorded in diagnostics, and identical for Noisy and Repair.

The comparison must preserve left/right foot identity and the ordered K-step
clock. Permuting roles, feet, phases, or scenario rows must either preserve the
same result under the corresponding identity permutation or fail closed.

## Contact-Phase-Conditioned ZMP

ZMP remains a core Physics variable. Contact phase determines its admissible
interpretation:

- during left, right, or double support, evaluate ZMP relative to the matching
  support domain;
- during a planned support transition, permit the reference action's transient
  excursion, but require return within the declared recovery window;
- during flight (`00`), ZMP is `N/A` and must be masked rather than compared to
  a static support polygon;
- if Repair changes the support phase illegally, its ZMP may not be rescued by
  evaluating it against the Repair-created support polygon.

This distinction allows planned aggressive motion while rejecting sustained
lean and unplanned compensation stepping.

## Non-Compensatory Core Decision

For each role `X` in `{Noisy, Repair}`, the Physics evaluator produces:

```text
contact_violation_X
zmp_violation_X
survival_violation_X
physics_admissible_X
physics_deficit_X
intent_quality_X
```

`physics_admissible_X` is true only when every required Contact, ZMP, and
survival condition is valid and passes. Missing or non-finite required evidence
is `UNCONFIRMED` and invalidates the row; it is never silently zero-filled.

Each required violation is normalized by a fixed, versioned evaluator scale,
never by the current batch:

```text
d_contact_X, d_zmp_X, d_survival_X in [0, 1]
physics_deficit_X = max(d_contact_X, d_zmp_X, d_survival_X)
intent_quality_X in [0, 1]
```

`N/A` ZMP steps are excluded by the phase mask; genuinely missing required
evidence invalidates the row. The `max` makes the Physics dimensions
non-compensatory before Intent is considered.

The scalar role utility uses disjoint intervals:

```text
utility_X = -1 - physics_deficit_X   if Physics is inadmissible
utility_X = intent_quality_X         if Physics is admissible
```

The tier mapping must guarantee:

1. any admissible role outranks any inadmissible role;
2. when both roles are inadmissible, smaller Physics deficit wins and Intent
   cannot compensate for a Contact, ZMP, or survival violation;
3. when both roles are admissible, Physics gives no unbounded bonus for becoming
   "more stable" and Intent quality determines the ordering;
4. Contact/support legality is resolved before ZMP is interpreted, so an
   unplanned Repair-created support polygon cannot legitimize the shortcut.

The paired scalar training signal remains:

```text
gain_total = utility_Repair - utility_Noisy - repair_penalty
```

`repair_penalty` retains full-6D magnitude and temporal-change cost, normalized
to `[0, c_max]` with fixed `c_max < 1`. It may rank two repairs inside the same
Physics tier, but the gap between the unsafe interval `[-2,-1]` and admissible
interval `[0,1]` means it cannot invert an admissible-versus-inadmissible
ordering. A zero action under identical paired evidence yields zero paired
improvement before repair cost.

This is one scalar Gain consumed by the existing scalar Critic and PPO. It does
not introduce `rho`, a second output, a second network, a second Critic, a
contact predictor, or a second optimizer.

## Intent Fidelity

Intent retains the v003 root-invariant definition. Both roles are compared to
the same deployment/Noisy q29 articulated-motion intent. Absolute Clean
root/global motion, Repair-vs-Noisy trajectory similarity, and full Clean
rollout similarity remain forbidden Intent targets.

## Single Active Owner And Consumers

`source/rsl_rl/rsl_rl/frontres/frontres_gain.py` is the unique semantic owner of
Physics admissibility, role utility, and paired `gain_total`.

The formal evidence route is:

```text
commands.py sealed scenario / expected support evidence
-> frontres_segment_live_probe.py ContactSensor + ZMP capture
-> frontres_segment_storage.py immutable one-action-K paired facts
-> frontres_gain.py FRS-GAIN-v004
-> return / replay priority evidence / grouped PPO / diagnostics / evaluation
```

All active consumers must carry `gain_contract_id=FRS-GAIN-v004` and reject
v002/v003 fallback, mixed evidence, partial phase masks, or missing provenance.
PPO reduction and optimizer ownership remain unchanged.

## Current Contract Mismatch

The current source does not implement this contract:

- `_height_contact_consistency_pair()` thresholds reference and robot foot
  height; it does not read authoritative `contact_forces` sensor state;
- `FrontRESRobotRolloutState.contact_state` exists, but production population
  from the sensor is not code-confirmed;
- `compute_paired_physics_gain()` uses an available mean of paired success,
  survival, ZMP, and contact differences;
- `compute_intent_physics_local_repair_gain()` uses the superseded additive
  `intent_weight * intent + physics_weight * physics - repair_weight * cost`;
- storage, diagnostics, return, priority, and evaluators identify v003.

These are `contract-mismatch` paths, not alternate active semantics.

## Required Diagnostics

At minimum, every valid policy row reports:

- scenario/noisy hash, transaction/motion/Segment/attempt identity, K and valid
  step mask;
- expected and actual left/right Contact sequence;
- aligned switch events, timing offsets, extra/missed/dragging violations;
- support phase, ZMP applicability mask, support-domain margin, transition
  recovery, and ZMP violations;
- survival and terminal status;
- per-role admissibility and Physics deficit;
- per-role Intent quality, paired utility difference, repair cost, and final
  scalar Gain;
- active Gain contract identity and exact formal consumers.

## Acceptance Evidence

| Gate | Required proof |
| --- | --- |
| S1 semantic owner | deterministic support modes, timing tolerance, extra/missed/dragging violations, planned step, flight ZMP mask, transition recovery, lexicographic dominance, no-op, missing/non-finite fail-closed, and permutation tests |
| S2 formal connectivity | one sealed scenario supplies the same expected support evidence to every attempt; actual ContactSensor/ZMP evidence enters immutable storage; return, priority, grouped PPO, diagnostics, local and held-out evaluation consume only v004 |
| S4 bounded live | one 8-env complete transaction records real sensor Contact, phase-conditioned ZMP, survival, admissibility, utility, Gain, advantage, gradient, exact-one update, and committed checkpoint with no v003 fallback |

## Stop Conditions

Stop implementation and return to design review if any of these occurs:

- expected support cannot be deterministically derived from the sealed Clean
  continuation without adding actor-visible Clean information;
- actual foot Contact cannot be obtained from the configured ContactSensor;
- phase/timing identity cannot remain aligned across roles and M attempts;
- aggressive planned motion cannot be distinguished from extra Repair-induced
  stepping without a new semantic label or predictor;
- repair cost or Intent can numerically override an inadmissible Physics tier;
- any active consumer mixes v003 and v004, silently fills missing evidence, or
  changes H, K, one-action-K, transaction, grouped PPO, HSL, actor, or Critic
  semantics.

Until S1/S2/S4 pass, `FRS-GAIN-v004` is accepted semantics and the current code
is an explicit implementation mismatch. Long training, X1, deployment
composition, and paper experiments remain blocked.
