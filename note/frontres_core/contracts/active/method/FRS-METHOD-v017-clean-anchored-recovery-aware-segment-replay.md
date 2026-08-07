---
contract_id: FRS-METHOD-v017
status: active
effective_date: 2026-08-01
updated_date: 2026-08-03
supersedes: FRS-METHOD-v016
scope: Clean-anchored Recovery-Aware local root-artifact repair with deployable future Intent, one full-6D action, one-action-K frozen-GMT evidence, one Clean plus one fixed Noisy baseline, exact-M Segment Replay, and grouped scalar PPO
---

# Clean-Anchored Recovery-Aware Segment Replay

## Design Delta

FRS-METHOD-v016 used a scalar paired Intent objective while Contact,
phase-ZMP, and survival separately constrained the actor through first-order
projection. Long-run evidence established that the evidence route was
executable, but the split authority did not express the accepted
Recovery-Aware ordering: Physics recovery must matter most under high remaining
physical pressure, while Intent must regain ranking authority near the Clean
physical regime.

FRS-METHOD-v017 replaces that split with one Clean-anchored scalar ordering:

```text
Clean Rollout            -> desired motion and support semantics
fixed Noisy Rollout      -> zero-action baseline
M Repair Rollouts        -> one-action reachable candidates
Intent + Physics + cost  -> one Recovery-Aware G_total per attempt
all valid attempts       -> grouped equal-mass PPO and one update
```

The old independent Physics projection, constraint Critic interpretation, KKT
actor gate, and fallback recovery update are retired from the active method.
Contact, support-foot drift, phase-ZMP, and survival remain indispensable raw
Physics evidence inside FRS-GAIN-v007.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-01` | Perturbation Data | `M-02` | Sealed Local Scenario |
| `FRS-DP-02` | Segment Replay | `SR-01` | Frozen-Policy Transaction |
| `FRS-DP-03` | K-step Curriculum | `M-06` | One-Action K Evidence |
| `FRS-DP-04` | FrontRES 6D Repair | `M-04` | Actor And Information Boundary |
| `FRS-DP-05` | Frozen GMT | `M-10` | Frozen Execution Boundary |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Clean/Noisy/Repair Evidence |
| `FRS-DP-07` | Repair Gain | `Q-01` | Recovery-Aware Ordering |
| `FRS-DP-08` | HSL Warmup | `M-03` | Training Authority |
| `FRS-DP-09` | Actor & Critic Warmup | `M-05` | Training Authority |
| `FRS-DP-10` | Future Motion Context | `M-11` | Actor And Information Boundary |

The ten parent design points and their interactions are frozen by the confirmed
Design Inspector. No new Contact, constraint, calibration, or
checkpoint parent block is introduced.

## Method Closure

The learned object is one full-6D current-frame repair under the robot's current
dynamics. Clean indicates the desired motion semantics, Noisy defines doing
nothing, and same-scenario Repair outcomes show what one action can currently
reach. Recovery-Aware is the continuous interaction between remaining Physics
pressure and Intent improvement inside each attempt's `G_total`. Segment Replay
does not invent another Physics/Intent rule; it exposes the candidate ordering
through all valid attempts.

The method does not require one Repair to equal Clean, generate a new recovery
trajectory, or solve arbitrary historical imbalance. It learns the best local
repair direction that is reachable from the replayed `x_t` under the frozen GMT.

## Actor And Information Boundary

FrontRES reads the deployable 158D prefix:

```text
current FrontRES state/artifact features                 100D
q29[t+1] and q29[t+2] from one sealed Noisy stream      58D
                                                         ----
                                                         158D
```

The q29 tail remains deployment/Noisy provenance even when a root-only artifact
leaves its numbers equal to Clean calibration. Clean continuation, expected
Contact, phase-ZMP, survival, noise labels, perturbation timing, and future
root/global quantities never enter the actor input.

The actor emits exactly one world-frame full-6D residual at `t`:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Perturbation family never narrows this output. There is no rho, confidence,
acceptance head, action mask, second actor, second Critic, `tanh`, `clip`, or
`clamp`. Upward `dz` is discouraged through the initialized direction,
Clean-anchored rollout consequence, and full-6D repair cost rather than a hard
output truncation.

## Sealed Local Scenario

One scenario seals before any attempt:

```text
x_t dynamics identity
current root artifact_t and single-family corruption identity
deployment/Noisy q29 intent[t:t+H]
Clean GMT-only continuation[t+1:t+K]
expected support evidence derived from that continuation
K, valid-step clock, scenario_id, noisy_segment_hash
frozen old-policy identity
```

`x_t` restores dynamics only and does not expose a Clean actor reference. The
artifact, strength, application point, future Intent, continuation, and hash
remain immutable across the zero-action Noisy rollout and all M Repair attempts.
Reset may not resample, mutate, or mix them. There is no Noisy physical prefix.

The active corruption family is single `local_rp`. Composite corruption is not
part of this contract. Perturbation strength follows the K-conditioned
four-class inner DR curriculum owned by FRS-TRAIN-v015; it is not monotonically
ramped and is not controlled by Gain or PPO.

## Clean/Noisy/Repair Evidence

Each sealed Segment executes exactly:

```text
one Clean Rollout
one fixed zero-action Noisy Rollout
M Repair Rollouts
```

Clean and Noisy execute once and their observed K-step outcomes are sealed and
read-only reused across all M comparisons. This removes avoidable baseline
resampling noise without claiming complete cancellation of simulator dynamics.
Only Repair is sampled from `pi_old` and only Repair contributes PPO rows.

Clean defines planned support changes, legitimate dynamic lean, phase-ZMP
behavior, and intended pose. Noisy is the causal no-repair zero point. Each
Repair records the consequence of its one sampled action. Missing or malformed
required evidence fails the complete transaction closed. Valid physical no-load
is a Contact violation and role-specific phase-ZMP `N/A`, not corrupt evidence.

## One-Action K Evidence And Frozen GMT

One attempt contains one policy/action row regardless of K. FrontRES acts only
at `t` and remains frozen for the complete K-step evidence horizon. The frozen
GMT and its original 770D observation suffix execute the same GMT-only Clean
continuation. K never becomes an actor input and never creates additional PPO
rows.

K evaluates whether one local repair remains useful over a longer consequence
horizon. H=2 remains only future Intent context for choosing the current action.
The two horizons must not be merged.

## Recovery-Aware Ordering

FRS-GAIN-v007 is the unique owner of:

```text
per-channel Clean-conditioned remaining problems
fixed semantic scales and K-step aggregation
Intent and Physics family aggregation
G_I, G_P, remaining Physics pressure, repair cost, and G_total
beta initialization and human-reviewed live calibration
```

The accepted ordering has no hard Physics/Intent stage switch. High remaining
Physics pressure amplifies Physics improvement or deterioration. Near the Clean
physical regime, Intent may accept a mild Physics trade-off when the total
recovery ordering improves. A Repair that creates severe imbalance raises its
own remaining pressure and is strongly penalized. Sustained lean without extra
support compensation remains an Intent/demo-quality error; unplanned stepping,
dragging, missed support, or changed support remains Physics error.

## Frozen-Policy Transaction

One transaction contains at least two selected Segments; the active campaign
uses exactly two. Each Segment contributes exact M Repair attempts from one
frozen `pi_old`. Collection performs zero optimizer steps. Every valid attempt
keeps equal structural mass and contributes its own `G_total`, value, and
advantage. Winner-only, argmax, best-of-M loss weights, priority weights, and
score-proportional row mass are forbidden.

Only after all Segment x M attempts and shared baselines are sealed may
FRS-PPO-v005 perform exactly one grouped optimizer update. A partial or mixed
transaction cannot update policy, Critic, optimizer, sampler, curriculum, or
checkpoint receipt.

## Training And Persistence Authority

FRS-TRAIN-v015 owns HSL-to-HRL initialization, the coordinated K x M schedule,
per-K inner DR progression, Critic-only recalibration, actor ramp, joint optimization,
calibration, and strict checkpoint identity. HSL initializes only the proposal
actor/std and 158D actor-prefix normalizer. The scalar Critic predicts the
complete Recovery-Aware `G_total` and is recalibrated whenever K increases.

Old checkpoint-v6 and earlier identities cannot resume the changed scalar
target. Strict resume requires the new versioned contract and committed
transaction identity before any mutable state is restored.

## Deployment Boundary

Deployment processes one pre-materialized Noisy/deployment reference sequence
frame by frame. At frame `t`, FrontRES reads the current 100D prefix and the two
Noisy q29 future offsets, repairs only the current reference, and hands it to
frozen GMT. Robot dynamics continue naturally, but the repaired reference is
not written back into the next actor reference. Clean Rollout and evaluator
evidence never become deployment inputs.

## Forbidden Old Assumptions

- scalar Intent-minus-cost with independent Physics projection;
- Physics constraint gradients, projection/KKT gate, or recovery fallback;
- Clean future, expected support, or evaluator labels in actor observations;
- Clean or Noisy as PPO rows;
- repeated Clean/Noisy rollout per attempt;
- one Repair action per K step;
- winner-only or score-weighted Segment Replay;
- per-Segment/per-K dynamic beta or Gain-driven corruption curriculum;
- rho, confidence, authority head, second network, or second optimizer.

## Required Evidence And Stop Conditions

Implementation must prove the complete route:

```text
config and checkpoint identity
-> sealed scenario and fixed baselines
-> deployable 158D actor input and one full-6D action
-> frozen GMT K-step Clean-anchored evidence
-> FRS-GAIN-v007 G_total for every valid Repair
-> FRS-PPO-v005 grouped exact-one update
-> FRS-TRAIN-v015 committed checkpoint and diagnostics
```

Stop if Clean reaches actor input, a baseline is resampled, a transaction mixes
scenario/K/M/policy identity, evidence is silently zero-filled, an old
projection path remains active, valid attempts are winner-filtered, more than
one optimizer step occurs, or runtime ordering contradicts the raw paired
evidence.
