# FRS-METHOD-v018 Proposal: Future-Conditioned State-Value Critic

Status: confirmed and activated as design rationale
Date: 2026-08-08
Workflow stage: Stage 2 complete; implementation governed by the current engineering plan
Affected candidate contracts: `FRS-METHOD-v018`, `FRS-TRAIN-v016`, `FRS-PPO-v006`

## Problem

The active Critic is a state-value baseline, but its formal input contains only
the current 289D privileged observation. The Actor already receives two sealed
future q29 frames because the same current physical state can have different
repair difficulty under different upcoming motion Intent. Omitting that Intent
from the Critic aliases states whose expected achievable `G_total` differs.

The completed K8/M2 TRAIN-v015 run also exposes an independent optimization
boundary. Critic calibration remains weak while the formal optimizer clips the
combined Actor and Critic gradient vector once. A large Critic error can
therefore reduce the Actor gradient through a shared clip factor even though
the optimizer already has disjoint Actor and Critic parameter groups.

These failures do not show that the Critic must distinguish actions. Each
Repair attempt already receives its own realized `G_total`; subtracting one
shared state baseline preserves the within-Segment action ordering.

## Scientific Classification

```text
Critic failure     = state representation aliasing
Actor suppression = optimization-boundary coupling
Not the problem    = missing action identity in a Q function
```

The indispensable variable is the same deployable future Intent already
sealed for the Actor. The smallest optimization repair is independent clipping
of the two already-disjoint gradient families before the existing one Adam
step.

## Candidate Design

### FRS-DP-09 / M-05: Actor & Critic Warmup

The Critic remains one scalar state-value function:

```text
V_RA(s_t) = expected G_total of a Repair sampled from frozen pi_old
```

It does not receive the sampled 6D Repair action and does not become
`Q(s_t, a_t)`. For one sealed Segment, all exact-M attempts therefore share one
old value baseline. Each attempt keeps its own realized return and advantage:

```text
return_m    = G_total_m
advantage_m = G_total_m - V_old(s_t)
```

The Critic target for one Segment is the mean of the exact-M returns sampled
under the same frozen policy and scenario. This makes the supervised object
explicitly match a state value rather than presenting M action-conditioned
outcomes as M different desired values for an identical state.

Actor and Critic gradients remain disjoint. Each family is clipped against the
existing `max_grad_norm=0.5` independently, followed by exactly one step of the
same two-group Adam. Actor LR remains `3e-6`; Critic LR remains `1e-5`; the
fixed phase schedule remains unchanged.

### FRS-DP-10 / M-11: Future Motion Context

The Critic input becomes:

```text
current privileged state                              289D
q29[t+1] and q29[t+2] from the sealed Noisy stream     58D
                                                       ----
state-value Critic input                               347D
```

The 58D tail is the same ordered, detached,
`deployment_noisy_q29`-provenance object already visible to the Actor. It is
not copied from Clean continuation and is not recomputed per attempt. The
Actor remains 158D, frozen GMT remains 770D, and the environment's current
privileged observation remains 289D before the FrontRES-specific concatenation.

The 6D Repair action, Clean continuation, expected Contact, phase-ZMP,
survival, perturbation label, perturbation timing and K do not enter the
Critic. K remains stage identity and triggers recalibration of the same Critic.

### FRS-DP-07 / Q-01: Repair Gain And Advantage

FRS-GAIN-v007 remains the unique owner of each attempt's `G_total`. The Critic
does not score which action is better. It supplies the expected score for the
sealed state; action-specific evidence remains the realized Repair rollout.

Subtracting one shared value from every attempt preserves all strict
within-Segment `G_total` orderings. The Actor therefore continues to learn
which Repair was better without an action-conditioned Critic.

## Persistence And Evaluation Identity

The changed Critic input width and optimization boundary require a fresh
Stage-3 identity:

```text
checkpoint_schema = frontres-v017-checkpoint-v11
method_contract_id = FRS-METHOD-v018
training_contract_id = FRS-TRAIN-v016
optimization_contract_id = FRS-PPO-v006
critic_input_dim = 347
critic_value_kind = state_value
critic_action_conditioned = false
gradient_clip_identity = separate-actor-critic-v1
```

Checkpoint-v10 remains historical evidence and must reject as a v11 resume
source before mutable restoration. A fresh campaign may still initialize the
158D proposal Actor from the accepted HSL-v2 artifact because HSL supplies no
Stage-3 Critic, optimizer or transaction state.

Critic calibration is evaluated against the Segment-level exact-M mean return,
not against its ability to reproduce each action-specific outcome. Telemetry
must expose finite Actor and Critic pre/post clip norms separately, the shared
within-Segment value, the Segment mean target, value error, Actor/Critic
parameter deltas and exact-one optimizer receipt.

## Preserved Behavior

- FRS-GAIN-v007 and every per-attempt `G_total` remain unchanged.
- Actor input remains the deployable 158D prefix.
- Actor output remains one direct full-6D world-frame `Delta SE(3)` at `t`.
- Frozen GMT remains restricted to its 770D suffix.
- K8/M2 -> K16/M3 -> K32/M4, per-K DR and warmup iteration counts remain
  unchanged.
- One transaction still seals two Segments and exact M attempts per Segment.
- One transaction still performs exactly one Adam optimizer step.
- No MOSAIC host, simulator, Gain, GMT, HSL target or deployment authority is
  changed.

## Rejected Alternatives

### Add the 6D Repair action directly to the current Critic

Rejected. This would create `Q(s,a)` but retain a PPO update derived for a
state-value baseline. Correct Q-based credit would require an additional
state-value or expected-Q baseline and would be a different Actor-Critic
algorithm.

### Give the Critic Clean future or evaluator evidence

Rejected. It would let the training baseline depend on information unavailable
to the deployed repair decision and would violate the active information
boundary.

### Change only the learning rates

Rejected. A larger or smaller LR does not restore missing future Intent and
does not prevent one gradient family from setting the other's global clip
factor.

## Falsifiable Predictions

1. For one Segment, permuting exact-M Repair actions changes neither the 347D
   Critic input nor the shared old value; it only permutes returns and
   advantages.
2. Subtracting the shared value preserves every strict within-Segment return
   ordering.
3. Scaling only Critic loss cannot change the clipped Actor gradient; scaling
   only Actor loss cannot change the clipped Critic gradient.
4. During critic-only, Actor parameters and Actor Adam state remain unchanged
   while the Critic updates exactly once.
5. A bounded official transaction reports 158D Actor, 347D Critic and 770D GMT
   identities, finite separate gradient norms and checkpoint-v11.

These predictions establish method and engineering closure. They do not claim
that a bounded transaction proves policy quality. Long training must still be
judged from the resulting Critic calibration and Actor learning trajectory.

## Activation Record

The user confirmed the updated Design Inspector cards for `Repair Gain`,
`Actor & Critic Warmup` and `Future Motion Context` on 2026-08-08. The accepted
semantics are active in `FRS-METHOD-v018`, `FRS-TRAIN-v016` and `FRS-PPO-v006`.
This file remains rationale only and cannot override those contracts.
