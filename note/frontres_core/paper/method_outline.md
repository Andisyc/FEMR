# FEMR / FrontRES Current Method Outline

Updated: 2026-07-19
Status: current paper-facing method view

Implementation truth remains in the active contract registry. This document
contains no historical alpha/rho/acceptance/authority method.

The current human-facing Concept Figure summarizes ten design points:
Perturbation Data, Segment Replay, K-step Curriculum, FrontRES 6D Repair,
Frozen GMT, Paired Rollouts, Repair Gain, HSL Warmup, and Actor & Critic
Warmup, plus Future Motion Context. Detailed semantics and stable figure-block
mappings are recorded in the active contract registry.

## 1. Problem

Motion references reconstructed from video or other imperfect sources may be
kinematically plausible but dynamically difficult for a frozen humanoid motion
tracker to execute. The tracker is not retrained. FEMR repairs the reference
before it reaches frozen GMT.

```text
corrupted reference + current tracking state
-> FrontRES full-6D task-space repair
-> repaired reference
-> frozen GMT
-> robot execution
```

The method is reference repair, not tracker replacement and not generation of a
new recovery motion.

## 2. Repair Policy

The actor consumes current robot/balance/tracking state, current Noisy
root/anchor artifact, and a short future 29DoF internal-motion intent window.
It does not read future raw root/global reference or Clean provenance. It
outputs:

```text
Delta g = [dx, dy, dz, droll, dpitch, dyaw]
```

All perturbation families retain all six repair dimensions because coupled
whole-body dynamics can require translation and rotation even when the injected
artifact is local roll/pitch. Named physical execution bounds may restrict
unsafe writes, but perturbation family never acts as an action mask.

## 3. Three-Stage Training

### Stage 1: Replayable Segment Cache

Long motions are divided into local dynamic segments. Each selected scenario
records a replayable Clean dynamic start, one current root-level artifact, a
future q29 intent window, and a common full Clean continuation for GMT.

### Stage 2: Full-6D HSL Initialization

HSL is intended to initialize the same six-dimensional actor later optimized by
PPO. Under v015, its existing Clean-oriented target route must be audited before
it is claimed as an active initializer: it may not leak Clean future information
or redefine the Noisy-to-Executable objective.

### Stage 3: Segment Replay PPO

Stage 3 samples segments from global, replay, and optional review sources. It
compares Noisy and Repair from the same dynamic reset: FEMR acts once at the
first frame, is frozen afterward, and frozen GMT executes the same Clean
continuation for K steps. K is evidence for that one action, not a sequence of
K policy actions.

Before joint Actor/Critic PPO training, Stage 3 preserves the HSL actor through
two optimization phases. Critic warmup learns the Segment return distribution
while holding the actor fixed against PPO updates. Actor warmup then introduces
the direct full-6D actor objective progressively before ordinary joint updates.
This is optimization protection for the same repair actor, not an acceptance,
rho, authority, or active-dimension mechanism.

Policy rows receive PPO credit. Search or counterfactual trials may update
segment evidence but cannot be relabeled as actions sampled by the policy.

## 4. Paired Repair Objective

One scalar environment reward is not used. The objective has two paired
improvements and one repair regularizer.

### Internal-Motion Intent

The trusted motion intention is the root-invariant 29DoF articulated motion
carried by the Noisy/deployment reference. The root/ground artifact is repaired;
the internal joint motion is preserved. Both rollouts are compared to this same
intent, excluding absolute root position and root orientation:

```text
G_intent = Q_internal(Repaired | I_noisy)
         - Q_internal(Noisy | I_noisy)
```

This is not direct similarity between Repaired and Noisy rollouts: that would
reward a no-op. Full Clean global rollout motion is not an actor target. Clean
only calibrates the q29-intent assumption and provides the shared K
continuation.

### Physical Executability

Physics quality measures whether frozen GMT executes the reference reliably:

- success/fall;
- survival;
- ZMP/support margin;
- contact consistency.

```text
G_physics = Q_physics(Repaired) - Q_physics(Noisy)
```

This separation asks whether the repair preserves the intended articulated
motion while making frozen-GMT execution physically more reliable.

### Repair Regularization

The residual pays an ordinary cost for:

- full-6D correction magnitude;
- temporal correction change;
- nonzero intervention on valid Clean/near-Clean rows when available.

This term prevents unnecessary or oscillatory correction but is not a third
task-success objective.

### Total Gain

```text
G_total = w_intent * G_intent
        + w_physics * G_physics
        - w_repair * C_repair
```

There is no epsilon-style mechanism, additional acceptance gate, confidence
head, rho, or learned authority variable. Every term is computed by paired
Noisy/Repaired evidence under the same segment and horizon.

## 5. Segment Replay

Uniform global sampling provides coverage but rarely revisits difficult local
states. Pure local repetition learns one state but loses motion diversity.
Segment Replay combines:

```text
global source -> discover segments
replay source -> revisit high-learning-value segments
review source -> recheck solved or stale evidence when configured
```

Priority is computed from rollout-time repair evidence and is independent of
post-update PPO KL, parameter deltas, or logger state.

## 6. Future Motion Context

The same current root artifact can require different repair directions when the
upcoming support phase or joint motion differs. FEMR therefore observes a short
future 29DoF internal-motion window from the deployment Noisy reference. This
context resolves repair-direction ambiguity; it does not reveal future raw
root/global artifact or Clean reference.

## 7. K-Step Curriculum

Short K exposes immediate repair effects cheaply. Longer K reveals accumulated
instability and delayed regret. The sampler may assign `8/16/32/64` according to
segment evidence.

Each K experiment authorizes one FEMR action at its first frame. FEMR is frozen
afterward, and GMT follows the shared full Clean continuation. K is part of the
method only when this lifecycle reaches reset, rollout accumulation, done masks,
PPO return, sampler evidence, diagnostics, and evaluation.

## 8. PPO Semantics

PPO directly optimizes the distribution over full-6D Delta SE(3). Sampled
action, old log-probability, old mean/sigma, stored action, and executed repair
share one representation. Advantages use sign-preserving scale-only scaling so
positive no-regret evidence remains positive.

Search rows are invalid for PPO before the batch boundary. Post-update trust
region diagnostics do not influence Segment Replay priority.

## 9. Evaluation

Periodic and sequence evaluation independently sample/reset paired segments and
report:

```text
Intent: root-invariant q29/relative-articulation fidelity, G_intent
Physics: success, fall, survival, ZMP/support, contact, G_physics
Repair: Delta SE norm, temporal change, C_repair
Summary: G_total, local artifact, intent provenance, continuation, start frame, effective K
```

Local K evaluation and full-sequence deployment composition evaluation are
separate: the latter tests repeated repairs under persistent artifacts and
never substitutes for the first-action K return. Missing evidence is
`UNCONFIRMED`, never zero.

## 10. Method Boundary

FEMR does not:

- replace or update GMT;
- generate a new recovery-reference manifold;
- optimize teleoperation, velocity-command, or generic environment reward;
- restrict repair dimensions by perturbation family;
- use a confidence, acceptance, alpha/rho, or authority actor;
- use future raw root/global reference or Clean future as actor context;
- claim implementation from a helper that formal training does not call.

## 11. Main Contributions

1. A lightweight full-6D task-space reference repair policy before a frozen
   humanoid tracker.
2. Dynamic Segment Replay that combines motion coverage, targeted revisitation,
   and variable-horizon evidence.
3. A future-intent observation that resolves repair-direction ambiguity without
   exposing Clean or future root artifact to the actor.
4. An auditable training/evaluation contract that uses one paired
   Intent/Physics/Repair Gain decomposition for PPO credit, sampler evidence,
   and runtime diagnostics, while keeping local K and full-sequence composition
   evidence distinct.
