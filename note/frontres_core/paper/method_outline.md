# FEMR / FrontRES Current Method Outline

Updated: 2026-07-14
Status: current paper-facing method view

Implementation truth remains in the active contract registry. This document
contains no historical alpha/rho/acceptance/authority method.

The current human-facing Concept Figure summarizes nine design points:
Perturbation Data, Segment Replay, K-step Curriculum, FrontRES 6D Repair,
Frozen GMT, Paired Rollouts, Repair Gain, HSL Warmup, and Actor & Critic
Warmup. Detailed semantics and stable figure-block mappings are recorded in the
active contract registry.

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

The actor consumes the current 870D FrontRES/GMT observation and outputs:

```text
Delta g = [dx, dy, dz, droll, dpitch, dyaw]
```

All perturbation families retain all six repair dimensions because coupled
whole-body dynamics can require translation and rotation even when the injected
artifact is local roll/pitch. Named physical execution bounds may restrict
unsafe writes, but perturbation family never acts as an action mask.

## 3. Three-Stage Training

### Stage 1: Replayable Segment Cache

Long motions are divided into local dynamic segments. Each segment records
motion identity, start state, Clean/Noisy dynamic state, perturbation metadata,
and enough state to reset or faithfully preroll frozen GMT.

### Stage 2: Full-6D HSL Initialization

HSL learns an anti-perturbation initialization for the same six-dimensional
actor later optimized by PPO. It is an initialization, not a separate proposal
whose authority is selected by another network.

### Stage 3: Segment Replay PPO

Stage 3 samples segments from global, replay, and optional review sources,
constructs paired Noisy/Repaired execution, rolls each row through its effective
horizon K, and directly updates the full-6D repair distribution.

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

### Original-Motion Style

Style quality measures the robot execution against immutable Clean motion:

- body MPJPE;
- root-orientation error;
- velocity error;
- acceleration error.

```text
G_style = Q_style(Repaired | Clean) - Q_style(Noisy | Clean)
```

The modified Repaired reference is never used as its own style target.

### Physical Executability

Physics quality measures whether frozen GMT executes the reference reliably:

- success/fall;
- survival;
- ZMP/support margin;
- contact consistency.

```text
G_physics = Q_physics(Repaired) - Q_physics(Noisy)
```

This separation follows the useful distinction between source-motion fidelity
and physical feasibility: a small loss in MPJPE may be accepted when it produces
a larger, explicit physical-executability improvement.

### Repair Regularization

The residual pays an ordinary cost for:

- full-6D correction magnitude;
- temporal correction change;
- nonzero intervention on valid Clean/near-Clean rows when available.

This term prevents unnecessary or oscillatory correction but is not a third
task-success objective.

### Total Gain

```text
G_total = w_style * G_style
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

## 6. K-Step Curriculum

Short K exposes immediate repair effects cheaply. Longer K reveals accumulated
instability and delayed regret. The sampler may assign `8/16/32/64` according to
segment evidence.

K is part of the method only when it reaches reset, rollout accumulation, done
masks, PPO return, sampler evidence, diagnostics, and evaluation. A planner that
can output several K values while formal training executes one fixed horizon is
not integrated.

## 7. PPO Semantics

PPO directly optimizes the distribution over full-6D Delta SE(3). Sampled
action, old log-probability, old mean/sigma, stored action, and executed repair
share one representation. Advantages use sign-preserving scale-only scaling so
positive no-regret evidence remains positive.

Search rows are invalid for PPO before the batch boundary. Post-update trust
region diagnostics do not influence Segment Replay priority.

## 8. Evaluation

Periodic and sequence evaluation independently sample/reset paired segments and
report:

```text
Style:  MPJPE, root orientation, velocity, acceleration, G_style
Physics: success, fall, survival, ZMP/support, contact, G_physics
Repair: Delta SE norm, temporal change, C_repair
Summary: G_total, perturbation, motion identity, start frame, effective K
```

Training and evaluation share component functions, units, signs, scales, and
K-step aggregation. Missing evidence is `UNCONFIRMED`, never zero.

## 9. Method Boundary

FEMR does not:

- replace or update GMT;
- generate a new recovery-reference manifold;
- optimize teleoperation, velocity-command, or generic environment reward;
- restrict repair dimensions by perturbation family;
- use a confidence, acceptance, alpha/rho, or authority actor;
- claim implementation from a helper that formal training does not call.

## 10. Main Contributions

1. A lightweight full-6D task-space reference repair policy before a frozen
   humanoid tracker.
2. Dynamic Segment Replay that combines motion coverage, targeted revisitation,
   and variable-horizon evidence.
3. A paired objective that separates original-motion style from physical
   executability and regularizes unnecessary repair.
4. An auditable training/evaluation contract that uses one paired
   Style/Physics/Repair Gain decomposition for PPO credit, sampler evidence,
   and runtime diagnostics, while keeping their downstream roles distinct.
