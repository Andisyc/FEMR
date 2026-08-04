---
contract_id: FRS-PPO-v004
status: superseded
effective_date: 2026-07-23
updated_date: 2026-08-01
superseded_by: FRS-PPO-v005
supersedes: FRS-PPO-v003
scope: equal-mass grouped PPO with one scalar Intent advantage, three independent Physics constraint gradients, joint first-order projection/recovery, and one optimizer step per sealed transaction
---

# Grouped First-Order Constraint-Gradient Projection

## Preserved PPO Domain

One eligible Repair attempt contributes one policy row containing one sampled
full-6D action, old policy statistics, scalar Intent return/advantage, vector
constraint evidence, and complete v016 transaction identity. K remains evidence
horizon only. Noisy, Clean, search, oracle, and invalid rows remain excluded.

The motion -> Segment -> attempt equal-mass reduction and sign-preserving scalar
Intent-advantage scaling from FRS-PPO-v003 are unchanged. Replay priority, K,
M, evidence length, constraint severity, or solver status may not change row
mass.

## Scalar Intent PPO Direction

Let `J_I(theta)` be the existing grouped clipped PPO actor surrogate using only
the v006 scalar Intent advantage `A_I`. Let

```text
p_I = grad_theta J_I(theta)
```

be the unprojected actor-ascent direction over all trainable actor-distribution
parameters, including std. The scalar value loss uses only `return_K=y_I` and
is not part of the actor projection.

## Independent Grouped Constraint Gradients

For `j in {contact,zmp,survival}`, v006 supplies detached per-row constraint
advantages `A^c_ij`. At the frozen old-policy transaction boundary:

```text
S_j(theta) = grouped_equal_mass_mean(
    policy_ratio_i(theta) * A^c_ij
)

g_j = grad_theta S_j(theta)
```

`g_j` points toward increasing expected violation. The three gradients retain
their names and are never added to the scalar Critic loss. Invalid/missing
constraint evidence rejects the complete transaction. An explicitly N/A ZMP
family is excluded from the active set and reported as N/A, not zero evidence.

## Active Constraint Set

The grouped absolute violation level is `C_j` from FRS-GAIN-v006. Define:

```text
A = {j | C_j > 0 and ||g_j|| > eps_grad}
```

A violated family with `||g_j|| <= eps_grad` is `NO_EMPIRICAL_DIRECTION`; it is
not silently dropped. The actor step then enters recovery/fail-closed handling.
`eps_grad` is a fixed versioned numerical tolerance, not a method coefficient.

## Joint First-Order Projection

For a parameter increment `p`, first-order non-worsening of active constraint
`j` requires `g_j^T p <= 0`. The primary projected direction is the unique
minimum-norm solution:

```text
p_proj = argmin_p 0.5 * ||p - p_I||_2^2
         subject to g_j^T p <= 0  for every j in A
```

All active constraints enter one solve. Sequential pairwise projection is
forbidden because it is order-dependent. The solver may use the equivalent
three-variable nonnegative dual:

```text
p_proj = p_I - G^T lambda*
lambda* = argmin_{lambda >= 0}
          0.5 * ||p_I - G^T lambda||_2^2
```

where rows of `G` are active `g_j^T`. KKT primal feasibility, dual feasibility,
complementarity, finite status, and deterministic family order must be checked.

If `A` is empty, `p_proj=p_I` exactly.

## Recovery And No-Common-Descent Behavior

If `p_proj` is finite, nonzero, satisfies every active halfspace, and strictly
decreases at least one violated family, it is accepted. Otherwise construct a
constraint-only common-recovery direction without a scalar Physics reward:

```text
n_j = g_j / ||g_j||
r_seed = mean_j n_j
r_proj = projection of (-r_seed) onto {p | g_j^T p <= 0 for all j in A}
```

The accepted recovery direction must be finite, nonzero, non-worsening for
every active family, and strictly decreasing for at least one. Its raw norm is
rescaled to

```text
max(||p_I||, RMS_j ||g_j||)
```

before the existing global actor-gradient clipping. This scale is derived from
the current named gradients; it is not a reward weight or persistent dual.

If projection/recovery cannot meet those conditions, status is
`NO_COMMON_FIRST_ORDER_DESCENT`. Actor and std gradients are set exactly to
zero for the transaction. The scalar Critic gradient remains legal, and the
single optimizer step may update only Critic parameters. The transaction logs
and persists the status; it never falls back to `p_I`, a weighted sum, row
masking, or another optimizer step.

## Warmup And Exact-One Update

FRS-TRAIN-v011 supplies `actor_loss_weight=w in [0,1]` for each coordinated
K x M stage. The solver first obtains
the unscaled `p_proj` or recovery direction, then applies:

```text
p_actor = w * p_accepted
```

Thus critic-only gives exact actor/std zero delta and actor ramp scales every
permitted/recovery direction consistently. Critic and projected actor gradients
are installed into their disjoint parameter sets, followed by exactly one call
to the existing optimizer step after the complete transaction seals.

The projected gradient is not the final Actor authority when the shared
optimizer has momentum or coordinate-wise preconditioning. Let
`delta_adam` be the Actor/std parameter increment produced by that single
optimizer call. Before commit, the same active halfspaces must be imposed on
the actual increment:

```text
delta_actor = projection(delta_adam,
                         {delta | g_j^T delta <= 0 for every j in A})
```

The committed parameter delta, not only the pre-optimizer gradient, must pass
the KKT/postcondition checks. For critic-only,
`NO_EMPIRICAL_DIRECTION`, or `NO_COMMON_FIRST_ORDER_DESCENT`, both Actor/std
parameters and their optimizer state are restored exactly after the one shared
optimizer call; Critic state remains eligible to advance. This restoration is
part of the same exact-one transaction and cannot create a second optimizer
step.

## Implementation Ownership

The existing algorithm owner `frontres_segment_ppo.py` performs the one shared
optimizer call, measures Adam's candidate Actor/std delta, applies the actual
parameter-space authority above, and commits or restores Actor/std state. The
formal live probe only supplies the sealed batch/snapshots and records the
detached result. `frontres_segment_diagnostics.py` validates those detached
postconditions before serialization; diagnostics may never feed back into the
optimizer. This ownership split is behavior-preserving and does not create a
new optimizer, runner, checkpoint field or projection formula.

## Replay Priority Boundary

Replay may use explicit named constraint/frontier diagnostics for selection or
bucket quotas. Those facts are detached and selection-only. They may not enter
`A_I`, `A^c_j`, grouped mass, actor-loss multipliers, projection scaling, or
Critic targets.

## Required Diagnostics And Identity

Every update reports:

- scalar objective/return/value/raw and scaled Intent advantage;
- raw per-family residuals, scales, `C_j`, constraint advantages, and gradient
  norms;
- pairwise constraint-gradient Gram matrix and objective-gradient dot products;
- active/N/A/no-direction families, solver/KKT status, dual coefficients,
  projected/recovery norm, pre-optimizer and actual-update KKT, and each
  `g_j^T delta_actor`;
- optimizer-candidate and committed Actor-delta norms plus exact Actor
  optimizer-state restoration status;
- actor/std/Critic parameter deltas, grouped mass, exact-one update count;
- `optimization_contract_id=FRS-PPO-v004` and projection schema fingerprint.

No persistent dual or learned constraint state exists. Checkpointing binds the
solver configuration/tolerances and schema identity only.

## Forbidden Behavior

- scalar Physics advantage, weighted actor loss, or cost Critic;
- sequential/order-dependent gradient surgery;
- treating a missing/zero empirical constraint gradient as safe;
- restoring unprojected Intent after solver failure;
- dropping adverse valid rows;
- a second optimizer step for constraints;
- changing equal grouped mass, one-row K semantics, HSL, or actor inputs.

## Acceptance And Stop Conditions

P2 S1 must cover feasible, inactive, conflicting, zero-gradient, recovery,
permutation, KKT, finite, and actor-ramp fixtures. S2 must prove one complete
transaction produces one scalar Critic gradient, one jointly projected actor
gradient, and exactly one optimizer step with v003 fallback rejected. S3 must
bind the unchanged solver schema in checkpoint-v6. K/M scheduling does not
alter this projection formula.

Stop if the QP result depends on constraint order, an infeasible case updates
actor/std parameters or optimizer state, an Adam-preconditioned committed
increment violates an active halfspace, a constraint reaches the Critic,
multiple optimizer steps occur, or the implementation requires a learned
dual/cost network.
