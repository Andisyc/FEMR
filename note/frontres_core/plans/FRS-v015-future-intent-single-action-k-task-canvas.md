# FRS-v015 Physics-Constrained Intent Task Canvas

Status: active volatile control surface. Updated: 2026-07-23.

## Objective

Keep the Noisy zero-action counterfactual, one full-6D actor, one scalar Critic,
one-action-K and exact-one grouped transaction, while moving Contact, phase-ZMP
and survival out of scalar Gain and into explicit actor-update constraints.

## Method Authority

- Concept Figure: `Q-PAIR Paired Rollouts` -> `Q-01 Repair Gain`
- Active contracts: METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010
- Active source route: METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010,
  offline-confirmed at E-FI-74
- P0 decision record:
  `FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Implementation plan: four main steps P1-P4 plus completed preparatory P0

## Current Cursor

`P2 / 4 complete at E-FI-74; P3 / 4 awaits simulator/material-cost authorization`

## Confirmed

- Noisy rollout is retained as the same-scenario zero-action baseline.
- Noisy answers whether FEMR caused improvement; it does not define Physics
  admissibility.
- Repair must satisfy expected Contact, phase-conditioned ZMP and survival
  independently of whether it is less bad than Noisy.
- scalar Critic target is paired Intent improvement minus repair cost only.
- raw signed/per-step/per-channel Physics evidence remains available until the
  optimization boundary.
- no rho, second actor, second Critic, contact predictor, new actor input, HSL
  change, Noisy prefix, or deployment Noisy rollout is introduced.

## Contradicted V004 Assumption

The C4/ZMP plateau evidence shows that `[0,1]` violation normalization,
temporal/channel `amax`, and unsafe scalar utility can map physically distinct
Noisy/Repair trajectories to the same target. Critic warmup cannot recover
evidence deleted before return construction.

## Active Steps

```text
P0 document/owner rebase [complete]
-> P1 constrained-update mathematics + contract activation [complete]
-> P2 one-shot offline implementation/S1/S2/S3 [complete]
-> P3 one bounded 8-env official sentinel [runtime authorization]
-> P4 policy-quality / longer-training admission [human decision]
```

## Non-Scope

Long training, multi-seed, deployment composition, paper experiments, HSL,
actor observation/output changes, GMT changes, multiple Critics/optimizers,
Noisy physical prefix, or scalar reward-weight tuning.

## Next Action

Authorize P3 only when ready to cross the simulator/material-cost boundary for
one 8-env, one-transaction, one-update official sentinel. P3 must record raw
Physics evidence, scalar Intent/value/advantage, constraint gradients and joint
projection, actor/Critic deltas, exact-one counts, and checkpoint-v5 identity.
It does not admit long training or policy-quality claims.
