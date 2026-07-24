# FRS-v015 Physics-Constrained Intent Task Canvas

Status: active volatile control surface. Updated: 2026-07-24.

## Objective

Keep the Noisy zero-action counterfactual, one full-6D actor, one scalar Critic,
one-action-K and exact-one grouped transaction, while moving Contact, phase-ZMP
and survival out of scalar Gain and into explicit actor-update constraints.

## Method Authority

- Concept Figure: `Q-PAIR Paired Rollouts` -> `Q-01 Repair Gain`
- Active contracts: METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010
- Active source route: METHOD-v016 / GAIN-v005 / PPO-v004 / TRAIN-v010,
  runtime-confirmed through one bounded transaction at E-FI-75
- P0 decision record:
  `FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Implementation plan: four main steps P1-P4 plus completed preparatory P0

## Current Cursor

`P4-S1 readiness closure completed at E-FI-77; critic continuation and actor-ramp remain unrun`

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
-> P3 one bounded 8-env official sentinel [complete]
-> P4-S0 policy-quality admission audit [complete]
-> P4-S1 resume + quality-evidence readiness closure [complete]
-> P4 critic continuation / actor-ramp admission [awaiting separate human decision]
```

## Active Blockers

- checkpoint-v5 is K8 `critic_only 1/200`; 199 additional critic-only updates
  would be required to reach the fixed TRAIN-v010 actor-ramp boundary, but no
  continuation run is authorized;
- no engineering-readiness blocker remains in the strict checkpoint-v5 route;
  the remaining 199 critic-only updates are intentionally unrun;
- policy efficacy and the numerical actor-ramp admission boundary remain
  unresolved experimental decisions, not missing wiring.

## Non-Scope

Long training, multi-seed, deployment composition, paper experiments, HSL,
actor observation/output changes, GMT changes, multiple Critics/optimizers,
Noisy physical prefix, or scalar reward-weight tuning.

## Next Action

Decide separately whether to run the remaining K8 critic-only continuation.
P4-S1 provides the strict resume command surface and complete atomic quality
evidence, but it does not authorize simulator/training/live execution,
actor-ramp length or numerical admission thresholds.
