# FRS-v015 Physics-Constrained Intent Task Canvas

Status: active volatile control surface. Updated: 2026-07-27.

## Objective

Keep the Noisy zero-action counterfactual, one full-6D actor, one scalar Critic,
one-action-K and exact-one grouped transaction, while moving Contact, phase-ZMP
and survival out of scalar Gain and into explicit actor-update constraints.

## Method Authority

- Concept Figure: `Q-PAIR Paired Rollouts` -> `Q-01 Repair Gain`
- Active contracts: METHOD-v016 / GAIN-v006 / PPO-v004 / TRAIN-v010
- Active source route: METHOD-v016 / GAIN-v006 / PPO-v004 / TRAIN-v010,
  runtime-confirmed through one bounded transaction at E-FI-75
- P0 decision record:
  `FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Implementation plan: four main steps P1-P4 plus completed preparatory P0

## Current Cursor

`E-FI-84 live-confirms sensor authority and checkpoint-v5; return-evidence ZMP N/A is repaired offline and the post-fix long rerun remains open`

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
-> P4-S2 K8 critic-only continuation to iteration 200 [complete]
-> P4-S3 actor-ramp lineage [runtime-complete]
-> P4 long training to model_2000 [runtime-complete; four pre-fix KKT violations]
-> P4 post-rescale KKT repair [offline contract-complete]
-> P4 contact-wrench ZMP authority [offline S1/S2/S3 complete]
-> P4 loaded-support applicability [offline S1/S2/S3 complete]
-> P4 bounded official sensor-authority sentinel [open]
-> P4 policy-quality/checkpoint-lineage decision [open]
```

## Active Blockers

- pre-fix `model_2000.pt` contains four accepted updates whose post-rescale
  recovery direction violated one active first-order Physics halfspace;
- E-FI-80 repairs the owner and formal consumer offline, but no post-fix live
  transaction or new checkpoint lineage exists;
- E-FI-81 replaces formal root/capture-point ZMP evidence with contact-wrench
  ZMP and sealed Clean-foot envelopes offline, but the server IsaacLab raw
  contact API and finite supported-phase values are not live-confirmed;
- E-FI-82 removes the invalid assumption that expected support alone implies a
  finite ZMP: valid physical no-load is Contact failure and role-specific ZMP N/A;
- aggregate training telemetry shows improved Critic calibration but no clear
  Intent/Gain improvement, and contains no sustained-lean evaluation field.

## Non-Scope

Long training, multi-seed, deployment composition, paper experiments, HSL,
actor observation/output changes, GMT changes, multiple Critics/optimizers,
Noisy physical prefix, or scalar reward-weight tuning.

## Next Action

Choose checkpoint lineage after E-FI-80: retain pre-fix `model_2000.pt` only as
a warm-start and continue with repaired projection, or require a strictly clean
lineage from the last checkpoint before the first recorded violation. No
further training is implied by this document update.
