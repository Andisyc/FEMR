# FRS-v015 Physics-Constrained Intent Task Canvas

Status: active volatile control surface. Updated: 2026-07-28.

## Objective

Keep the accepted Physics-constrained Intent method unchanged while replacing
the K8/M2 pilot training identity with one immutable coordinated K x exact-M
checkpointed curriculum.

## Method Authority

- Concept Figure: `Q-PAIR Paired Rollouts` -> `Q-01 Repair Gain`
- Active contracts: METHOD-v016 / GAIN-v006 / PPO-v004 / TRAIN-v011
- Current source route: METHOD-v016 / GAIN-v006 / PPO-v004 / TRAIN-v011,
  exact coordinated K x M and strict checkpoint-v6 at E-FI-89
- P0 decision record:
  `FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Implementation plan: completed P0-P4 history plus P5-A/P5-B/P5-C follow-on

## Current Cursor

`P5-B E-FI-89 exact-M/checkpoint-v6 offline engineering closure complete; P5-C awaits user-controlled material runtime authority`

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
- the complete v011 schedule is K8/M2 -> K16/M3 -> K32/M4 with two Segment
  sources, max absolute iteration 8000 and fixed review boundaries;
- `model_2000.pt` remains the K8/M2 checkpoint-v5 pilot and cannot resume v011.

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
-> P4 bounded official sensor-authority sentinel [complete at E-FI-84]
-> P4 Repair/Noisy applicability carrier closure [offline complete at E-FI-85]
-> P4 raw-contact capacity and formal K8 continuation [live complete at E-FI-86]
-> P4 policy-only deployment final consumer [offline complete at E-FI-87]
-> P4 single deployment quality run [ready]
-> P5-A coordinated K x M contract activation [complete at E-FI-88]
-> P5-B one-shot exact-M/checkpoint-v6 engineering closure [complete at E-FI-89]
-> P5-C checkpointed official campaign [ready; runtime authority required]
```

## Active Blockers

- pre-fix `model_2000.pt` contains four accepted updates whose post-rescale
  recovery direction violated one active first-order Physics halfspace;
- E-FI-80 repairs the owner and formal consumer offline; E-FI-84 later
  live-confirms one strict v006 sensor-authority transaction/checkpoint;
- E-FI-81 replaces formal root/capture-point ZMP evidence with contact-wrench
  ZMP and sealed Clean-foot envelopes; E-FI-84 confirms the server raw-contact
  API and finite supported-phase values;
- E-FI-82 defines valid physical no-load as Contact failure plus role-specific
  ZMP N/A; E-FI-85 now retains explicit Repair/Noisy aggregate applicability
  through GainResult, ReturnEvidence and both formal serializers;
- E-FI-86 increases raw-contact capacity to 256 per foot/env while retaining
  saturation fail-closed, and live-confirms 1999 committed transactions through
  absolute iteration 2000 with KKT max zero and a formal `model_2000.pt` save;
- E-FI-87 closes the policy-only JSON/CLI for Intent, real Contact,
  phase-ZMP, survival, sustained lean and unplanned Contact events; real values
  remain unavailable until the single deployment script is run.
- E-FI-89 installs the exact-M formal owner, isolates sampler trial state,
  enforces role width `4*M`, serializes active-M/count telemetry and persists
  strict checkpoint-v6; no material runtime evidence exists yet for M3/M4.

## Non-Scope

Simulator/training/live execution, multi-seed, matched-route evaluation, paper
experiments, HSL, actor observation/output changes, GMT changes, multiple
Critics/optimizers, Noisy physical prefix, or scalar reward-weight tuning.

## Next Action

P5-C is the next boundary: start a fresh strict HSL-v1 v011 campaign and use
the existing checkpoint boundaries and telemetry to review the first real
K16/M3 and K32/M4 transactions. This requires separate user authorization
because it starts simulator/training. P4 policy-only deployment remains an
independent open quality readout.
