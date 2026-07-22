# FRS-v015 Task Canvas

Status: active volatile control surface. Updated: 2026-07-22.

## Objective

Replace the harmful additive Physics/Intent tradeoff with one scalar
FRS-GAIN-v004 ordering, then calibrate the existing scalar Critic before
gradually releasing the HSL-initialized actor. Preserve the one-actor,
one-Critic, one-optimizer architecture.

## Method Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Registry: `../contracts/README.md`
- Active contracts: Method v015 / Training v008 / Gain v004 / PPO v003 /
  Eval v003
- Design points remain `Q-PAIR` and `Q-01`; no top-level Contact module exists.
- `M-03` remains proposal-only HSL; `M-05` owns critic-only -> actor-ramp ->
  joint PPO.

## Current Cursor

`G5-P1 - v004 Gain and Critic-ready Actor Curriculum rebased; current formal code contract-mismatch; implementation awaiting explicit authorization`

## Preserved Evidence

`E-FI-0--E-FI-65` preserves the sealed q29/928/158/770/two-role/one-action-K/
grouped/exact-one/HSL/persistence foundation. `E-FI-64` triggered the method
boundary with harmful repairs; `E-FI-65` proved complete paired Physics carrier
connectivity but retained height-proxy Contact and additive v003 semantics.
Historical E68--E70 prove Warmup scheduling only on the old update/v002 Gain
route. The latest v015 S4 run explicitly used critic/actor Warmup `0/0`.

## Design Delta

```text
same Clean continuation frame identities + reference foot kinematics
-> evaluator-only expected support sequence 11/10/01/00

contact_forces ContactSensor
-> actual left/right Contact sequence

expected Contact phase + actual Contact + ZMP + survival
-> non-compensatory Physics admissibility / deficit

Physics tier first, then Intent, then bounded repair cost
-> one scalar paired Gain

HSL-v1 actor + fresh scalar Critic
-> critic-only v004 calibration
-> linear actor takeover with Critic still learning
-> joint grouped PPO
```

Clean continuation remains GMT/Physics-evaluator evidence only. It never enters
actor observation, Intent target, or deployment input.

## Active Files

- `note/frontres_core/contracts/active/reward/FRS-GAIN-v004-support-mode-physics-admissibility.md`
- `note/frontres_core/contracts/active/training/FRS-TRAIN-v008-critic-ready-v004-actor-curriculum.md`
- `note/frontres_core/plans/FRS-v015-future-intent-single-action-k-engineering-plan.md`
- `note/frontres_core/checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- `note/testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`
- `note/architecture/concept/03_frontres_concept_tabs.data.json`
- `note/architecture/runtime/02_frontres_flow.data.json`
- `note/architecture/runtime/05_policy_quality_audit.data.json`

## Unresolved Implementation Facts

- exact reference-foot threshold/velocity rule for expected-contact derivation
  during Clean continuation materialization;
- production owner that populates `contact_state` from `contact_forces`;
- timing tolerance and transition recovery carrier identities;
- v004 utility normalization that proves tier separation without a second
  output or optimizer;
- complete migration/isolation of return, priority, diagnostics, and evaluators.
- formal v015 Warmup phase consumption, actor/std gradient isolation, and
  v008/v004 checkpoint/resume identity.

## Stop Rule

Stop if expected support needs actor-visible Clean data, actual Contact cannot
come from the existing sensor, phase identity cannot remain sealed, planned
aggressive motion requires a new predictor/label, Intent/cost can invert the
Physics tier, formal v015 must fall back to the legacy update owner, actor/std
changes in critic-only, Critic receives no v004 gradient, resume restarts the
schedule, or a frozen H/K/actor/Critic/PPO/HSL boundary must change.

## Next Action

After explicit authorization, execute G5-P1 as one closure: implement all
owners and consumers, run focused S1/S2/S3 contracts, run exactly one bounded
8-env critic-only transaction, inspect v004 Physics/Intent/credit and separate
actor/std/Critic telemetry, and refresh evidence/Architecture. Do not enter X1,
long training, multi-seed, actor-ramp progression, or deployment composition.
