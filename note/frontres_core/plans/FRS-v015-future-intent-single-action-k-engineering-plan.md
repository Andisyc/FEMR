# FRS-v015 One-Shot HRL Engineering Plan

Status: active, volatile engineering plan. Updated: 2026-07-22.

This plan applies `one-shot-execution`: owner changes, focused tests, formal
connectivity, one bounded smoke, evidence capture, and documentation refresh are
embedded checks inside one engineering unit. They are not separate user-visible
approval steps.

## Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Contract registry: `../contracts/README.md`
- Active contracts: Method v015, Training v008, Gain v004, PPO v003, Eval v003
- Acceptance checklist: `../checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- Evidence ledger: `../../testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`

The accepted design delta is entirely inside existing `Q-PAIR` and `Q-01`:
expected support-mode preservation, Contact-phase-conditioned ZMP, and
non-compensatory Physics admissibility. It does not add a top-level Contact
module. The same G5-P1 now also closes `M-05`: v004 critic-only calibration,
linear actor takeover, then joint PPO. It does not add an actor, Critic,
optimizer, HSL target, or change H, K, transaction, or grouped PPO semantics.

## Preserved Foundation

`G0--G4`, the G5 formal route, and `E-FI-0--E-FI-65` remain valid evidence for:

- immutable local scenarios and deployment-provenance q29 Intent;
- `928D = 158D FEMR + 770D frozen GMT` authority;
- Repair/Noisy roles, one FEMR action, and K-step frozen-GMT evidence;
- one policy row per attempt, multi-Segment x M transaction, grouped reduction,
  exact-one update, and committed Stage3-v015 persistence;
- strict HSL-v1 actor-only initialization;
- the existing scalar Critic, 289D critic observation, value/return/advantage
  carrier, and historical Warmup scheduler/tests;
- complete paired survival/ZMP/contact carrier and row-aligned credit
  diagnostics.

Those facts do not validate the v004 Gain. `E-FI-64` showed the formal additive
v003 route could produce harmful repairs, and `E-FI-65` still used a foot-height
contact proxy and additive available-mean Physics. E68--E70 validate Warmup only
on the older update/v002 Gain route. HSL remains frozen.

## Closure State

G5-P1 implementation, focused S1/S2/S3 contracts, and the bounded S4
critic-only transaction are complete at `E-FI-68`. The formal route now uses
the sealed expected-support carrier, actual `contact_forces`, phase-conditioned
ZMP, FRS-GAIN-v004 consumers, FRS-TRAIN-v008 phase selection, critic-only
actor/std isolation, and v008/v004 persistence identity.

The S4 transaction is an engineering proof, not a policy-quality claim. All
four sampled Repair rows remained Physics-inadmissible, so both role utilities
were tied at the unsafe tier and `gain_total` consisted only of the negative
bounded repair penalty. This is admissible Critic calibration data and the
Critic updated, but actor-ramp quality remains unconfirmed until X1.

## G5-P1: One-Shot v004 Physics Gain And Critic-Ready Actor Migration

### Terminal outcome

The official Stage3-v015 path uses one immutable expected support sequence,
authoritative actual ContactSensor evidence, phase-conditioned ZMP, and the
single scalar FRS-GAIN-v004 utility in every formal consumer. A cold HSL-v1
actor then enters v004 critic-only calibration before actor ramp and joint PPO.
One bounded 8-env critic-only transaction proves the formal route without
changing the frozen method boundaries.

### Unique owners and data route

| Semantic object | Unique owner | Input -> output |
| --- | --- | --- |
| Expected support carrier | `commands.py::MultiMotionCommand.materialize_frontres_local_scenario()` | same ordered Clean frames and materializer-owned reference foot kinematics -> immutable left/right support modes `[K,2]` plus phase/tolerance identity; `[K,65]` remains the GMT command |
| Actual Contact evidence | `frontres_segment_live_probe.py::_capture_physics_frame()` | `contact_forces` ContactSensor + role rows -> paired actual contact `[K,B,2]` |
| Immutable paired facts | `frontres_segment_storage.py` | expected/actual contact, ZMP, survival, valid masks, scenario identity -> one-action-K paired facts |
| v004 ordering | `frontres_gain.py` | per-role Physics admissibility/deficit + Intent quality + repair cost -> one scalar paired Gain `[B]` |
| Warmup phase | `frontres_segment_warmup.py::frontres_segment_warmup_phase()` | persisted iteration plus explicit `N_c/N_a` -> phase and actor loss weight |
| Phase-aware formal update | `frontres_segment_live_probe.py::run_frontres_v015_formal_transaction_update()` | complete v004 transaction + phase -> critic-only / actor-ramp / joint exact-one update |
| Warmup persistence | `frontres_checkpointing.py` | v008/v004 identity + `N_c/N_a` + absolute iteration/phase -> exact save/resume or fail-closed |

Expected support is materialized once from the same Clean frame identities
already gathered by the command owner; it must be covered by the sealed
scenario hash and reused by all M attempts. Actual Contact must come from the configured sensor. The
existing `contact_state` field is the allowed carrier; no predictor, label, or
actor input is added.

### Formal consumers

The same `gain_contract_id=FRS-GAIN-v004` result must reach:

```text
immutable storage
-> return and advantage
-> replay priority evidence
-> grouped PPO transaction
-> live diagnostics
-> local and held-out evaluation
```

Every v015 formal consumer rejects v002/v003 fallback, missing component
zero-fill, mixed scenario/phase identity, and partial transaction evidence.
The phase is fixed for the complete transaction. PPO reduction, optimizer
count, scalar Critic interface, and value-loss formula remain unchanged.

### Scope

- derive expected support modes from the GMT-only Clean continuation without
  actor exposure;
- read left/right actual contact from the existing ContactSensor;
- implement timing-tolerant planned/extra/missed/dragging Contact alignment;
- implement support-phase ZMP masks, domains, transition recovery, and flight
  `N/A`;
- implement non-compensatory admissibility/deficit/Intent tier ordering and
  bounded repair penalty in the existing scalar Gain owner;
- migrate every formal v015 consumer and diagnostic to v004;
- connect the existing Warmup scheduler to the formal v015 owner;
- require explicit nonzero `N_c/N_a` for formal training, with current
  engineering defaults 200/500;
- in critic-only, update only the scalar Critic while actor/std remain exactly
  unchanged; then linearly ramp actor loss while Critic remains enabled;
- bind v008/v004, schedule, absolute iteration, and phase into Stage3
  checkpoint/resume identity and reject v003/v007 resume before mutation;
- run focused deterministic, formal-connectivity, and one bounded live proof;
- append evidence and refresh checklist/canvas/Architecture inside the same
  closure.

### Non-scope

- actor observation, action shape, scalar Critic architecture, critic
  observation, HSL, normalizer, or generic checkpoint-format changes;
- `rho`, dual output, serial networks, second Critic/optimizer, contact
  predictor, or new actor input;
- Noisy physical prefix, noise label, perturbation time, Clean actor future, or
  future root/global actor input;
- changing H, K, one-action-K, sealed transaction, grouped PPO, or optimizer
  formula;
- long training, multi-seed, deployment composition, or paper experiments.

### Embedded evidence

S1 deterministic contracts must cover support codes `11/10/01/00`, left/right
identity, planned step, static support, early/late tolerance, extra/missed
switches, dragging, transition recovery, flight ZMP masking, survival failure,
both-safe/both-unsafe/cross-tier ordering, no-op, missing/non-finite evidence,
row/role permutation, actor-input exclusion, all Warmup phase boundaries,
actor-weight formula, and critic-only actor/std invariance.

S2 formal connectivity must prove one sealed support carrier across M attempts,
real ContactSensor owner reachability, `[K,B,2]` row alignment, shared valid
masks, all v004 consumers, unchanged one-row/grouped/exact-one behavior, and
v002/v003 isolation. The same formal v015 owner must prove v004 return reaches
the Critic during critic-only, actor/std stay fixed, actor-ramp weight is
monotonic, and no legacy update connector is used.

S3 persistence must prove cold HSL-v1 starts at iteration 0 critic-only, exact
v008/v004 save/reload preserves `N_c/N_a` and phase, and v003/v007/unversioned
resume rejects before state mutation.

S4 bounded evidence is one 8-env complete critic-only transaction. It must record
expected/actual contact sequences, switch violations, phase/ZMP masks and
recovery, survival, per-role admissibility/deficit, Intent quality, paired
utility, repair cost, Gain, return, raw/scaled advantage, gradient, group mass,
Warmup phase/weight, actor/std zero delta, nonzero Critic delta, exact-one
update, committed checkpoint, and v008/v004 identity. Deterministic fixtures
cover actor-ramp/joint; actual long phase progression belongs to X1.

### Stop conditions

Stop the one-shot implementation at the first true design boundary if:

- expected support cannot be deterministically derived from Clean continuation
  while remaining evaluator-only;
- `contact_forces` cannot provide stable left/right actual contact;
- role/foot/phase/scenario identity cannot stay sealed across attempts;
- planned aggressive motion cannot be separated from Repair-induced stepping
  without a new label or model;
- numerical scaling allows Intent or repair cost to invert the Physics tier;
- any formal consumer requires mixed v003/v004 evidence or a frozen method
  boundary must change;
- the formal owner falls back to `run_frontres_segment_single_update()`, still
  rejects nonzero Warmup, changes actor/std during critic-only, gives the Critic
  no gradient, crosses phase inside a transaction, restarts phase on resume, or
  accepts a v003/v007 Stage3 checkpoint;
- after one complete implementation/repair cycle, the bounded live route is
  still contradictory, no-op, regressing, or harmful.

## X1: Formal Experiments And Composition

X1 remains the next high-cost boundary only after G5-P1 passes. It owns the
training budget/seeds, checkpoint trajectory, paired deployment composition,
and paper artifacts. It is not authorized by this plan rebase.

## Cursor

Current cursor: `G5-P1 completed at E-FI-68; X1 remains the next high-cost,
separately authorized boundary`.

No actor-ramp progression, long training, multi-seed run, deployment
composition, or paper experiment has been authorized or executed.
