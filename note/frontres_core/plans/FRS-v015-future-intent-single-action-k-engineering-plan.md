# FRS-v015 Physics-Constrained Intent Migration Engineering Plan

Status: active, volatile engineering plan. Updated: 2026-07-27.

## Authority

- Concept Figure: `../../architecture/concept/03_frontres_concept_tabs.data.json`
- Active contracts: FRS-METHOD-v016 / FRS-GAIN-v006 / FRS-PPO-v004 /
  FRS-TRAIN-v010
- Active source/runtime route: FRS-METHOD-v016 / FRS-GAIN-v006 /
  FRS-PPO-v004 / FRS-TRAIN-v010
- P0 decision record:
`FRS-GAIN-v005-vector-physics-constrained-intent-proposal.md`
- Checklist: `../checklists/FRS-v015-future-intent-single-action-k-checklist.md`
- Evidence: `../../testing/evidence_ledger_v015_future_intent_single_action_k_2026-07-19.md`

## Terminal Outcome

Preserve the Noisy zero-action counterfactual and all existing v015 replay
semantics, while replacing v004 scalar Physics/Intent utility with:

```text
scalar paired Intent objective - repair cost -> one scalar Critic
Contact / phase-ZMP / survival K-step evidence -> actor constraints
```

The final engineering closure must prove that Physics determines admissible
actor-update directions, Intent improves only inside that space, and one sealed
multi-Segment x M transaction still produces exactly one optimizer update and
one compatible checkpoint.

## Preserved Foundation

E-FI-0--E-FI-71 remain valid evidence for local scenarios, deployment q29 H,
928/158/770 observation authority, two roles, one action, frozen-GMT K evidence,
multi-Segment x M atomicity, grouped equal mass, proposal-only HSL, K-stage
Critic curriculum, exact-one update, and v004 runtime connectivity. They do not
prove that the v004 scalar target preserves Physics direction information.

The active source/runtime route now implements v016/v006/v004/v010. Historical
v004/v003/v009 identities remain isolated and reject on the formal route.

## Source Of Truth And Migration Surface

| Semantic object | Proposed owner | Current incompatible owner | Required isolation |
| --- | --- | --- | --- |
| Noisy baseline | existing sealed scenario / two-role collector | same owner | retain exactly one shared zero-action baseline; never score it as a PPO row |
| K-step Physics evidence | command + live probe + storage | evidence exists but is reduced for v004 | retain ordered Contact/phase-ZMP/survival and semantic masks until loss |
| scalar objective | `frontres_gain.py` v006 owner | v004 tiered `gain_total` | only paired Intent improvement minus repair cost reaches scalar Critic |
| constrained actor update | `frontres_segment_ppo.py` v004 owner | scalar-advantage PPO v003 | separate Physics constraint surrogates; no scalar Physics reward fallback |
| exact-one transaction | formal update owner | v009/v003 path | same sealed rows and grouped mass; one projected actor + scalar-Critic step |
| replay priority | existing sampler evidence owner | v004 scalar Gain projection | selection-only evidence; cannot multiply actor loss or recreate scalar Physics reward |
| persistence | `frontres_checkpointing.py` | checkpoint-v4 binds v004/v003/v009 | new identity rejects old objective/constraint state before mutation |
| diagnostics/evaluation | existing diagnostic owners | v004 deficit/utility summaries | raw per-channel/time constraints, projection facts, objective/value/advantage |

## Step Map

### P0 (Preparatory): Design Rebase And Owner Audit

Objective: record the confirmed Noisy/Physics/Intent authority split and expose
the remaining constrained-update decision.

Scope: v005 proposal, Concept Figure Q-PAIR/Q-01 wording, registry mismatch,
current Architecture, plan, checklist, canvas, evidence ledger, and read-only
CodeGraph owner/consumer audit.

Non-scope: active-contract supersession, source code, tests, checkpoint I/O,
simulator, training, or live run.

Evidence: E-FI-72 plus CodeGraph-confirmed route
`frontres_gain -> storage return -> grouped PPO -> formal update -> checkpoint`.

Stop: the user has not confirmed the exact constrained-update mechanism.

Why separate: selecting projection/recovery semantics changes the optimization
method and checkpoint identity; this is a real human scientific decision.

Status: completed in this document-only rebase.

### P1 / 4: Contract Activation And Constraint Mathematics

Objective: freeze one executable constrained-update rule before code changes.

Scope: define physical-unit Contact/ZMP/survival residuals, time aggregation
without saturation, grouped constraint surrogates, the constraint-gradient
projection/recovery rule, scalar Intent-Critic target, infeasible-case behavior,
selection-only replay-priority authority, and coordinated METHOD-v016 /
GAIN-v005 / PPO-v004 / TRAIN-v010 identities.
Activate contracts, registry, Concept Figure mapping, and current Architecture.

Non-scope: source code, test execution, checkpoint I/O, simulator, training,
live run, new actor input/output, second actor/Critic/optimizer, or HSL changes.

Expected evidence: formula fixtures showing that different severe ZMP/contact
states remain distinguishable; safe-state collisions are semantically
irrelevant; Noisy cannot waive an absolute Repair constraint; the scalar Critic
target contains no Physics mixture.

Stop: projection requires a second learned network, a hand-weighted scalar
Physics score, a hard adverse-row mask, an undefined infeasible fallback, or a
new actor-visible privileged signal.
Status: completed document-only. E-FI-73 activates the four coordinated
contracts, archives their superseded predecessors, and leaves source/runtime
deliberately blocked for P2.

### P2 / 4: One-Shot Offline Engineering Closure

Status: completed offline at E-FI-74. The active source route now implements
the v005 scalar/vector authority split, PPO-v004 grouped joint projection,
TRAIN-v010 gradient phases, and checkpoint-v5 strict identity. This is not
simulator or policy-quality evidence.

Objective: implement the complete new contract through the formal update and
persistence boundary in one local engineering unit.

Scope: existing evidence/result schemas, `frontres_gain.py`, storage/candidate
carrier, grouped PPO constrained-loss owner, formal exact-one update,
diagnostics/evaluation, checkpoint save/reload identity, legacy isolation, and
all focused deterministic S1/S2/S3 contracts. Reuse existing modules; do not
create a parallel training stack.
Unique owners: `frontres_gain.py` owns scalar target and physical residuals;
`frontres_segment_ppo.py` owns grouped actor/constraint gradients and their
joint projection; the existing formal transaction/update owner owns exact-one
commit; `frontres_checkpointing.py` owns checkpoint-v5 validation before any
state mutation; the existing training schedule owner implements v010 fresh
Critic entry and per-global-K recalibration.

Non-scope: simulator, real training, live run, long horizon experiment,
multi-seed, deployment composition, HSL, actor observation, GMT, H/K/M, or
group-mass changes.

Embedded checks:

- S1: raw K-step identity, phase/N-A semantics, no saturation collision,
  constraint projection/recovery, permutation and missing-evidence fail-closed;
- S2: one complete multi-Segment x M transaction carries the same Noisy
  baseline, scalar objective, vector constraints, equal group mass, and exactly
  one optimizer step; v004/v003 formal fallback rejects;
- S3: new checkpoint binds all contract/layout/curriculum/constraint identities,
 committed receipt and solver identity; no learned or persistent dual state is
 introduced; old v004/v009, tampered, or partial resume rejects before mutation;
- curriculum: first v010 entry fresh-initializes Critic/critic normalizer and
 optimizer while actor/std remain frozen; every global K increase preserves the
 same v010 Critic, re-enters critic-only recalibration, then actor ramp and joint;
- regression: 928/158/770, q29 provenance, one-action-K, K curriculum, HSL
  isolation, frozen GMT, and actor/Critic gradient authority remain unchanged.

Engineering acceptance: deterministic evidence proves the scalar Critic learns
only the Intent objective and actor gradients are transformed only by the named
Physics constraints, with no hidden scalar Physics utility.

Stop: any constraint evidence is silently filled, saturated, reduced before its
semantic owner, mixed across scenarios/roles/K, fed to the actor, or requires
more than one committed optimizer step.

### P3 / 4: Bounded Official Physics-Constraint Sentinel

Status: completed at E-FI-75. One real 8-env, 2-Segment x 2-attempt, K8
critic-only transaction produced finite Contact/phase-ZMP/survival evidence,
one constraint-recovery direction, exactly one optimizer update, and one
committed `frontres-v015-checkpoint-v5` artifact. This is runtime/connectivity
evidence, not actor-ramp efficacy or long-training admission.

Objective: prove the new route on one real 8-env sealed transaction.

Scope: one explicit Stage3-v015 engineering run at one K stage, one iteration,
one complete transaction, one optimizer update, one committed new-identity
checkpoint, log review, evidence/checklist/Architecture closeout.

Required telemetry: scenario/Noisy hashes, raw Contact/phase-ZMP/survival by K,
semantic masks, scalar Intent objective/return/value/advantage, each constraint
residual and gradient, projected actor direction, actor/Critic parameter deltas,
group mass, exact-one counts, and save identity.

Non-scope: long training, K transition, multiple seeds, deployment composition,
paper experiment, or method tuning.

Stop: missing raw evidence, v004/v003 fallback, constraint-gradient absence,
actor update that worsens every violated Physics channel, scalar-Critic Physics
contamination, nonfinite projection, or update count other than one.

Why separate: this is the simulator/material-cost and external-runtime boundary.

### P4 / 4: Policy-Quality Admission

Status: actor-ramp and joint training reached `model_2000.pt`, but E-FI-79 found
four post-rescale KKT violations in 1749 otherwise committed exact-one
transactions. E-FI-80 closes the deterministic source/consumer defect offline;
the pre-fix `model_2000.pt` remains a diagnostic/warm-start artifact rather
than a contract-clean final checkpoint.

Objective: decide whether the new target/update is informative enough to admit
actor-ramp training.

Scope: compare the bounded sentinel's Noisy/Repair causal evidence, constraint
movement, scalar objective/value calibration, action non-collapse, and demo-
quality hacking sentinels; then define the smallest justified training horizon.

Non-scope: automatic long training, multi-seed, deployment composition, or
paper claims.

Expected evidence: the new route distinguishes raw improvements hidden by v004,
does not reward sustained lean/unplanned stepping, and provides a nonzero
Physics correction direction when Repair is inadmissible.

Stop: no feasible/corrective direction, no-op collapse, systematic Physics
regression, Intent-only shortcut, or evidence that the chosen constrained
optimizer does not realize the confirmed concept. Such a result returns to P1,
not to coefficient tuning.

#### P4-S1: Readiness Closure Contract

Objective: close only the formal checkpoint-v5 continuation and quality-
evidence interfaces required before any critic continuation or actor-ramp run.

Unique orchestration owner:
`scripts/rsl_rl/train.py::main()`. The Stage3 shell launcher remains a connector;
`frontres_checkpointing.py::load_runner()`,
`frontres_segment_live_training.py::run_frontres_segment_live_training_loop()`
and `frontres_policy_quality_eval.py` remain their existing persistence,
training-loop and report semantic owners. No parallel runner or evaluator may
be created.

Scope:

- add one explicit Stage3-v015 checkpoint-v5 full-resume route, mutually
  exclusive with HSL-v1 initialization, restoring the exact actor, scalar
  Critic, optimizer, sampler, 928/158/770 normalizer identity, schedule,
  committed receipt history and absolute iteration;
- preserve `((8,200,500,0),)` and expose an exact bounded continuation from
  K8 phase iteration 1 to the critic-only boundary at iteration 200; no actor
  update is allowed inside this readiness closure;
- extend the existing atomic held-out report with read-only expected/actual
  Contact sequences, phase-ZMP applicability/violation/recovery trajectories,
  survival trajectories and an evaluator-only sustained lateral-lean trace;
- keep every new evaluation field outside actor observation, scalar Critic
  target, Gain/PPO, sampler, priority, optimizer and checkpoint identity.

Non-scope: changing TRAIN-v010 counts, running the remaining 199 critic-only
updates, actor-ramp, simulator/training/live execution, long training, multiple
seeds, deployment composition, HSL changes, Gain/PPO formulas, new actor input,
new learned network, or Concept Figure changes.

Focused evidence:

- S1 launcher/config: HSL initialization XOR strict v5 resume; legacy,
  unversioned, schedule-mismatched and partial checkpoints reject before
  mutation; exact resume begins at absolute iteration 1 and a bounded 199-
  update budget resolves to the iteration-200 critic boundary;
- S2 formal connectivity: a semantic checkpoint-v5 fixture restores actor,
  Critic, optimizer, sampler, prefix normalizer, curriculum and committed
  receipt, then one complete fake transaction advances exactly one iteration
  without reinitialization or a second optimizer step;
- S1/S2 quality: report projections preserve row order, scenario/hash/K,
  expected/actual Contact, phase-ZMP N/A masks and recovery trajectories,
  survival and sustained-lean traces; missing evidence fails closed rather
  than being filled with zero;
- S3 persistence/isolation: resumed save remains checkpoint-v5 with the same
  coordinated contracts and complete-or-idle transaction identity; quality
  evaluation leaves optimizer, sampler, transaction and normalizers unchanged.

Stop condition: stop before any simulator or training if resume reloads HSL,
resets or omits Critic/optimizer/sampler/iteration, changes the schedule or
928/158/770 identity, restores a partial transaction, permits more than one
update per committed transaction, or if the atomic report cannot expose raw
Contact/phase-ZMP and sustained-lean evidence without leaking evaluator-only
state into training.

Implementation result (E-FI-77): `train.py::main()` now owns an explicit
checkpoint-v5 full-resume route. The Stage3 launcher selects exactly one of
HSL-v1 initialization or v5 resume. `load_runner()` validates actor, Critic,
optimizer, sampler, 928/158/770 layout, TRAIN-v010 schedule and complete-or-
idle transaction identity before mutable restore, and preserves the last
committed receipt across an idle resumed save. The existing v015 held-out
evaluator now emits ordered expected/actual Contact, phase-ZMP applicability,
N/A, violation and recovery trajectories, raw survival, and evaluation-only
paired root-roll/cumulative-lean traces. Missing raw evidence fails closed.
Focused deterministic S1/S2/S3 contracts passed; no simulator or training ran.

#### P4-S2: K8 Critic-Only Continuation Closeout

Status: completed at E-FI-78.

Objective: exhaust the fixed TRAIN-v010 K8 critic-only budget without allowing
an actor/std update, while preserving one sealed transaction -> one optimizer
step and coordinated checkpoint-v5 persistence.

Live result: repository-root `v015_p4_critic_k8_to_200_gpu3.log` contains 199
formal transactions and 199 serialized transaction telemetry records covering
training iterations 1--199. Every accepted transaction has K8, four valid
policy rows, equal attempt mass `(0.25, 0.25, 0.25, 0.25)`,
`optimizer_step_delta=1`, `update_count=1`, actor weight zero, actor/std maximum
delta zero, and a nonzero finite Critic delta. Rejected scenarios did not step
the optimizer. The final coordinated save is
`/hdd1/cyx/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-07-24_14-58-03_P4_CRITIC_K8_TO_200/model_200.pt`
at absolute iteration 200 with METHOD-v016 / GAIN-v005 / PPO-v004 /
TRAIN-v010 and schedule `((8,200,500,0),)`. Its serialized `phase=actor_warmup`
describes the next update; all 199 executed updates remained `critic_only`.

Limit: changing scenarios across the 199 transactions makes aggregate
return/value drift unmatched evidence. E-FI-78 proves schedule, gradient-
authority, transaction and persistence closure, not Critic calibration or
policy efficacy.

#### P4-S3: First Actor-Ramp Bounded Sentinel Contract

Status: runtime-complete in the checkpoint lineage consumed by E-FI-79. The
current long-run log directly confirms strict full resume from `model_251.pt`;
the earlier single-update evidence is no longer the active plan boundary.

Objective: prove the first TRAIN-v010 actor-ramp update uses the constrained
PPO-v004 direction without crossing into a longer training or policy-quality
claim.

Unique orchestration owner: `scripts/rsl_rl/train.py::main()` using the
existing strict checkpoint-v5 resume connector,
`frontres_segment_live_training.py::run_frontres_segment_live_training_loop()`
and the existing coordinated save owner. No new runner or evaluator is
permitted.

Scope: resume the exact E-FI-78 `model_200.pt`; run one 8-env, K8, 2-Segment x
2-attempt sealed transaction and exactly one optimizer update; save the
committed iteration-201 checkpoint; review the final serialized telemetry and
refresh current evidence documents. The update must start at absolute
iteration 200 in `actor_warmup` with actor loss weight `1/500 = 0.002`.

Required S4 evidence:

- strict v5 resume restores absolute iteration 200, stage 0, K8, actor/Critic,
  optimizer, sampler, 928/158/770 normalizers and committed receipt without HSL;
- one complete transaction retains four valid rows and equal attempt mass,
  then reports exactly one optimizer step and one committed receipt;
- actor parameter delta becomes nonzero and finite under weight 0.002; std
  delta remains finite and follows its existing TRAIN-v010 authority, while
  the scalar Critic also changes and remains trained only on paired Intent
  improvement minus repair cost;
- Contact/phase-ZMP/survival evidence remains finite and row-aligned; the named
  projection/recovery status, directional derivatives and KKT residual prove
  that the selected actor direction obeys PPO-v004 rather than scalar Physics;
- the iteration-201 save remains checkpoint-v5 with the same four contract
  identities, schedule, constraint schema and complete transaction identity.

Non-scope: a second actor-ramp update, long training, numerical threshold
tuning, multi-seed, deployment composition, HSL/Gain/PPO/TRAIN changes, actor
observation changes, or a claim that one update establishes policy efficacy.

Stop condition: stop after the single committed update, or earlier on HSL/
legacy fallback, phase/iteration drift, actor weight other than 0.002, missing
or nonfinite Physics evidence, invalid projection/KKT facts, actor zero or
nonfinite actor/std delta, Critic zero/nonfinite delta, mixed K/scenario identity,
optimizer step count other than one, or non-v5/partial save. Any no-op,
systematic Physics regression, sustained-lean or unplanned-step evidence keeps
P4 quality admission blocked and returns to mechanism audit rather than longer
training.

#### P4 Long Training And KKT Postcondition Repair

Status: long training runtime-complete at E-FI-79; deterministic repair
contract-complete at E-FI-80; no post-fix simulator/training run has occurred.

Runtime result: `v015_train_to_model2000_gpu1.log` resumes `model_251.pt` and
executes 1749 unique K8 transactions through absolute iteration 2000. Every
transaction has four valid policy rows, equal grouped mass and exact-one
optimizer update. Actor-ramp ends at iteration 699 and joint training covers
700--1999. The final save reports `model_2000.pt` at iteration 2000.

Contradiction: iterations 445, 653, 1309 and 1394 report
`CONSTRAINT_RECOVERY` with positive post-rescale KKT residuals. The first
invalid owner is `frontres_segment_ppo.py`: a recovery direction is projected,
then norm-rescaled without re-establishing the active halfspace postcondition.
The formal telemetry consumer previously accepted every finite nonnegative KKT
value instead of enforcing checkpoint-v5 tolerance.

Repair: the algorithm now reprojects the norm-rescaled recovery direction and
accepts it only when every active directional derivative is at most the
versioned tolerance and at least one is strictly decreasing. Otherwise it
falls through to `NO_COMMON_FIRST_ORDER_DESCENT`. The formal telemetry adapter
rejects KKT above `1e-8` and rejects disagreement between reported KKT and the
serialized directional derivatives. Focused projection and formal-transaction
contracts plus Python compilation pass.

Non-scope: Gain, scalar Critic, actor/PPO loss, grouped mass, checkpoint format,
TRAIN-v010 schedule, simulator, training, deployment composition or a policy-
efficacy claim.

Stop: no further scale training may use the repaired code until the user
chooses whether the four-violation `model_2000.pt` is retained only as a
warm-start or a strictly clean checkpoint lineage is required.

#### P4 Physics Evidence Authority Closure

Status: deterministic S1/S2/S3 contract-complete at E-FI-81; no simulator,
training or live run has occurred.

Owner route: `commands.py` derives and seals Clean-continuation expected
Contact plus an oriented `[K,6]` foot support envelope; G1 config installs
separate filtered left/right foot-to-ground raw-contact sensors;
`frontres_segment_live_probe.py` computes world-frame contact-wrench ZMP and
signed margin against the expected envelope; existing v006 storage/Gain/PPO
consume the resulting phase-ZMP evidence unchanged. `frontres_checkpointing.py`
binds the estimator/carrier identity and rejects older checkpoint-v5 payloads
that do not name it.

Evidence: pure resultant/moment, translation, contact permutation,
inside/outside/flight, missing-resultant and raw-PhysX-buffer fixtures pass;
local-scenario hash/provenance, two-role reset, one-action-K, formal transaction,
928/158/770 observation isolation and strict save/resume contracts pass. The
old root/capture-point balance proxy remains only in the frozen observation
context and is unreachable from formal v006 Physics capture.

Non-scope: Gain/PPO mathematics, budgets/scales, actor/GMT observation, HSL,
TRAIN-v010 schedule, simulator, training, long-run lineage and deployment.

Stop: before any further training, one bounded official sensor-authority
sentinel must prove that the deployed IsaacLab version exposes raw points,
normals and force magnitudes with the configured terrain filters, yields finite
supported-phase ZMP, preserves flight N/A and saves the new strict checkpoint
identity. Missing raw API, supported-phase missing resultant, role/hash drift,
proxy fallback or actor-visible Clean geometry is a hard stop.

#### P4 Loaded-Support ZMP Applicability Closure

Status: E-FI-84 live-confirmed sensor authority; E-FI-85 closes the subsequently
exposed carrier mismatch offline. GainResult, ReturnEvidence, transaction
telemetry and held-out reports now retain explicit Repair and Noisy aggregate
applicability rather than inferring Noisy applicability from finite/NaN values.

FRS-GAIN-v006 resolves the first live sensor-authority contradiction without
changing the scalar objective or PPO projection. Valid ContactSensor no-load is
a scored Contact violation, not corrupt evidence. ZMP is role-specific N/A when
that role has no actual loaded support. Malformed raw payload and the converse
case, actual loaded support without a finite raw-wrench resultant, still fail
closed. One-action-K, paired Gain facts, return evidence, transaction telemetry,
atomic quality reports and checkpoint-v5 now carry this distinction. Strict
resume rejects GAIN-v005/schema-v1 before mutable restore.

The one-shot E-FI-85 closure makes Repair and Noisy aggregate applicability
explicit from GainResult through ReturnEvidence and both final serializers.
`zmp_constraint_applicable` remains the Repair-only PPO constraint mask;
`physics_zmp_gain` is finite only when both role masks are true. The four role
applicability combinations, row permutation, missing fields and fabricated
finite diagnostics must fail or pass according to FRS-GAIN-v006 before another
live or long run is admitted.

## Why Four Main Steps

The plan has four execution steps plus preparatory P0. The splits are not by
file or test tier:

- P1 is the completed scientific/optimization contract decision;
- P2 is the largest safe local implementation and verification closure;
- P3 requires simulator/material-cost authority;
- P4 is the policy-quality and longer-training authorization boundary.

Removing any one would either hide a human semantic decision, cross a costly
runtime boundary, or admit training without causal evidence. All owner edits,
focused tests, formal connectivity, persistence, and routine document refresh
are deliberately merged into P2.

## Cursor

Current cursor: `P4 ZMP applicability carrier closure complete offline at E-FI-85; one bounded/continued server run may now validate the repaired formal path before further scale training`.
