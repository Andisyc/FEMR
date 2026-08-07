# FRS-TRAIN-v015 Fixed Split-LR One-Shot Engineering Plan

Status: implemented and offline-verified; bounded official transaction pending

## Terminal Outcome

The existing FEMR Stage-3 route uses one `FrontRESTrackedAdam` with exactly two
named parameter groups:

```text
actor:  residual_actor parameters, lr=3e-6
critic: scalar Critic parameters, lr=1e-5
```

The task-space exploration std remains a fixed buffer. The schedule is fixed,
so no KL adapter may overwrite either LR. A fresh Stage-3 campaign starts from
the accepted HSL-v2 actor with a fresh Critic and optimizer. New checkpoints
round-trip the two-group identity; checkpoint-v9 and malformed group layouts
reject before mutable restoration.

This plan closes engineering readiness only. It does not claim that the chosen
LRs improve policy quality.

## Accepted Behavior And Non-Scope

Accepted behavior:

- preserve one optimizer and one scalar `G_total` Critic;
- preserve `critic_only -> actor_ramp -> joint` and the active per-K counts;
- preserve disjoint gradients: Actor receives the weighted PPO actor loss and
  Critic receives only the value loss;
- preserve exact two-Segment x M sealing and exactly one optimizer step per
  committed transaction;
- preserve frozen GMT, direct finite `[B,6]` Delta SE(3), 158D actor input,
  fixed task-space std, FRS-GAIN-v007 and FRS-PPO-v005;
- persist group names, disjoint membership, both configured LRs, shared step
  count, optimizer moments and transaction identity.

Non-scope:

- no MOSAIC code or configuration change;
- no second optimizer, scheduler, Actor, Critic or loss;
- no network, observation, action, Gain, PPO reduction, K/M/DR or simulator
  behavior change;
- no migration of optimizer/Critic/curriculum state from an existing Stage-3
  checkpoint;
- no long training, policy-quality verdict or deployment claim.

## Causal And Evidence Boundary

Observed facts:

- the inspected Stage-3 checkpoints persisted one shared `1e-6` LR;
- that run showed very small Actor functional movement and weak held-out Critic
  calibration;
- current code constructs one optimizer group and the Segment adaptive setter
  writes one LR into every group.

Interpretation boundary:

- these facts establish that the current configuration cannot express separate
  Actor/Critic time scales;
- they do not prove that split LR is the unique cause or that `3e-6/1e-5` will
  improve policy quality;
- the user-selected intervention is therefore a new bounded campaign identity,
  not a retroactive bug-fix claim about the previous run.

Falsifier: if the completed official single-update sentinel does not expose two
stable groups with the exact LRs, a frozen Actor during critic-only, a changing
Critic, one shared optimizer step count and an exact reload, engineering closure
fails and long training remains blocked.

## Source-Of-Truth Migration

| Object | Single owner after change | Consumers | Retired or rejected path |
| --- | --- | --- | --- |
| training semantics | new active FRS-TRAIN-v015 | Inspector, registry, formal route | v014 remains history |
| optimizer group construction | `FrontRESUnified` | grouped formal update | anonymous one-group Stage-3 optimizer |
| actor/critic LR input | active FrontRES runner config, composed by `train.py` | `FrontRESUnified` constructor | generic Stage-3 shared-LR override |
| fixed schedule | Stage-3 config composition | live policy update | adaptive group-wide LR overwrite |
| persistence identity | FrontRES interface/checkpoint owner | save/resume/eval | checkpoint-v9 full resume |
| emitted LR facts | committed transaction telemetry owner | logs/evidence | one ambiguous scalar LR |

No wrapper, service, registry, manager or parallel configuration system is
admitted. Existing owners are modified in place.

## Engineering Boundary Record

Change reason and owner:

- `FrontRESUnified` owns the complete optimizer parameter partition and creates
  the only Adam instance.
- The stable public input is two positive finite floats: Actor LR and Critic LR.
  The stable output is one optimizer with two named, disjoint, exhaustive groups.
- Callers do not inspect policy internals to construct groups.

Dependency direction:

```text
explicit Stage-3 CLI/config
-> FrontRESUnified optimizer composition
-> grouped PPO gradient/update owner
-> checkpoint Gateway and read-only telemetry
```

Forbidden dependencies:

- checkpointing and telemetry cannot repartition parameters;
- runner update code cannot infer group identity from list position alone;
- config cannot reach policy private fields;
- adaptive scheduling cannot mutate the fixed two-group identity;
- no fallback may accept an unnamed, missing, overlapping or mixed group.

State and lifecycle:

```text
HSL-v2 actor load
-> fresh Critic
-> fresh named two-group Adam
-> critic-only/ramp/joint exact-one commits
-> atomic checkpoint-v10 save
-> strict v10 reload with identical groups/LRs/moments/step count
```

Checkpoint-v9 is valid evidence for the previous campaign but is not resumable
under this optimizer identity. Rejection must occur before actor, Critic,
optimizer, curriculum, sampler or receipt mutation.

Legacy safety:

- Characterization Test: current Stage-3 optimizer has one group containing
  residual Actor plus scalar Critic, and adaptive LR writes all groups equally.
- Effect Sketch: config -> constructor -> gradients -> Adam step -> telemetry ->
  checkpoint save/load.
- Pinch Points: `FrontRESUnified` optimizer construction and strict checkpoint
  envelope validation.
- Seam: hand-built tiny `FrontRESActorCritic` plus strict checkpoint payload.
- Enabling Point: Stage-3 composition in `scripts/rsl_rl/train.py`.

Component placement:

- CCP keeps parameter partition with `FrontRESUnified`, persistence validation
  with checkpointing and output projection with telemetry.
- CRP exposes only the two LR inputs and immutable group facts.
- ADP/SDP keep CLI and filesystem details outside optimizer semantics.
- The existing checkpoint Gateway and transaction Unit of Work remain admitted;
  no new pattern is needed.

Hotspot delta: `frontres_unified.py` gains one coherent optimizer-composition
reason; checkpointing gains only the corresponding strict identity validation.
No runner or loss owner gains optimizer partition authority.

## File Responsibility Map

Governance and current-state projection:

- create `contracts/active/training/FRS-TRAIN-v015-fixed-split-lr.md`: accepted
  fixed split-LR and fresh Stage-3 identity;
- move v014 to `contracts/history/training/` and update the contract registry;
- update `frontres_interfaces.py`: new checkpoint/training identity;
- refresh Design Inspector, Design Register and Architecture references from
  proposed to active without changing the parent Concept Figure block.

Production/configuration:

- modify `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`: validate two LRs,
  partition parameters once and construct one two-group `FrontRESTrackedAdam`;
- modify the active `whole_body_tracking/utils/rsl_rl_cfg.py` and G1 FrontRES
  runner config: declare the Critic LR and fixed Stage-3 defaults;
- modify `scripts/rsl_rl/train.py` and `run_stage3.sh`: route explicit Actor and
  Critic LR values and reject shared/adaptive Stage-3 overrides;
- modify `frontres_checkpointing.py`: pre-mutation v10 validation, exact
  save/reload and no v9 optimizer migration;
- modify committed transaction diagnostics/telemetry only as needed to carry
  the two authoritative LR facts to the final serialized summary.

The unused duplicate `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py` is not made a
second authority. It is left unchanged unless an active production consumer is
found during execution; such a consumer would be a plan blocker, not permission
for silent duplication.

## Runtime-Probing Step Contract

Scope: trace Actor LR and Critic LR from composition through optimizer update,
telemetry and save/reload.

Victim / trigger: an anonymous shared-LR optimizer group observed at Stage-3
construction, update or resume.

Owner: `FrontRESUnified` optimizer composition.

Owner invariant:

1. exactly two named groups, `actor` and `critic`;
2. group parameters are disjoint, exhaustive and role-correct;
3. Actor LR is exactly `3e-6`, Critic LR exactly `1e-5`, schedule fixed;
4. critic-only preserves Actor parameters and Actor Adam state while Critic
   parameters change;
5. both groups report the same exact-one persisted optimizer step count;
6. v10 reload reproduces group names, membership, LRs, moments and count;
7. v9 or malformed group identity rejects before mutation.

Mutation surface: CLI/config assignment, optimizer construction, gradient
installation, optimizer step, rollback, checkpoint save, checkpoint load and
resume-LR reset. Telemetry is read-only.

Annotation scope:

- `FrontRESUnified.__init__` and its optimizer partition helper: B1 validate
  role inputs, B2 partition, B3 construct named groups;
- checkpoint optimizer validation/restore owner: B1 validate envelope/groups,
  B2 restore atomically, B3 publish restored identity;
- Stage-3 LR composition helper: B1 validate explicit fixed values, B2 install
  config, B3 emit composed identity.

Core parameter path:

```text
CLI/config values
-> algorithm constructor arguments
-> named optimizer param_groups
-> gradient role and optimizer.step
-> committed telemetry
-> checkpoint serializer
-> fresh reload
```

Test classes:

- core param path: semantic tiny policy with hand-identified Actor/Critic
  parameters, critic-only and joint updates, state and delta assertions;
- secondary contract path: CLI routing, fixed schedule, serialized telemetry and
  negative identity checks;
- bounded live sentinel: one official critic-only transaction after all offline
  contracts pass; log only group names/LRs, update count and role deltas.

Lifecycle matrix:

| Case | Required result |
| --- | --- |
| fresh construction | exact named 3e-6/1e-5 groups |
| critic-only | Actor params/state unchanged; Critic changes |
| actor-ramp/joint | both allowed to change under existing weights |
| exact-one commit | both groups share step count delta 1 |
| rollback/rejected update | params, moments and group identity restored |
| v10 save/reload | exact group identity and state roundtrip |
| v9/missing/duplicate/overlap/non-finite LR | fail closed before mutation |

Stop condition: every deterministic contract passes, final serialized facts are
exact, the bounded official transaction crosses the new identity once, and the
final review has no P0/P1. Otherwise stop before long training and report the
first invalid owner.

Closure claim: `owner-lifecycle-contract-confirmed` after offline tests;
`live-confirmed` only after the bounded official transaction.

## Human-Confirmable Module Test Cards

These four cards are the complete new testing questions. Approval of this plan
confirms them together; execution refreshes the Module Test Atlas before test
implementation and does not reopen them as separate approvals.

| Card | Rule | Artificial case | Independent expected result |
| --- | --- | --- | --- |
| Split-LR optimizer partition | `FrontRESUnified` alone partitions the one optimizer | toy policy with two explicitly identifiable Actor tensors and two Critic tensors, configured `3e-6/1e-5` | exactly two named, disjoint, exhaustive role-correct groups with the configured LRs; fixed std and frozen GMT absent |
| Phase-specific optimizer commit | the existing phase owner controls which role may commit | hand-computable nonzero Actor and Critic losses under critic-only, ramp weight 0.5 and joint weight 1 | critic-only changes only Critic and preserves Actor Adam state; ramp scales Actor gradient by 0.5; joint permits both; every case commits exactly one shared step |
| Checkpoint-v10 identity | persistence preserves the complete optimizer identity or rejects before mutation | save/reload a two-group optimizer, then attempt v9, missing-name, duplicate-role, overlap and non-finite-LR payloads | valid v10 reproduces names, membership, LRs, moments and count; every invalid payload rejects before model/optimizer/curriculum mutation |
| Stage-3 composition and final telemetry | callers provide explicit fixed values and the final serializer exposes owner facts | compose `actor_lr=3e-6`, `critic_lr=1e-5`, `schedule=fixed`; then try shared-only and adaptive inputs | valid composition reaches the optimizer and serialized receipt unchanged; shared/adaptive inputs fail closed; telemetry cannot mutate training state |

These cards prove optimizer/config/persistence behavior only. They do not use
policy-quality outcomes as their oracle.

## Step Map

### Step 1 / 1: Fixed Split-LR Engineering Closure

Objective: activate, implement and verify the accepted fixed split-LR identity
through the existing FEMR Stage-3 route.

Scope: TRAIN/checkpoint versioning, one-optimizer parameter grouping, explicit
Stage-3 composition, fixed-schedule isolation, committed telemetry, strict
save/reload, focused formal connectivity and one bounded official critic-only
transaction when server authority exists.

Non-scope: MOSAIC, model/loss/Gain/PPO/K/M/DR changes, old Stage-3 optimizer
migration, long training, policy quality and deployment.

Owner files/modules: active Training contract and registry;
`FrontRESUnified`; active FrontRES runner config and `train.py` composition;
grouped PPO consumer without loss changes; checkpoint Gateway; committed
telemetry serializer; focused existing test modules and current Atlas views.

Public input/output: positive finite `actor_lr` and `critic_lr` plus fixed
schedule -> one named two-group Adam identity carried unchanged through update,
telemetry and checkpoint reload.

Dependency direction and forbidden dependencies: config -> algorithm owner ->
existing update consumer -> persistence/telemetry. No policy-private traversal
outside the algorithm owner, no repartition in runner/checkpoint, no adaptive
writer, no fallback and no second optimizer.

State/transaction boundary: fresh HSL-v2 Actor plus fresh Critic/optimizer;
exact-one committed transaction; atomic v10 persistence; v9/malformed rejection
before mutation; retry starts from the unchanged last committed v10 state.

Legacy seam/effect boundary: characterize the current anonymous one-group
optimizer; use constructor and checkpoint-envelope Pinch Points; reject the old
shared/adaptive entry and checkpoint-v9 resume path.

Failure and cleanup behavior: any invalid group identity, LR, schedule,
checkpoint or role delta aborts without model, optimizer, curriculum, sampler,
receipt or filesystem commit. A rejected live update restores parameters and
optimizer state before retry.

Expected evidence: four confirmed Module Test Cards; compile/JSON/static checks;
owner, gradient, CLI, negative and persistence contracts; affected Stage-3
suite; focused formal reachability; final code review; bounded official
critic-only telemetry when authorized.

Stop condition: all offline evidence and reviews pass with no P0/P1, then one
official bounded transaction emits exact group/LR/delta/count/checkpoint facts.
If server authority is unavailable, close only as offline-ready and package one
self-contained sentinel command; do not start long training.

Split rationale: there is one user-visible engineering step because governance,
implementation, tests, review and bounded integration share one accepted
behavior and rollback boundary. Long training remains separate because it is a
materially costly policy-quality experiment.

## One-Shot Execution Unit

Main execution unit:

```text
activate/version TRAIN-v015 and checkpoint-v10
-> implement the existing-owner split-LR route
-> run focused owner, gradient, CLI, negative and persistence contracts
-> run the complete affected offline Stage-3 suite
-> refresh Inspector/Architecture/checklist/evidence
-> run construction and final code-discipline reviews
-> resolve in-scope P0/P1 and re-run affected checks
-> run one bounded official critic-only transaction when server authority exists
-> inspect the emitted group/LR/delta/step/checkpoint facts once
```

Embedded checks are acceptance evidence, not separate approval steps:

- Python compilation and JSON parsing;
- optimizer partition and gradient-role contract;
- critic-only Actor parameter plus Adam-state freeze;
- exact-one shared step count across both groups;
- CLI/config fixed identity and shared/adaptive negative cases;
- checkpoint-v10 save/reload plus v9 pre-mutation rejection;
- final telemetry serialization of both LR values;
- affected Stage-3 pseudo suite and `git diff --check`;
- `formal-runtime-audit` focused on config -> official update -> checkpoint
  reachability;
- `code-review-expert` construction and final gates.

## Engineering Acceptance

The engineering unit is complete only when:

1. the active contract, Inspector, config, runtime and checkpoint all name the
   same fixed split-LR identity;
2. no active Stage-3 shared/adaptive override remains;
3. the formal grouped update uses the existing single optimizer exactly once;
4. critic-only preserves Actor parameters and optimizer state;
5. checkpoint-v10 round-trips and checkpoint-v9 rejects pre-mutation;
6. final telemetry exposes exact Actor/Critic LR and role deltas;
7. focused formal reachability is closed and final review has no P0/P1;
8. no unrelated user worktree changes are modified.

Policy quality, convergence and long-horizon physical efficacy remain
unconfirmed after this engineering closure.

## Conditional Escalation And True Stop Boundary

Correct in-scope implementation/test/review failures inside the same authorized
unit. Stop and return to the user only if:

- implementation requires MOSAIC host changes or a second optimizer;
- an active consumer requires the duplicate config owner;
- exact v9 pre-mutation rejection cannot be achieved without destructive or
  compatibility migration behavior;
- a P0/P1 finding changes accepted semantics or materially expands scope;
- the server, checkpoint, HSL artifact or GPU authority needed for the bounded
  official transaction is unavailable.

Long Stage-3 training is the next material-cost boundary. It starts only after
this unit closes, from HSL-v2 at K8/M2 critic-only iteration 0, in a new clearly
named run directory. Existing Stage-3 checkpoints remain comparison evidence,
not resume sources.
