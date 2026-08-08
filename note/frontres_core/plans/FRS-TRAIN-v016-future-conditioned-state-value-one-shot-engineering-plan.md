# FRS-TRAIN-v016 Future-Conditioned State-Value One-Shot Engineering Plan

Status: PHASE B LIVE-CONFIRMED; long-training decision pending
Date: 2026-08-08

## Terminal Outcome

The active Stage-3 route implements `FRS-METHOD-v018`, `FRS-TRAIN-v016` and
`FRS-PPO-v006` without changing the FrontRES Actor, Gain, GMT, simulator or
curriculum:

```text
Actor input   = 158D deployable FrontRES prefix
Critic input  = 289D current privileged state + 58D sealed Noisy q29 future = 347D
GMT input     = unchanged 770D suffix
Actor target  = each attempt's own PPO advantage
Critic target = exact-M mean G_total for the sealed Segment state
commit        = separate Actor/Critic clip(0.5), exactly one two-group Adam step
checkpoint    = frontres-v017-checkpoint-v11
```

The terminal engineering proof is one bounded official critic-only transaction
that exposes the exact observation identities, shared Segment value/target,
separate finite gradient norms, frozen Actor state, changed Critic, exact-one
step and atomic v11 reload. That proof permits the user to decide whether to
start a fresh long campaign; it is not policy-quality evidence.

## Accepted Behavior And Non-Scope

Accepted behavior:

- append the already sealed `(t+1,t+2)` deployment-Noisy q29 object to the
  current 289D Critic observation exactly once;
- reuse one identical 347D state row and one old scalar value across the exact-M
  attempts belonging to a Segment;
- keep each attempt's realized `G_total` and Actor advantage distinct while
  assigning the exact-M return mean as the Critic target;
- install and clip Actor/std gradients and Critic gradients independently;
- perform one step of the existing named split-LR Adam and preserve the current
  critic-only rollback behavior;
- save and strictly reload the complete v11 identity.

Preserved behavior and non-scope:

- no 6D action, Clean continuation, evaluator evidence, K, corruption label or
  perturbation timing enters the Critic;
- no Actor/Critic network topology change beyond the Critic input width;
- no Gain, per-attempt return, PPO ratio, grouped Actor reduction, LR, warmup
  iteration, K/M/DR, HSL, GMT, MOSAIC host or simulator semantic change;
- no second optimizer, scheduler, Critic, value head or silent legacy fallback;
- no checkpoint-v10 Stage-3 state migration; historical v10 artifacts remain
  available only through the strict read-only quality-inspection boundary; no
  long training before the formal runtime evidence gate.

## Engineering Boundary Record

### Requested and preserved behavior

The request repairs state aliasing, value-target ambiguity and cross-family
gradient clipping. It does not turn the Critic into `Q(s,a)`. The shared
state-value subtraction must preserve every within-Segment ordering induced by
the unchanged per-attempt `G_total` values.

### Semantic owners and public interfaces

| Boundary | Single owner | Public input | Public output |
| --- | --- | --- | --- |
| future-conditioned Critic state | active FrontRES observation gateway | current privileged `[B,289]`; sealed Noisy q29 tail `[B,58]` | normalized Critic state `[B,347]` |
| state-value target | Segment PPO algorithm | sealed rows, Segment identity, per-attempt returns | row-aligned exact-M Segment mean targets and per-attempt advantages |
| gradient boundary | Segment PPO algorithm | Actor loss, Critic loss, disjoint parameter identities, max norm | installed separately clipped gradients plus immutable clip facts |
| transaction commit | existing formal transaction Unit of Work | verified gradients, one named split-LR Adam | one committed receipt and role-specific deltas |
| persistence | FrontRES checkpoint gateway | active contract/layout/optimizer/receipt state | atomic checkpoint-v11 or pre-mutation rejection |
| projection | existing diagnostics owner | already-computed target/value/gradient/identity facts | read-only telemetry |

The runner remains thin orchestration. It may compose dimensions and call these
owners, but it may not recompute a target, partition parameters, or infer a
checkpoint identity.

### Dependency, state and transaction direction

```text
sealed Noisy future Intent + current env Critic observation
-> observation gateway
-> sealed candidate rows
-> Segment PPO state-value target and disjoint gradients
-> existing one-step transaction Unit of Work
-> diagnostics projection
-> atomic checkpoint-v11 Gateway
```

The future tail is detached and scenario-sealed before exact-M rollout. It is
not resampled per attempt. Target construction and gradient installation occur
inside one uncommitted transaction. Only the existing optimizer step and
checkpoint receipt cross the commit boundary. Any missing, malformed,
non-finite, mixed-Segment or legacy identity fails before mutable restoration or
commit.

### Forbidden dependencies

- observation code cannot inspect Repair action, Gain or evaluator results;
- PPO cannot read simulator or command private attributes;
- diagnostics cannot recompute Gain, target, clipping or identity;
- checkpointing cannot synthesize dimensions or repartition optimizer groups;
- callers cannot access another layer's private state or pass a new stable
  cross-layer `dict` contract;
- checkpoint-v10 and older payloads cannot partially initialize Stage-3 Critic,
  optimizer, normalizers, sampler, curriculum or receipt state.

## Change-Discipline Analysis

Characterization Test: the current active path provides a 289D privileged
Critic row to every attempt, uses each attempt return as that row's value target,
and clips one concatenated Actor/Critic gradient vector. Existing v015 tests
remain historical characterization and are not rewritten as v016 proof.

Effect Sketch:

```text
config identity
-> model Critic width / privileged normalizer width
-> live observation concatenation
-> storage row
-> exact-M target construction
-> value loss
-> disjoint gradient installation and clipping
-> exact-one step
-> telemetry
-> v11 save/load
```

Pinch Points:

- FrontRES future-intent observation gateway for all Actor/Critic context;
- Segment PPO loss/gradient owner for target and clipping semantics;
- FrontRES checkpoint envelope validation before restoration.

Seams and Enabling Points:

- tiny tensor fixtures can call observation composition without a simulator;
- a toy policy and sealed two-Segment x M batch can exercise the algorithm;
- a fresh runner/checkpoint fixture can prove pre-mutation rejection and exact
  round-trip;
- the bounded official entrypoint remains the only runtime Enabling Point.

CCP keeps each fact with its owner. CRP exposes only validated tensors and
immutable facts. ADP keeps filesystem/checkpoint mechanics out of PPO. SDP keeps
environment reads out of algorithm code. The existing Unit of Work and Gateway
are sufficient; no wrapper, Protocol, service or class hierarchy is added.

## File Responsibility Map

Production changes are limited to the active path and may be narrowed further
if the tests locate an existing owner:

- `frontres_rollout_step.py` / `frontres_segment_one_action_k.py`: reuse one
  validated future-tail composition path for Actor and Critic, with explicit
  928/158/347/770 identity and fail-closed shape/provenance checks;
- `on_policy_runner.py`: compose the active 347D Critic width and matching
  normalizer width at the existing model-construction boundary;
- `frontres_segment_ppo.py`: construct exact-M Segment mean value targets,
  preserve per-attempt Actor advantages, install and independently clip the two
  parameter families;
- `frontres_segment_formal_transaction.py`: orchestrate the v006 algorithm owner,
  project separate gradient facts and retain exactly-one commit;
- `frontres_checkpointing.py` and the active checkpoint-quality boundary: add
  strict checkpoint-v11 identity and pre-mutation v10 rejection while preserving
  unrelated current user edits;
- active FrontRES configuration: select v018/v016/v006/v11 explicitly; no shared
  or implicit default;
- diagnostics: serialize existing v016 facts only, without recomputation.

Focused v016 tests are added rather than silently relabeling the existing v015
evidence. Any necessary edit to a currently dirty file is applied against its
current contents and must not discard unrelated user work.

## Module Test Cards And Oracles

Execution requires human confirmation of the changed Atlas cards:

- `TEST-05 Observation Layout`: exact shared tail, 347D Critic, 158D Actor and
  770D GMT, permutation and malformed-input rejection;
- `TEST-15 Segment PPO`: exact-M mean target, shared old value, preserved Actor
  ordering, independent clipping and exact-one step;
- `TEST-16 Checkpointing`: complete v11 round-trip, HSL-v2 Actor-only cold start,
  and v10/malformed pre-mutation rejection;
- `TEST-18 Runtime Diagnostics`: faithful projection of dimensions, shared value,
  Segment target, separate gradient facts and v11 receipt.

Independent oracles are hand-computable tensors, parameter/state snapshots and
atomic payload sentinels. Current implementation output is never used to define
the expected result.

## Runtime-Probing Step Contract

Scope: trace the new Critic state and value target from sealed future Intent to
the committed v11 checkpoint, including both gradient families.

Victim/trigger: a 289D or action-conditioned Critic row, per-attempt Critic
target for one identical Segment state, shared clip coefficient, or v10
checkpoint admitted on the active route.

Owner invariants:

1. Actor/Critic/GMT dimensions are exactly `158/347/770`;
2. exact-M rows of one Segment have identical Critic state and old value;
3. Critic target equals the finite exact-M return mean;
4. each Actor advantage remains its own return minus the shared old value;
5. changing only one loss family cannot alter the other's clip coefficient;
6. critic-only preserves Actor parameters and Adam state while Critic changes;
7. one transaction increments the shared optimizer step exactly once;
8. v11 round-trips, while v10 or malformed identity rejects pre-mutation.

Annotation scope for changed important functions:

- observation composition: `B1` validate current/tail provenance and shapes,
  `B2` concatenate once, `B3` publish immutable layout facts;
- value-target construction: `B1` validate sealed Segment/exact-M groups, `B2`
  compute mean targets, `B3` align targets back to policy rows;
- gradient installation: `B1` partition by explicit Critic module identity,
  `B2` install and independently clip, `B3` publish finite norms/coefficients;
- checkpoint validation: `B1` validate complete v11 envelope, `B2` restore into
 temporary state, `B3` publish only after atomic success.

## Phase B Probe Plan

Phase A Method-Code Alignment was human-confirmed on 2026-08-08. The current
checklist is `18 passed / 0 partial / 0 blocked`; production implementation,
offline telemetry and checkpoint-v11 round-trip are complete. The user
confirmed this probe plan on 2026-08-08. Probe status is now
`runtime-confirmed`: commit `b74efd7` completed one bounded official v016
transaction and emitted every `AUDIT-B01..B08` marker exactly once without a
traceback.

The Phase B change is audit-only. It updates the existing formal-audit
projection and adds one read-only post-save verification at the checkpoint
Gateway. It does not add a probe to rollout, Gain or PPO because their final
immutable transaction telemetry already owns all required facts.

| Probe | Formal owner and exact location | Runtime fact and expected value | Design points | Failure / stop condition |
| --- | --- | --- | --- | --- |
| `AUDIT-B01` | `frontres_formal_runtime_audit.py::print_formal_route_audit()`; called by `frontres_segment_live_training.py` before collection | `METHOD-v018 / GAIN-v007 / PPO-v006 / TRAIN-v016`, HSL-v2/TRAIN-v014 cold start, K8/M2, fixed split LR | DP01, DP03, DP08, DP10 | any legacy identity, resume, alternate route or implicit schedule |
| `AUDIT-B02` | existing sealed transaction projection in `print_phase_b_telemetry_audit()` | two transaction sources, exact M=2, four policy attempts, eight Repair/Noisy role rows, stable source/scenario/hash identity | DP01-DP03 | mixed source, resample, missing attempt or duplicate identity |
| `AUDIT-B03` | `print_v017_repair_attempts_audit()` after observation construction in `frontres_segment_one_action_k.py` | Actor/Critic/GMT `158/347/770`; Critic is state-value and action-conditioned=false; the Critic tail is the same detached finite 58D sealed Noisy q29 tail | DP04, DP07, DP10 | 289D Critic, different tail, attached/non-finite tensor or forbidden provenance |
| `AUDIT-B04` | existing one-action-K projection after attempt collection | finite direct `[4,6]` actions, one Actor sample per attempt, K=8, frozen GMT | DP03-DP05 | repeat Actor call, squashing/mask, wrong K or trainable GMT |
| `AUDIT-B05` | existing Gain-v007 projection from committed immutable reports | Clean=2, Noisy=2, Repair=4 and finite `G_I/G_P/P_N/P_R/lambda_RA/cost/G_total` | DP06-DP07 | missing/malformed evidence, non-finite Gain or diagnostics recomputation |
| `AUDIT-B06` | `print_phase_b_telemetry_audit()` over `build_frontres_transaction_telemetry()` | each source's Critic target equals the exact-M arithmetic mean; target repeats per source while each Actor advantage remains its own `G_total - shared V(s)` | DP02, DP07, DP09 | per-attempt Critic target, cross-source grouping, target/advantage mismatch or ordering change |
| `AUDIT-B07` | same committed telemetry projection after `run_frontres_formal_transaction_update()` | separate Actor/Critic pre/post norms, coefficients and nonzero counts; each post-norm <=0.5; critic-only Actor/std delta=0, Critic delta>0; one optimizer step; LR `3e-6/1e-5` | DP02, DP07, DP09 | shared clip fact, non-finite norm, wrong delta, step delta !=1 or LR drift |
| `AUDIT-B08` | `frontres_checkpointing.py::save_runner()` after `os.replace`, using the existing strict loader/validator and a read-only audit emitter | saved and reloaded `frontres-v017-checkpoint-v11`, 158/347/770 layout, complete receipt/optimizer/normalizer/curriculum/GMT identity; no runner mutation | DP01, DP02, DP05, DP09, DP10 | missing field, checksum/readback mismatch, v10 admission, partial receipt or non-atomic save |

Skipped in this bounded audit: actor-ramp, joint-phase optimization, resume
training, K16/K32 transitions, calibration quality and policy quality. Those
require more optimizer transactions and do not help prove the new v016 route.

Exact tiny official command after probe-plan confirmation, offline regression,
review, Git synchronization and server identity verification:

```bash
cd /hdd0/yuxuancheng/FEMR

CUDA_VISIBLE_DEVICES=0 \
FEMR_ROOT=/hdd0/yuxuancheng/FEMR \
FEMR_DATA_ROOT=/hdd0/yuxuancheng \
CACHE_DIR=/hdd0/yuxuancheng/AMASS_G1Segment \
LOG_PATH=/hdd0/yuxuancheng/FEMR/log/FRS_TRAIN_V016_STATE_VALUE_PHASE_B_SENTINEL.log \
FRONTRES_G5_S4_BOUNDED=1 \
FRONTRES_CHECKPOINT_INTERVAL=1 \
FRONTRES_SEGMENT_PPO_SCHEDULE=fixed \
FRONTRES_SEGMENT_ACTOR_LR=3e-6 \
FRONTRES_SEGMENT_CRITIC_LR=1e-5 \
FRONTRES_V015_K_CURRICULUM='8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381' \
bash run_stage3.sh \
  /hdd0/yuxuancheng/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt \
  /hdd0/yuxuancheng/AMASS_G1NPZ_Final \
  8 1 1 train
```

Bounded cost: GPU0, eight environments, one K8/M2 transaction, four policy
attempts, one optimizer step and one checkpoint-v11 save/reload. Wall-clock
duration is not claimed in advance because it depends on server startup and
cache state. Any `AUDIT-B01..B08`, finite, identity, exact-one or atomic-readback
failure stops the run; it never falls through to long training.

### Phase B Observed Evidence

- Server source identity: `b74efd7`.
- Raw log: `/hdd0/yuxuancheng/FEMR/log/FRS_TRAIN_V016_STATE_VALUE_PHASE_B_SENTINEL.log`.
- Every `AUDIT-B01..B08` marker occurs exactly once; `Traceback` occurs zero
  times and the final save status is `OK`.
- B03 observed Actor/Critic/GMT `158/347/770` and exact shared source-state
  rows; B04 observed one finite `[4,6]` action sample and K=8 frozen-GMT
  execution.
- B07 observed one optimizer step, split LR `3e-6/1e-5`, zero Actor/std
  parameter delta in `critic_only`, and nonzero Critic parameter delta.
- B08 atomically reloaded
  `/hdd0/yuxuancheng/FEMR/g1_flat_frontres_stage3_segment_hrl/2026-08-08_08-29-51_G5_S4_BOUND_V015/model_1.pt`
  as `frontres-v017-checkpoint-v11` with `runner_mutated=0`.
- Independent artifact inspection observed iteration 1, Critic normalizer shape
  `[1,347]`, 16 legitimate zero-variance dimensions, and
  `std.square() == var` within the strict validator tolerance.

This closes Phase B engineering/runtime connectivity only. It does not claim
Critic calibration, policy quality, or authorize an automatic long run.

## One-Shot Execution Step

### Step 1/1: implement, prove, audit and prepare restart

Precondition: the four changed Module Test Cards are human-confirmed.

Actions:

1. add the failing v016 module tests from the confirmed cards;
2. implement the smallest owner-local changes above;
3. run targeted tests, `python -m py_compile`, contract/Atlas checkers and the
   relevant existing regression suite;
4. perform Engineering Plan Review, construction review, removal review,
   security review and research-ML review; repair all in-scope P0/P1 findings;
5. run Formal Runtime Audit Phase A and present the design-point matrix;
6. after the required Phase-B probe-plan confirmation, synchronize the reviewed
   files, run one bounded official critic-only transaction and classify facts;
7. after the user's final evidence/cost decision, start one fresh official
   TRAIN-v016 campaign from HSL-v2 Actor-only initialization. Never resume the
   bounded sentinel or a checkpoint-v10 Stage-3 state.

Stop conditions:

- any contract, owner, information-boundary or legacy-path contradiction;
- any failed confirmed module oracle or regression;
- any non-finite tensor, mixed Segment, wrong exact-M, dimension drift, shared
  clip coupling, optimizer step delta other than one, Actor mutation in
  critic-only, partial receipt or non-atomic checkpoint;
- unavailable server identity/permissions or a required runtime fact that
  cannot be observed safely;
- no long training until the user accepts the completed Phase-B evidence and
  its cost.

## Engineering Plan Review

Verdict: PHASE B LIVE-APPROVED; LONG-TRAINING DECISION PENDING.

- Scope and non-scope exactly match the confirmed Design Inspector and active
  contracts; no MOSAIC host or method expansion is admitted.
- Responsibilities remain with existing semantic owners; the runner stays an
  orchestrator and no abstraction is added without removing a named dependency.
- Public inputs/outputs, dependency direction, transaction/persistence
  boundaries, forbidden dependencies and fail-closed behavior are explicit.
- The four cards isolate the changed observation, optimization, persistence and
  projection responsibilities with independent oracles.
- The proof route separates module correctness, formal connectivity and later
  policy quality. The four confirmed cards pass, Phase A and the Phase B plan
  are human-confirmed, and the bounded official transaction passed B01-B08.
  The next true gate is the human cost/evidence decision for a fresh long run.
