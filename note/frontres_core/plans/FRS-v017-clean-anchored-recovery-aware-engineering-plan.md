# FRS-v017 Clean-Anchored Recovery-Aware Engineering Plan

Status: DP10 Future Motion Context is integrated offline and Phase A reviewed
at E-FI-119. All ten Phase A design points are now reviewed offline. E-FI-121
adds and offline-tests the eight Phase B fail-closed runtime checks. E-FI-122
records the first official run: B01 passed, then the first Clean reset exposed
and offline-closed an install-before-mode lifecycle defect. E-FI-123 records
that the rerun crossed reset/observation/action/K execution, then exposed and
offline-closed mixed `[K,4]`/`[K,8]` Repair trajectory rows; B02-B08 require the
same bounded live rerun. No
Gain/PPO semantics or live boundary was reopened
under `FRS-METHOD-v017 / FRS-GAIN-v007 / FRS-PPO-v005 / FRS-TRAIN-v014 /
FRS-EVAL-v004`. E-FI-105 proves the previous v012 module surface;
E-FI-106 isolates legacy updater selection. Those facts remain valid only for
unchanged behavior. TRAIN-v013 curriculum, checkpoint-v8 and affected Test
Cards are now covered by E-FI-109 S1/S2/S3 evidence. Simulator, training and
live remain closed.

The current closure removes the formal HSL/Stage-3 `tanh` plus per-axis action
scale mismatch found by the DP04 Phase A review. It versions HSL as
`frontres-v017-hsl-proposal-v2` and Stage-3 persistence as
`frontres-v017-checkpoint-v9`; HSL-v1 and checkpoint-v8 reject before mutation.

## Prior TRAIN-v013 Curriculum Outcome

One offline engineering closure replaces the active v012/g_K/checkpoint-v7
curriculum identity with TRAIN-v013:

```text
explicit per-K DRStageSpec
-> lower-to-higher four-class DR progression
-> committed K transition resets DR difficulty
-> same-Critic recalibration, actor ramp and joint optimization
-> exact-M sealed transaction and grouped exact-one update
-> strict checkpoint-v8 save/resume identity
-> official offline route and read-only telemetry
```

DP04 retains that curriculum and upgrades the active action/persistence edge to
HSL-v2 and checkpoint-v9. The active route preserves METHOD-v017, GAIN-v007,
PPO-v005, EVAL-v004,
928/158/770 authority, one-action-K, exact two-Segment x M transaction, scalar
Critic, full-6D action and all existing Physics/Intent formulas. It performs no
simulator, training, live run, policy-quality experiment or deployment
composition.

## Engineering Boundary Record

### Accepted Behavior

- K/M remains `K8/M2 -> K16/M3 -> K32/M4`; K64 stays inactive.
- Every K owns one explicit immutable DR stage specification and one
  lower-to-higher inner DR curriculum.
- At a committed K transition, DR returns to the new stage's lower informative
  distribution; Actor/std freeze; the same Critic recalibrates; ramp and joint
  then resume.
- Four strength classes remain present at 20/30/40/10 relative to the current
  explicit stage-local `d_cap`; 2.381 is the measured maximum reliable
  frozen-GMT perturbation boundary for the current setup. The current campaign
  configures it directly; optional offline re-probing is only needed after the
  GMT, robot or perturbation definition changes.
- One sealed Segment samples one class/strength/artifact and reuses it for its
  Noisy baseline and all M Repair attempts; Clean remains uncorrupted.
- Cross-horizon ordering is diagnostic-only and cannot feed sampler, PPO, Gain
  or schedule state.
- Only a committed transaction may advance phase, K/M, inner-DR progress,
  iteration, receipt or checkpoint state.

### Preserved Behavior And Non-Scope

- no method/Gain/PPO/HSL/evaluation formula change;
- no actor/GMT observation or action-frame change; only the contradictory
  bounded action coordinate is replaced by the already-contracted direct 6D
  coordinate;
- no rho, second actor/Critic/optimizer, online adaptive controller, winner-only
  replay, composite perturbation or Gain/PPO-to-sampler feedback;
- no mandatory per-K `g_K`, online episode-length controller, hidden DR default
  or old checkpoint compatibility mutation; an optional offline frozen-GMT
  boundary probe may only produce the frozen `reference_ceiling` consumed by
  the same schedule;
- no new runner, evaluator, Service Layer, registry, wrapper or MOSAIC host
  change;
- no source edit outside the current offline engineering closure; no live
  action is included in this document-only activation.

### Single Owners And Public Boundaries

| Semantic fact | Existing owner retained | Public input/output |
| --- | --- | --- |
| TRAIN/checkpoint/DR schema IDs | `frontres/frontres_interfaces.py` | versioned constants and validated immutable identity |
| K/M phase plus inner-DR progress | `frontres/frontres_segment_warmup.py` | frozen stage spec -> resolved committed stage identity |
| CLI/config composition | `scripts/rsl_rl/train.py::main()` and existing launcher | explicit schedule/DR args -> validated active config |
| sealed transaction identity | `runners/frontres_segment_formal_transaction.py` plus existing transaction Aggregate | resolved stage identity -> one homogeneous transaction plan |
| corruption materialization | existing sampler/perturbation owner | sealed stage/class/strength -> one immutable local scenario |
| grouped scalar update | existing `frontres_segment_ppo.py` owner | complete `2*M` rows -> one optimizer commit |
| v9 persistence | `runners/frontres_checkpointing.py` | committed runner state plus direct-action identity -> strict v9 payload / atomic restore |
| telemetry | existing diagnostics and `frontres_segment_training_telemetry.py` | owner-produced immutable facts -> read-only serialized fields |
| outer composition | `scripts/rsl_rl/train.py::main()` | selects existing concrete owners; owns no curriculum math |

The current `FrontRESKStageSpec` / `FrontRESKStageIdentity` boundary is refined
in place to carry the minimum immutable DR facts. A new wrapper or parallel
schedule class is rejected unless implementation proves the existing boundary
cannot preserve one named invariant.

### Dependency Direction And Forbidden Dependencies

```text
explicit CLI/config values
-> frontres_segment_warmup deterministic schedule owner
-> immutable stage identity
-> transaction Aggregate and perturbation materializer
-> grouped PPO commit
-> checkpoint Gateway and read-only telemetry
```

The schedule owner cannot import runner/simulator state, Gain, PPO,
checkpointing or diagnostics. The sampler cannot infer progress from outcome
metrics. Diagnostics cannot mutate curriculum. Checkpointing validates and
restores owner state but cannot resolve a different schedule.

### State, Transaction, Failure And Resume Boundary

- stage spec, current `d_cap`, class/strength, K/M/phase and curriculum
  fingerprint are sealed before scenario materialization;
- reset and all attempts reuse that identity without resampling;
- failure, partial rows, non-finite evidence, identity drift or serializer
  failure aborts without advancing schedule/optimizer/sampler/receipt;
- one successful exact-one commit advances the inner DR cursor and any outer
  K/M transition atomically;
- v9 saves only adjacent to a committed receipt and persists complete schedule,
  cursor, sampler/RNG, optimizer and model/normalizer state;
- v8/v7, v012 `g_8/g_16/g_32`, hidden controller state, partial receipt or
  mismatched spec rejects before mutable restoration;
- retry opens a new transaction from the unchanged last committed curriculum
  state.

### Legacy Characterization, Effect Sketch And Pinch Points

Characterization retains current K/M phase arithmetic, exact-M widths,
critic-only Actor/std freeze, same-Critic transition, no-resample scenario,
grouped exact-one update and pre-mutation checkpoint rejection. It does not
preserve v012 `g_K` semantics or checkpoint-v7 acceptance.

```text
explicit DRStageSpec
 -> schedule owner resolves active d_cap/class support
 -> transaction seals K/M/DR identity
 -> sampler materializes one immutable artifact
 -> exact-one commit advances cursor
 -> checkpoint-v9 persists the same curriculum and direct-action identity
```

- schedule Pinch Point: `resolve_frontres_k_stage_identity()` and its in-place
  v013 successor behavior;
- transaction Pinch Point: existing formal transaction plan validation;
- persistence Pinch Point: existing `save_runner()` / `load_runner()` Gateway;
- Seam: hand-built stage specs and committed iteration counters;
- Enabling Point: `train.py::main()` selects explicit real configuration once.

The retired `frontres_dr_curriculum.py` episode-length/frontier controller is a
negative characterization target only. It cannot become an active fallback.

### Component And Pattern Admission

- CCP keeps K/M/DR phase resolution in the existing schedule owner.
- CRP exposes only the resolved immutable stage identity to transaction and
  sampler consumers.
- ADP/SDP keep runner, simulator, checkpoint and serializer dependencies outside
  deterministic curriculum policy.
- Existing transaction Aggregate and checkpoint Gateway remain admitted because
  they protect atomicity and external IO respectively.
- No new Service Layer, Repository, Protocol, manager, registry or wrapper is
  admitted; none removes a named dependency.
- `frontres_segment_warmup.py` and checkpointing are hotspot inspection points,
  not automatic split targets. The change replaces old identity rather than
  adding another change reason.

## Source-Of-Truth Migration Matrix

| Object | Active owner after migration | Legacy authority to reject/isolate | Required proof |
| --- | --- | --- | --- |
| training/checkpoint IDs | `frontres_interfaces.py` | TRAIN-v012/checkpoint-v7 literals | identity/import negatives |
| stage spec and progress | `frontres_segment_warmup.py` | frozen `g_K` and global frontier controller | hand-calculated boundary/progress tests |
| config/launcher | `train.py` and existing shell launcher | hidden/default DR schedule | explicit-argument and rejection tests |
| transaction identity | formal transaction + Aggregate | mixed K/M/DR rows or partial advance | seal/abort/commit/permutation tests |
| strength sampling | existing perturbation owner | outcome feedback/resample/composite route | four-class/no-feedback/no-resample tests |
| persistence | `frontres_checkpointing.py` | v8/v7/`g_K`/partial resume | v9 roundtrip and pre-mutation rejection |
| telemetry | diagnostics/telemetry owners | recomputation or feedback | required-field/zero-write tests |

## Step Map

### DP04 Direct-Action Closure Inside Step 1

Objective: make HSL, Stage 3, PPO storage/log-prob, checkpoint verification and
deployment consume the Actor's direct finite `[B,6]` world-frame residual.

Scope: existing Actor, HSL warmup, rollout action/log-prob, formal policy-update
consumer, checkpoint identity/verification, focused tests and current
Architecture/governance projections.

Non-scope: 158D inputs, GMT writer/composition, Gain/PPO objective mathematics,
K/M/DR curriculum, simulator, training, live evaluation and policy quality.

Owner and public boundary: `FrontRESActorCritic` owns `[B,158] -> [B,6]`; HSL
and Stage 3 consume that same coordinate without recomputation. Checkpointing
persists the coordinate identity but does not transform the action.

Dependency/state boundary: Actor -> rollout/HSL -> command/PPO; HSL-v2 and
checkpoint-v9 reject old identities before actor, normalizer, optimizer,
sampler or transaction mutation.

Legacy seam/effect boundary: characterize the existing squashed distribution,
then remove `_frontres_bounded_proposal`, inverse-`tanh` log-prob and HSL target
range projection from all active consumers. Retired quality/algorithm branches
may remain only when named legacy and unreachable from the active route.

Evidence: direct hand-computed Actor/HSL owner cases; permutation/nonfinite
negatives; Actor -> rollout/PPO -> command connectivity; HSL-v2/checkpoint-v9
roundtrip and v1/v8 pre-mutation rejection; focused regression and final code
review.

Stop: a required HSL target cannot be expressed in direct coordinates; an
active consumer still applies action bounding; old and new checkpoint identity
cannot be separated before mutation; or the change requires Gain/PPO/GMT or
MOSAIC-host semantics.

### Historical Step 1 / 2: TRAIN-v013 Offline Engineering Closure

**Objective:** implement the complete v013 curriculum and checkpoint-v8 route,
then prove module correctness, official offline connectivity and maintainable
ownership in one authorization unit.

**Scope:** existing interfaces, schedule, config/launcher, transaction,
perturbation, telemetry and checkpoint owners plus focused contracts and current
governance/Architecture refresh.

**Non-scope:** simulator, training, live, policy-quality, deployment, method,
Gain, PPO, HSL, actor/GMT observation and Concept Figure.

**Internal execution order:**

1. update the confirmed affected Module Test Cards without weakening their
   independent answers;
2. replace v012 identity in the existing schedule/config/transaction owners;
3. connect four-class immutable materialization, committed-only cursor advance
   and read-only telemetry;
4. implement strict checkpoint-v8 save/resume and v7/`g_K` pre-mutation reject;
5. run affected S1/S2/S3 cards, the unchanged 18-card regression, construction
   review, formal-runtime Phase A and final-gate review;
6. fix in-scope P0/P1, rerun affected proof, and refresh evidence/checklist/
   Architecture once.

**Expected evidence:** hand-calculated DR boundaries/weights; K transition
metamorphic cases; exact-M/mixed identity negatives; no-resample/no-feedback;
critic-only state delta; committed-only progression; v8 roundtrip/tamper/v7
reject; official config -> transaction -> update -> save/telemetry connectivity;
no active import of the retired controller.

**Stop condition:** any new semantic choice, hidden schedule default, duplicate
owner, actor/GMT or MOSAIC change, inability to reject partial/mixed state,
unresolved P0/P1, or formal bypass stops before live.

### DP07 Repair Gain Read-Only Projection Closure

**Accepted behavior:** `frontres_gain.py` remains the unique owner of all
FRS-GAIN-v007 values. The existing local report and formal telemetry must carry
the owner-produced normalized Intent/Physics channels, `I_N`, `I_R`, `P_N`,
`P_R`, `G_I`, `G_P`, `lambda_RA`, weighted Physics Gain, full-6D cost,
cost-free score, beta, penalty and `G_total` without recomputation.

**Preserved/non-scope:** no formula, PPO, Critic, optimizer, transaction,
checkpoint, HSL, observation, simulator or live change. No legacy v006 field,
projection/KKT value, zero-fill or runner-private reconstruction is admitted.

**Owner and public boundary:**

```text
FrontRESRecoveryAwareGainResult
-> build_frontres_v017_local_evaluation_report()
-> immutable FrontRESV017LocalEvaluationReport
-> build_frontres_transaction_telemetry()
-> row-aligned serialized fields
```

The report builder is the sole projection owner; telemetry is the final
serializer. Both are read-only consumers of the Gain result. Attempt,
scenario, noisy-hash and transaction identity remain aligned through the
existing report-to-PPO row permutation.

**Dependency direction:** reporting depends on the stable Gain result record;
Gain never imports reporting. The serializer consumes the immutable report and
does not read Gain helpers, evidence internals or runner private state.

**State/failure boundary:** no training state is mutated. Missing fields,
non-finite required values, malformed channel shape, mixed identity or an
incomplete row permutation fails before telemetry publication.

**Legacy seam/effect boundary:** the current report already forwards a subset
of v007 scalars without recomputation; this is the Characterization Test and
Pinch Point. The change extends that existing immutable record in place rather
than adding a wrapper, second report or second Gain owner. The legacy v006
report remains outside the v017 formal transaction.

**Proof route:** focused TEST-18 sentinel forwarding and negative cases;
TEST-13 formula regression; official fake Stage3 transaction through final
telemetry serializer; `py_compile` and affected regression. Phase B remains
closed because real Contact/ZMP values are live-only.

**Annotation scope:** refresh only the existing report/result docstrings or
`B1/B2/B3` comments needed to make owner-produced versus projected fields
explicit. Do not annotate legacy code or add probes.

**Hotspot delta:** the report and serializer gain fields but no semantic
responsibility or dependency. Caller knowledge remains one immutable report;
no new module, wrapper, Protocol or service is created.

**Stop condition:** stop if any required value must be recomputed, the public
Gain result lacks an active-contract value, row identity cannot be preserved,
legacy projection becomes reachable, or a focused test reveals a Gain formula
contradiction.

### DP08 HSL Direct Full-6D Target Offline Readiness Closure

**Accepted behavior:** Stage-1 HSL consumes the deployable 158D actor prefix and
learns the current-frame anti-DR target as the exact finite world-frame
`[B,6]` value. Position is `-anchor_dr_delta_pos`; orientation is the existing
anti-DR quaternion correction expressed as RPY. No axis is masked, scaled,
clipped or clamped. Upward `dz` remains available to the full-6D proposal and
is discouraged only by the learned initialization, Clean-anchored consequence
and full-6D repair cost already owned by the active contracts.

**Preserved/non-scope:** HSL remains actor-only and proposal-only. The 158D
input, HSL-v2 payload, Critic/optimizer/sampler exclusion, Stage-3 zero
supervised target, HSL/resume mutual exclusion, quaternion numerical-domain
protection, METHOD/GAIN/PPO, simulator and live behavior remain unchanged.

**Owner and public boundary:** the existing
`get_supervision_target_task_space()` is the unique Stage-1 anti-DR target
producer; `validate_frontres_hsl_current_frame_target()` checks the same public
`[B,6]` value immediately before the existing actor-only loss. Neither owner
may silently replace an axis value. The runner remains orchestration only.

**Dependency/state boundary:** command-owned current perturbation fields flow
to one detached finite target and then to the HSL actor-only optimizer. Clean
future, local Scenario, Stage-3 storage, Critic, transaction and checkpoint
resume state are forbidden dependencies. This change has no transaction or
persistence mutation.

**Legacy seam and proof route:** the former asymmetric `dz clamp(max=0)` in the
producer, validator and matching fixture is the characterized legacy behavior.
The producer/validator pair is the Pinch Point. TEST-10's confirmed direct
full-6D rule supplies the independent answer: a negative perturbation `dz`
must produce the corresponding positive anti-DR target, not zero. Run the HSL
S1/S2 target, actor-only, Stage-3 isolation and HSL-v2 checkpoint regressions,
then the affected direct-action/observation aggregate.

**Hotspot delta and stop condition:** remove one duplicated hard-axis policy
from two existing owners and add no module, wrapper, private dependency or
caller fact. Stop if exact anti-DR requires changing the action contract,
Clean/future input, Stage-3 supervision, checkpoint schema, simulator behavior,
or if any focused regression exposes a second active target authority.

### Step 2 / 2: Bounded Official TRAIN-v014 Sentinel

**Why separate:** this consumes simulator/GPU resources and creates a real
checkpoint; it is a material permission and evidence boundary.

After Step 1 closes and the user separately authorizes it, run one bounded
official K8/M2 transaction from strict HSL-v2, one exact optimizer update, one
committed checkpoint-v9, one fresh reload and one read-only local report. The
command must use explicit first-stage DR start/advance values. Stop before a
second update, long training, multi-seed, beta tuning or deployment.

## Acceptance Matrix

- S1 owner: stage arithmetic, four-class support, no-resample, fail-closed
  identity and critic-only isolation.
- S2 connectivity: official config -> resolved stage -> transaction -> sampler
  -> grouped exact-one commit -> telemetry with no fallback.
- S3 persistence: checkpoint-v9 atomic save/resume; v8/v7/`g_K`/partial/tampered
  payload pre-mutation rejection.
- S4 live: separate Step 2 only; real Contact/ZMP/quality is not inferred from
  offline evidence.

## Plan Review Status

Independent `code-review-expert::engineering_plan_review` is stored at
`../reviews/FRS-v017-train-v013-engineering-plan-review.md`. Source work is
executable only if that report says `READY`. For this closure the user explicitly
authorized implementation before the three affected card projections were
updated; their independent answers remained contract-derived and were not
weakened after observing implementation output.

## Cursor

TRAIN-v014 direct full-6D action semantics are implemented and offline-closed
at E-FI-114. HSL, Stage 3, PPO storage/log-prob and persistence now share one
finite direct `[B,6]` action; HSL-v1/checkpoint-v8 reject before mutation. The
prior TRAIN-v013/checkpoint-v8 curriculum closure remains recorded at E-FI-109:
the three affected cards and the 49-contract aggregate pass, prior offline
Formal Runtime Audit Phase A evidence exists, and construction/final review
found no P0/P1. E-FI-110 satisfies the module prerequisite for the current
human Phase A review; E-FI-111 records its Perturbation Data correction,
E-FI-112 records the confirmed Segment Replay owner/consumer/legacy-isolation
review, and E-FI-113 records the confirmed K-step Curriculum route plus removal
of the retired state-driven K/M planner from the sealed formal transaction.
E-FI-114 closes DP04. E-FI-115 closes DP05 Frozen GMT offline readiness.
DP06 Paired Rollouts was reviewed without a source change. E-FI-116 closes the
DP07 Repair Gain diagnostics mismatch and confirms its formal producer,
consumer and legacy isolation offline. E-FI-117 removes the retired HSL `dz`
target clamp, proves both translation signs through the official Stage-1
producer/validator/loss edge, and closes DP08 offline. E-FI-118 closes DP09 by
unifying the formal phase identity and proving same-Critic K-transition
recalibration plus critic-only Actor/std optimizer-state preservation. E-FI-119
then fixes the only legal future deployment/Noisy offsets to `(1,2)` and proves
the unmocked offline `870D + 58D -> 928D -> FrontRES 158D / GMT 770D` route.
Formal Runtime Audit Phase B has runtime evidence only for B01. E-FI-122 closes
the first-invalid reset lifecycle offline. E-FI-123 closes the next
first-invalid one-action-K row alignment offline. E-FI-124 closes the formal
request builder's missing grouped-batch dependency and checks the remaining
request-to-checkpoint owner chain for unresolved production symbols; B02-B08
remain live-pending.

### DP10 Future Motion Context Offline Readiness Closure

**Accepted behavior:** the actor receives exactly two future deployment/Noisy
q29 frames at offsets `(1,2)`. Their 58D tail is appended to the 870D host
observation, producing 928D. FrontRES sees only the first 158D; frozen GMT sees
only the original 770D suffix. Clean continuation, future root/global state,
noise labels and perturbation timing remain unavailable to the actor.

**Owner and failure boundary:** `FrontRESFutureIntentLayout` is the one
semantic owner. Config parsing, formal transaction validation and checkpoint-v9
consume that identity and reject any other offsets before mutating runner or
checkpoint state. No fallback, padding or silent correction is admitted.

**Evidence:** focused TEST-04/05/10/16 contracts, the unmocked offline
observation-to-policy route, the independent formal exact-one transaction
contract and the 49/49 deterministic aggregate pass. E-FI-119 records the
details. This closes Phase A offline review only; it does not establish Phase B
runtime or policy-quality evidence.

### Phase B Runtime Audit Control Surface

Phase B is one bounded execution unit, not eight independent live tests. Atlas
06 exposes eight edge cards in the official Stage3 order: launch identity,
sealed scenario/transaction, reset/observation authority, one-action-K/frozen
GMT, paired v007 Gain, storage/return, grouped exact-one update, and committed
checkpoint/telemetry. E-FI-121 installs their read-only assertions in the
existing formal owners and proves them with deterministic offline fixtures;
this is instrumentation evidence, not live evidence.

The eventual run is fixed to fresh K8/M2, 8 envs, one transaction, one update
and checkpoint interval 1. The runtime-confirmed HSL-v2 artifact is
`/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt`.
The historical HSL-v1 path remains rejected. The executable command is:

```bash
cd /hdd1/cyx/FEMR
CUDA_VISIBLE_DEVICES=3 \
CACHE_DIR=/hdd1/cyx/AMASS_G1Segment \
LOG_PATH=/hdd1/cyx/FEMR/v017_phase_b_runtime_audit_gpu3.log \
RUN_NAME=V017_PHASE_B_RUNTIME_AUDIT \
FRONTRES_G5_S4_BOUNDED=1 \
FRONTRES_CHECKPOINT_INTERVAL=1 \
FRONTRES_V015_K_CURRICULUM='8:2:200:500:1300:lower-k8:0.5:linear-joint-v1:1300:2.381,16:3:300:300:900:lower-k16:0.6:linear-joint-v1:900:2.381,32:4:400:300:625:lower-k32:0.7:linear-joint-v1:625:2.381' \
bash run_stage3.sh \
/hdd1/cyx/FEMR/g1_flat_frontres_stage1_hsl/2026-08-04_18-14-12_V017_HSL_V2_FULL/model_warmup.pt \
/hdd1/cyx/AMASS_G1NPZ_Final 8 1 1 train
```

Instrumentation and artifact verification are complete. The first GPU7 run
confirmed B01 and then stopped because formal collection selected
`clean_baseline` before reset installed the sealed scenario. E-FI-122 moves the
mode through the reset request so the environment owner now performs
`install -> mode -> refresh`; focused offline regressions pass without weakening
the command fail-closed check. Rerunning the same bounded command is the
remaining live boundary for B02-B08.

### DP09 Actor & Critic Warmup Offline Readiness Closure

**Accepted behavior:** Stage 3 has exactly three formal phase identities:
`critic_only`, `actor_ramp` and `joint`. During critic-only, Actor/std
parameters and their optimizer state remain exactly unchanged. Actor-ramp
releases the same scalar actor loss continuously; joint uses full Actor and
Critic optimization. At K8->K16 and K16->K32, DR restarts lower, Actor/std
freeze, and the same Critic object and learned state recalibrate.

**Owner and consumers:** `frontres_segment_warmup.py` remains the sole pure
phase/K/DR resolver. Typed transaction requests, the formal transaction owner,
read-only telemetry and checkpoint-v9 consume its immutable phase identity;
none may rename, infer or recompute it. The existing PPO owner remains the
single optimizer-step and Actor/std rollback boundary.

**Legacy seam and isolation:** `actor_warmup` was a retired runtime label still
accepted by request validators and persisted by checkpoint-v9. It is now
rejected. Existing configuration fields ending in `actor_warmup_iterations`
remain transport-compatible duration names only and do not define the public
phase identity.

**Evidence and stop condition:** deterministic S1/S2/S3 contracts prove
monotonic actor-ramp weights, formal transaction/telemetry forwarding,
checkpoint-v9 roundtrip and old-label pre-mutation rejection. K16/M3 and
K32/M4 transactions use the same Critic identity and update it while preserving
pre-existing Actor/std Adam state exactly. Stop on Critic replacement/reset,
Actor/std or optimizer-state drift in critic-only, phase-label fallback,
schedule-duration change, or any HSL/Gain/PPO semantic change.

### DP05 Frozen GMT Offline Readiness Closure

Phase A found two offline-fixable gaps. `OnPolicyRunner.train_mode()` recursively
sets the composed policy to training mode and can therefore undo the internal
GMT policy/normalizer/estimator `eval()` state. Checkpoint-v9 also validates the
770D layout but does not bind the configured GMT checkpoint SHA256, so a strict
resume could accept a shape-compatible different GMT.

The bounded repair keeps ownership in the existing components:

- `FrontRESActorCritic` owns one public frozen-GMT mode invariant: Actor/Critic
  may train, while GMT policy, GMT normalizer and the optional estimator remain
  `eval`, `requires_grad=False` and outside the optimizer;
- the existing `OnPolicyRunner.train_mode()` stays unchanged: its standard
  `policy.train()` call reaches the overridden policy boundary and cannot
  reopen the GMT family;
- `frontres_checkpointing.py` reuses the existing file-SHA owner and persists
  one GMT identity containing checkpoint SHA256, 770D suffix layout and GMT
  normalizer identity; strict resume recomputes it and rejects mismatch before
  actor, Critic, normalizer, optimizer, sampler or receipt mutation.

Preserved behavior: METHOD/GAIN/PPO/HSL, command application, one-action-K,
transaction semantics and the MOSAIC host. Evidence is limited to deterministic
S1 mode/optimizer contracts, S3 checkpoint roundtrip/tamper contracts and the
affected offline regression. Simulator, training and live remain closed.

Status: completed offline in E-FI-115. Parent `.train()` now leaves the
FrontRES Actor/Critic trainable while immediately restoring the GMT policy,
GMT normalizer and optional estimator to frozen inference mode. Checkpoint-v9
now stores one shared GMT identity and rejects a different same-shape artifact
before restoring any mutable runner state. Focused S1/S3 contracts and the
49/49 affected deterministic suite pass.
