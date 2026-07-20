# FrontRES Policy-Quality And Transaction-Audit Plan

Status: retained Q1/Q2 plan and audit evidence; not the current v015 implementation plan
Updated: 2026-07-19

## Current Method Route

The active implementation surface is:

    note/frontres_core/plans/FRS-v015-future-intent-single-action-k-engineering-plan.md

It implements active FRS-METHOD-v015, FRS-TRAIN-v006, FRS-GAIN-v003, and FRS-PPO-v003. This
file preserves the Q1/Q2 evidence and the Q-E24 mismatch that motivated that
plan. It is not authorization to modify observation, sampler, storage, PPO,
Gain, optimizer, or run a live job.

## Objective

Build an isolated Policy Quality evaluator that compares zero, frozen HSL, and
tested policy actions from one immutable manifest and one identical scoring
state, without changing formal training, periodic online eval, existing offline
eval, or sequence eval behavior.

## Historical Governance Identity

- Concept Figure: `note/architecture/concept/03_frontres_concept_tabs.data.json`.
- Historical audit scope: Q1/Q2 policy quality and transaction causality.
- Current active contracts: `FRS-METHOD-v015`, `FRS-TRAIN-v006`,
  `FRS-GAIN-v003`, `FRS-PPO-v003`, and `FRS-EVAL-v003`.
- Current implementation plan and checklist own the semantic migration;
  this historical document retains no competing active route.
- Current runtime prerequisite: the joint sentinel log reached
  `phase=joint`, `actor_weight=1`, and saved `model_701.pt`; governance sync as
  E70 is required before quality runtime evidence is accepted.

## Non-Scope

- No edits to existing periodic, offline, or sequence evaluator behavior.
- No changes to Segment sampler, Gain, PPO, action projection, warmup, GMT,
  checkpoint format, or formal Stage 3 defaults.
- No policy-quality claim from offline fixtures.
- No long training or checkpoint trajectory run in implementation steps.
- No execution of privileged `supervised_target` as the HSL baseline.

## Source-Of-Truth Table

| Object | Active owner | Consumers | Isolation rule | Evidence gate |
| --- | --- | --- | --- | --- |
| Quality manifest | new `frontres_policy_quality_manifest.py` | quality evaluator only | checkpoint and sampler cannot mutate it | S1/S2 manifest contract |
| Comparison signature | manifest owner | result rows/trajectory aggregator | unmatched rows fail closed | S1 hash/metamorphic test |
| Scoring state | new quality runner owner + existing env reset APIs | zero/HSL/policy routes | capture once; restore before every route | S2 fake lifecycle + S4 sentinel |
| Zero baseline | quality runner owner | canonical rollout/Gain | fixed zero FrontRES action | S2 route contract + S4 execution |
| HSL baseline | frozen actor-only model_200 adapter | canonical rollout/Gain | inference-only; no optimizer/sampler/warmup load | S2 frozen/checkpoint contract + S4 execution |
| Tested policy | current runner policy | canonical rollout/Gain | cannot alter manifest or HSL actor | S2 isolation + S4 execution |
| Quality result | quality runner owner | offline trajectory aggregator | JSON/schema only; no training feedback | S1 schema/aggregation |
| Existing eval paths | existing owners | current users | no imports/calls into new quality owner | S0/S2 isolation regression |

## Why The Work Is Split

The manifest is pure deterministic data, scoring-state control touches the env
lifecycle, counterfactual execution touches checkpoint inference and rollout,
entrypoint wiring touches formal dispatch, and real state equality is live-only.
Combining them would make failures impossible to localize and would risk
silently changing existing evaluators.

## Step Map

### Step Q1-0 / 7: Governance And Runtime Prerequisite - Completed

Objective: close the Phase B document state before Q implementation.

Scope:
- record the joint sentinel as E70;
- update formal audit status, source comments, Runtime Atlas, checklist, and
  Design Point Register;
- state that policy quality remains unproven.

Non-scope: evaluator code and quality claims.

Owner files/modules:
- formal runtime evidence ledger and audit document;
- Runtime Audit Atlas builder/data;
- source audit comments and governed registers.

Expected evidence: documentation consistency contract, Runtime Atlas rebuild,
`git diff --check`.

Stop condition: model/checkpoint/iteration identity differs across any current
artifact, or E70 is promoted to quality evidence.

Step result: E70 records joint `actor_weight=1.0`, four accepted updates with
valid rows, frozen GMT, and complete `model_701.pt`. Runtime closure is distinct
from policy-quality evidence. Next executable step is Q1-A.

### Step Q1-A / 7: Immutable Manifest And Signature - Completed

Objective: implement the pure comparison-identity owner.

Scope:
- create immutable manifest/item/state-identity/result-route data objects;
- canonical serialization and deterministic comparison signature;
- validate motion/frame/perturbation/K/seed/checkpoint-independent identity;
- reject missing, mutable, duplicate, or mismatched rows.

Non-scope: runner, env, checkpoint loading, rollout, Gain, and CLI.

Owner files/modules:
- create `source/rsl_rl/rsl_rl/frontres/frontres_policy_quality_manifest.py`;
- create `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_manifest_contract.py`.

Expected evidence: S1 `T-schema/T-hash/T-permute/T-missing/T-immutable` with a
hand-checkable manifest fixture.

Stop condition: the same semantic item hashes differently, a changed control
variable retains the same signature, or checkpoint/sampler state enters the
manifest identity.

Step result: Q-E1 confirms canonical immutable serialization, order-stable
comparison signatures, control-variable sensitivity, duplicate/missing-field
rejection, and checkpoint/sampler isolation. No runner or evaluator path was
changed. Next executable step is Q1-B.

### Step Q1-B / 7: Scoring-State Capture And Restore - Completed

Objective: establish a checkpoint-independent scoring start state.

Scope:
- define the exact dynamic-state fields required at eval start;
- run zero-FrontRES/GMT-only preroll once;
- capture state after preroll and restore it before each route;
- hash root/joint pose and velocity, command/reference/correction caches,
  episode lifecycle, origin, frame/K/perturbation, and relevant RNG state.

Non-scope: HSL loading, tested policy comparison, Gain, trajectory aggregation,
and old evaluator changes.

Owner files/modules:
- create `source/rsl_rl/rsl_rl/runners/frontres_policy_quality_eval.py` with
  state capture/restore helpers only in this step;
- create `source/rsl_rl/rsl_rl/tests/frontres_policy_quality_state_contract.py`.

Expected evidence: S1/S2 `T-state/T-role/T-frame/T-cache/T-RNG/T-restore` using
a semantically complete fake env lifecycle.

Stop condition: any restored field differs, preroll reads the tested policy,
or implementation requires modifying existing eval owners.

Step result: Q-E2 confirms exact zero-action preroll and offline capture/
restore of robot, lifecycle, reference/correction cache, per-env perturber, and
RNG state under one `initial_state_hash`. Existing evaluators remain untouched;
real simulator equality remains reserved for Q1-F. Next executable step is Q1-C.

### Step Q1-C / 7: Zero, Frozen-HSL, And Policy Execution - Completed

Objective: execute three counterfactual routes from the identical captured
state through existing rollout and canonical Gain owners.

Scope:
- load model_200 residual actor as an inference-only HSL adapter;
- verify observation layout and normalizer identity explicitly;
- execute zero/HSL/policy routes after state restore;
- emit `QUALITY-ID-01`, `QUALITY-ACTION-01`, `QUALITY-GAIN-01`, and
  `QUALITY-EXEC-01` result objects;
- fail closed on state/signature mismatch.

Non-scope: training optimizer, sampler, warmup state, PPO update, trajectory
comparison, and changes to canonical Gain/action application.

Owner files/modules:
- extend `runners/frontres_policy_quality_eval.py`;
- create `tests/frontres_policy_quality_eval_contract.py`.

Expected evidence: S2 `T-counterfactual/T-frozen/T-source/T-shape/T-forward/
T-isolation/T-metamorphic`; zero action remains zero, HSL is frozen, policy is
independent, and all routes preserve one comparison signature.

Stop condition: HSL requires privileged target execution, runner load mutates
optimizer/sampler/warmup, or route identity differs.

Step result: Q-E3 closes offline zero/frozen-HSL/policy orchestration under one
restored state, explicit observation/normalizer identity, full-6D bounds, and
training-state isolation. Canonical owner callbacks are shared but not yet
wired to the formal runner; that integration is Q1-D. Next executable step is
Q1-D.

### Step Q1-D / 7: Thin Entry And Old-Path Isolation - Completed

Objective: expose a dedicated `policy_quality_eval` mode without changing old
runtime behavior.

Scope:
- add CLI/config parsing for manifest, HSL checkpoint, tested checkpoint, and
  result path;
- add a thin `OnPolicyRunner` connector and `train.py` dispatch;
- old modes must never import/call the quality evaluator at runtime;
- quality mode must not call sampler sampling or optimizer update.

Non-scope: editing old evaluator functions, changing Stage 3 presets/defaults,
or sharing a quality flag with sequence/offline/periodic eval.

Owner files/modules:
- modify `scripts/rsl_rl/train.py`, CLI owner, and
  `runners/on_policy_runner.py` only as thin connectors;
- add `tests/frontres_policy_quality_entrypoint_contract.py` and isolation
  assertions to the aggregate suite.

Expected evidence: S0/S2 `T-route/T-import/T-mode/T-no-call/T-state`; all old
modes have zero quality-owner calls and quality mode has zero old-eval calls.

Stop condition: an old mode changes output/call order, quality mode enters
training, or connector owns evaluation logic.

Step result: Q-E4 closes dedicated CLI/MODE dispatch, lazy runner delegation,
request validation, cold-start checkpoint isolation, and zero calls into old
eval/training branches. Missing formal executor fails closed. Q1-E must prove
that executor's real owner wiring offline before Atlas/live work.

### Step Q1-E / 7: Quality Atlas And Offline Preflight

Objective: make instrumentation human-readable and prove the complete offline
quality route before live execution.

Scope:
- assemble the formal manifest executor from current observation,
  task-space-application, rollout, canonical Gain, and execution owners;
- prove executor isolation with fake owner adapters before any simulator run;
- create a Quality Audit Atlas in the existing reading-atlas viewer;
- add eight planned `QUALITY-*` cards, while Q1 runtime initially populates
  ID/ACTION/GAIN/EXEC and marks DATA/CREDIT/UPDATE/TRAJECTORY pending;
- synchronize source comments, checklist, test inventory, and evidence ledger;
- run manifest, state, evaluator, entrypoint, isolation, and aggregate suites.

Non-scope: simulator quality claims and checkpoint trajectory.

Owner files/modules:
- `note/architecture/runtime/*quality*.data.json` and builder/view entry;
- quality source comments, current checklist, test control board, evidence
  ledger, and aggregate suite.

Expected evidence: S0-S2 Atlas/source/checklist identity, 8 cards with valid
links, all focused contracts PASS, aggregate suite PASS, `git diff --check`.

Stop condition: source/Atlas/checklist IDs disagree, an old eval path imports
the quality owner, or any offline identity/isolation contract fails.

Step result: completed by Q-E6. Q-E5 alone proved executor order/schema with
fake owners and was insufficient. The corrected official entry now installs a
production `FrontRESPolicyQualityFormalOwnerBundle` and reaches canonical
reset, observation, action, rollout, Gain, and execution adapters through an
independent control flow. The S2 official-entry contract observes all six
owners and unchanged optimizer/sampler/warmup state; the aggregate suite passes
`51/51`. Q1-F remains blocked pending explicit user authorization.

### Step Q1-F / 7: Live State-Identity Sentinel

Objective: prove the real simulator restores one scoring state for
zero/HSL/policy without contaminating existing eval/training state.

Scope:
- one immutable manifest item, one seed, one checkpoint, minimal env count;
- print state hashes before each route and compact quality snapshot;
- verify optimizer/sampler/warmup are unchanged;
- write observed facts beside source probes and into quality evidence.

Non-scope: checkpoint trajectory, generalization, reward/PPO tuning, or long
training admission.

Owner files/modules:
- independent quality evaluator and its Quality Atlas/evidence rows only.

Expected evidence: S4 `T-live/T-state/T-identity/T-frozen/T-isolation`; all
three pre-rollout state hashes and comparison signatures match.

Stop condition: any hash differs, HSL/policy checkpoint identity is ambiguous,
old eval/training state changes, or a route cannot execute canonical Gain.

Preparation result: Q-E7 freezes one reviewable item at
`note/testing/manifests/frontres_policy_quality_q1f_single_v1.json`, using the
actor-update-free `model_200.pt` HSL baseline, resumed-lineage `model_701.pt`
policy, KIT/572 frame 163, local_rp at DR 1.25, K=8, and seed 42. Item and
manifest signatures pass S1 identity checks. Live remains blocked until human
review and server checkpoint existence/SHA-256 preflight.

Runtime result: Q-E9 records one completed three-route run. The immutable
signature and all three initial-state hashes match, and the corrected canonical
Gain tensors are one value per paired item. The first interpretation that a
zero mask on the noisy role meant missing corruption was rejected after reading
the active command owner: one policy-row realization is copied to the noisy
base row by `_sync_frontres_pairs(sync_perturbation=True)`, while clean is reset
to the unperturbed reference. Q1-F remains partial because the current artifact
does not persist local-coordinate robot-state deltas or cached quaternion deltas
for the policy/noisy pair; zero-route Gain `0.007556` is therefore an observed
noise floor, not yet classified as simulator divergence or identity failure.

Instrumentation result: Q-E10 adds `QUALITY-ID-01` B4 at the exact boundary
after canonical reset and scoring-state capture but before any counterfactual
route. It persists and prints policy/noisy world-root, env-origin, local-root,
root pose/velocity, joint, and cached perturbation deltas, plus policy/clean
cache deltas proving corruption presence. A semantic fake deliberately uses
different world origins while keeping local dynamics/cache matched. Focused
owner and Atlas contracts pass. The next action is one rerun of the same Q1-F
manifest; no reset, Gain, PPO, or checkpoint parameter changes are permitted.

Closure result: Q-E11 closes Q1-F on the real simulator. The three routes share
one manifest signature and one scoring-state hash. Policy/noisy local root,
root pose/velocity, joint state, and cached perturbation deltas are zero within
floating precision; the 40 m world-root difference is exactly the env-origin
difference. Policy/clean cached quaternion delta `0.061488` proves the local_rp
corruption is present. The zero route defines an observed eight-step paired-env
noise floor `0.007556`. On this item HSL-Zero is `0.041726`, Policy-Zero is
`0.041802`, and Policy-HSL is only `0.000076`; therefore both repair routes beat
zero, but PPO improvement over HSL is unconfirmed. The next bounded step is Q2:
one immutable bank with at least 8 motions and 2 matched seeds. No code or Gain
change is authorized before that matched bank is reviewed.

### Step Q2 / 7: Counterfactual Oracle Bank

Status: manifest accepted; independent offline reporter implemented.

Objective: determine whether HSL and model_701 beat explicit zero across a
fixed 8-motion x 2-seed bank, and whether Policy-HSL exceeds each item's own
zero-route noise floor.

Scope, thresholds, item bank, cost, stop conditions, and the independent
offline reporting gap are defined in
`note/frontres_core/plans/policy_quality_q2_counterfactual_plan.md`. Proposed
manifest: `note/testing/manifests/frontres_policy_quality_q2_bank_v1.json`.

Non-scope: live execution before human approval, training-code changes,
checkpoint trajectory, PPO/Gain tuning, and long training.

Expected evidence: S1/S2 manifest and report contracts, followed only after
offline closure by S4 `Q-matched/Q-causal` per-item evidence. The reporter
preserves every item and separates technical corruption from a valid negative
scientific result.

Stop condition: identity/schema failure, non-finite or non-scalar route Gain,
role corruption failure, or HSL positive-control failure on both seeds for at
least 3 motions.

Closure result: Q-E13 passes the technical matched-comparison gate for all 16
items but triggers the scientific stop condition. HSL-Zero is negative on both
seeds for 4/8 motions and positive on only 1/8. Policy-HSL has no stable bank
advantage. The first divergence is HSL/proposal versus canonical Gain, before
PPO. Q3 and long training remain blocked; the next bounded step is an offline
HSL/Gain learnability decomposition, not another live run.

### Step Q2-A: Offline Gain Learnability Decomposition

Status: completed by Q-E14.

Objective: distinguish HSL execution degradation from Repair Cost dominance
and paired-env zero noise without changing the active Gain formula.

Scope: infer the runtime Repair weight from persisted Style/Physics/Cost/total,
compute pre-cost route differences, and classify each item before aggregation.
Non-scope: weight tuning, PPO, target reconstruction, and simulator reruns.

Closure: the effective Repair weight reconstructs as 0.15. Only 1/16 HSL
items is a clear Repair-Cost-dominance failure; 5/16 already degrade
Style+Physics before cost. Therefore removing or weakening Repair Cost is not a
supported root-cause fix. Walking-run zero noise is dominated by paired Physics
divergence and remains unsuitable for resolving K=8 route differences.

### Step Q2-B: HSL Output-To-Target Alignment

Status: integrated offline by Q-E15; S4 evidence pending.

Objective: compare model_200 full-6D output with the canonical post-step HSL
supervised target on the failed Q2 items.

Scope: reuse `frontres_hsl_rollout_target.py` semantics inside only the
dedicated policy-quality evaluator, persist target/weight/action alignment, and
add S1/S2 contracts before one bounded rerun. Non-scope: existing online,
offline, sequence eval, training, Gain, or PPO control flow.

Reason for split: the target depends on post-`env.step` Clean/Noisy/FEMR root
states and action-cone projection. It is absent from the Q2 result and cannot
be reconstructed honestly from model_200 or the persisted action trajectory.

Offline implementation: the canonical HSL owner now returns its target/weight
object while retaining the training-default transition write. The dedicated
quality evaluator calls it with `write_transition=False` only for the HSL
route and persists K-step target, sample/harm weight, nonzero mask, cosine, L2,
and per-dimension sign agreement. The Q2 reporter can require this schema with
`--require-hsl-supervision`. Existing online/offline/sequence evaluators and
training control flow are unchanged.

Live correction: Q-E16 found that the first Q2-B attempt incorrectly required
the Stage 2 transition-write flag to be enabled. The audit availability and
training-write gate are now separate: formal training retains the original
default, while the dedicated evaluator computes without transition mutation.
S1/S2 flag-off regression evidence passes; S4 remains pending.

Closure result: Q-E17 completes Q2-B on the real matched bank. All 16 items
contain K=8 canonical targets. Model_200 action directions are usually aligned
with those targets (item-level cosine median 0.910), but action norms remain in
a narrow 0.133-0.149 range while target norms span 0.006-0.116. The median
action/target norm ratio is 10.65x and the maximum is 23.29x. The first quality
failure localizes an HSL initialization defect in Stage 2 magnitude
calibration; it does not explain why Stage 3 HRL retained that defect instead
of correcting it. Reset identity and action direction are excluded at this
boundary, but PPO quality causality and Gain-to-credit correction are not.
Q3 and long training remain blocked.

### Step Q2-C1 / Q2-C2: Stage 2 Magnitude Calibration Audit

Status: Q2-C1 partial; Q2-C2 completed by Q-E18.

Objective: localize whether Q2-B over-correction comes from unprovable
checkpoint lineage, the observed HSL target scale, or ineffective supervised
loss/gradient balance.

Q2-C1 audits checkpoint iteration, source lineage, training objective, and
effective supervised-loss configuration. Q2-C2 replays the active supervised
formula on persisted Q2-B action/target/sample/harm evidence and reports every
raw/weighted loss component and proposal-gradient contribution. The Q2-B bank
is held-out target evidence, not the complete Stage 2 training distribution.

Owner modules: `frontres_checkpointing.py`, `frontres_unified.py`, and a
dedicated offline policy-quality audit module. Required evidence: S1
`T-schema/T-lineage/T-value/T-gradient/T-scale/T-metamorphic`. Stop when the
first contradicted boundary is identified or the only remaining gap requires
the original Stage 2 log/checkpoint artifact. No live run, PPO/Gain change,
retraining, Q3, or long training is authorized by this step.

Observed result: the held-out Q2-B bank contains 128 valid K-step rows. Under
the active current weights, position-direction loss contributes proposal-level
gradient L2 `1.157e3`; magnitude and over contribute `0.00619` and `0.00494`.
Direction is therefore `1.04e5x` their combined scale gradient before the
shared global gradient clip. This is the first deterministic formula-level
contradiction and explains why direction can improve while magnitude remains
poorly calibrated. Q2-C1 remains partial because the local checkout has no
model_200/model_warmup artifact, and `save_runner()` does not persist the
effective supervised config or source-checkpoint identity. Do not infer the
old model's training config from the current runtime print.

### Step Q2-D: Stage 3 Over-Amplitude Correction Causality

Status: Q2-D1/Q2-D2 completed offline; Q2-D3 partial. The physical scale sweep exists, while transaction-complete Gain identity and the real official PPO credit artifact await one bounded rerun/update sentinel.

Objective: determine why Stage 3 HRL did not correct the over-amplitude HSL
initialization exposed by Q-E17/Q-E18.

Scope: reuse fixed Q2 motion/seed/observation evidence; sweep scaled versions of
the same HSL action; identify the locally Gain-preferred magnitude; then trace
that same sample through Gain -> return/advantage -> one controlled PPO update
-> policy mean. Non-scope: changing HSL/PPO/Gain weights, retraining Stage 2,
starting Q3, or long training.

Reason for split: previous PPO contracts prove the generic update mechanism can
change parameters, while Q-E17/Q-E18 prove an erroneous HSL starting point.
Neither proves that real over-amplitude samples receive corrective credit or
move the policy mean toward a better magnitude.

Required evidence: Q-formula action-scale sweep, Q-causal credit-sign trace,
and Q-causal one-batch mean-direction test on the same matched evidence. Stop at
the first contradiction: Gain sensitivity, advantage construction, PPO update
direction, or training-distribution weighting.

Step split:

- Q2-D1: independent scaled-HSL route executor and immutable result schema for
  `0/0.25/0.5/0.75/1/1.25x`; it reuses only lower-level state restore,
  observation, action application, rollout, Gain, and execution owners.
- Q2-D2: offline credit/update-direction oracle. It records Gain/return/
  advantage identity and evaluates the Gaussian score-gradient
  `advantage * (raw_action - old_mean) / sigma^2`, then checks whether a cloned
  one-update mean delta projects toward the Gain-preferred action.
- Q2-D3: one bounded real execution only after D1/D2 contracts pass. It is the
  sole owner of physical Gain ordering and real-batch update direction.

Offline result: the independent evaluator restores one state before each of six
scaled-HSL routes and calls the existing observation/action/rollout/Gain/
execution owners without installing or modifying the old quality evaluator.
The credit oracle exposes the Gaussian score direction
`E[A*(raw_action-old_mean)/sigma^2]`; the controlled-update oracle runs the
canonical Segment PPO loss on a policy clone and verifies mean projection while
leaving the source policy unchanged. These contracts prove instrumentation and
formula direction only. They do not establish which scale wins physical Gain
or what direction the failed real Stage 3 batch supplies.

Q-E22/Q-E23 narrowed the remaining boundary. The first scale artifact shows
that smaller HSL magnitudes often beat 1.0x, but its Gain transaction metadata
was `UNCONFIRMED`; this result is diagnostic, not closure evidence. The Q2-D
owner now injects a complete transaction/batch identity before canonical Gain,
and the official Stage 3 single-update owner can atomically capture the exact
pre-update bounded/raw action, old mean/sigma, Gain, return, advantage, valid
mask, and segment ID tuple. Offline S1/S2 contracts prove fail-closed identity,
row alignment, official-owner reachability, and no storage mutation. No Gain,
PPO, sampler, or optimizer semantics changed.

### Step Q2-E: Double Segment Replay Transaction Alignment

Status: code-confirmed mismatch retained as the superseded v013 migration baseline.

Objective: compare the accepted `FRS-METHOD-v012` transaction against the
formal code path.

Scope: trace sampler selection, trial expansion, fixed-policy identity,
storage accumulation, PPO-batch construction, and `optimizer.step` timing.

Non-scope: no sampler/PPO/Gain implementation change, no new live test, and no
exploration-sigma decision.

Expected evidence: code-confirmed call order and row roles, including attempts
per Segment before each optimizer step and whether all eligible attempts share
one old-policy snapshot.

Stop condition: classify the formal path as aligned, partially aligned, or
contract-mismatch and return control before implementation planning.

Read-only result: `contract-mismatch`. `sample_rollout_rows()` expands each
Segment into exactly one `policy` row followed by `search` rows. Storage masks
out every `search` row, and `run_frontres_segment_live_probe()` immediately
calls `run_frontres_segment_single_update()` for the current sampler step.
`run_frontres_segment_live_update_loop()` repeats this complete sample ->
rollout -> optimizer transaction for every configured `update_step`; it does
  not accumulate repeated on-policy attempts under one fixed old policy before
  the optimizer step. The v013 plan is superseded; the separate bounded v015
  implementation plan is active. No implementation step has started from this
  historical audit result.

## Planned Step Order

```text
Q1-0 governance
-> Q1-A pure manifest
-> Q1-B state capture/restore
-> Q1-C counterfactual execution
-> Q1-D isolated entrypoint
-> Q1-E Atlas/offline integration
-> user approval
-> Q1-F one live sentinel
-> Q2 reviewed 8-motion x 2-seed counterfactual bank
```

No step starts before the previous Step End Report is reviewed. A `partial` or
`blocked` step returns control to the user.
