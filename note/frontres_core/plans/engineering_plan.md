# FrontRES Current Engineering Plan

Status: `Q1-policy-quality-evaluator-planned`
Updated: 2026-07-17

## Objective

Build an isolated Policy Quality evaluator that compares zero, frozen HSL, and
tested policy actions from one immutable manifest and one identical scoring
state, without changing formal training, periodic online eval, existing offline
eval, or sequence eval behavior.

## Governance Identity

- Concept Figure: `note/architecture/concept/03_frontres_concept_tabs.data.json`.
- Design points: `FRS-DP-02/03/04/05/06/07/08/09` remain unchanged.
- Active contracts: `FRS-METHOD-v011`, `FRS-TRAIN-v003`, `FRS-GAIN-v002`,
  `FRS-PPO-v001`, and `FRS-EVAL-v002` remain active.
- Change class: diagnostic/evaluation tooling; no method-semantic or reward
  migration and therefore no new contract version.
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
```

No step starts before the previous Step End Report is reviewed. A `partial` or
`blocked` step returns control to the user.
