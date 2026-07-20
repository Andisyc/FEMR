# FRS-v013 Engineering Plan, superseded fixed-65D-tape route

Status: superseded as the active plan by FRS-v015-future-intent-single-action-k-engineering-plan.md; preserved as v013 implementation/evidence history only
Updated: 2026-07-19
Superseded contracts:

- FRS-METHOD-v014-single-policy-row-k-evidence.md
- FRS-TRAIN-v005-single-policy-row-k-evidence-transaction.md
- FRS-PPO-v003-single-policy-row-k-evidence-grouped-reduction.md

## Historical Objective

Bring the formal Stage 3 route into alignment with the accepted local-repair
method:

    fixed Noisy future context
    + repeatable Clean dynamic x_t
    + M on-policy attempts under one frozen policy
    + grouped cross-Segment PPO
    -> one semantically valid update

This plan starts from a documented code mismatch. It authorizes no implementation
step merely by existing; each step needs its own acceptance evidence and Step
End Report.

## Active Semantic Boundaries

- x_t is the K-step dynamic start. Clean x_t is reproducibility state, not
  actor-visible Clean reference.
- Current, future, and K-step executed reference are taken from one fixed Noisy
  segment for each repeated-attempt scenario.
- Future context is sparse, positive, local, deployment-available command
  context. H and K are separate horizons.
- All valid policy-sampled M attempts may contribute PPO evidence. Best-of-M,
  manual, and counterfactual results are replay evidence only.
- Motion -> Segment -> attempt -> one policy row controls actor-loss mass.
  K-step validity remains return evidence only; priority affects sampling only.

- The segment-selection owner samples that Noisy sequence once before the
  M-attempt loop; resets may restore x_t but may not resample, mutate, or mix
  the scenario reference. Its semantic binding ends after M-attempt evidence
  aggregation, while an immutable cache may retain it for replay evidence.

## Source-Of-Truth Migration Table

| Semantic object | Active owner | Active consumers | Legacy path / retirement rule | Implementation proof | Formal-route proof | Live gap |
| --- | --- | --- | --- | --- | --- | --- |
| Fixed Noisy reference segment and future offsets | command tape plus FrontRES observation bridge | Stage 3 actor, GMT execution, rollout metadata | Legacy joint/static window is isolated; forbid Clean/mixed fallback | S1 carrier/hash | S2 reset -> actor -> K rollout completed offline | S3 checkpoint/HSL and S4 per-row reference hash |
| Clean dynamic start x_t | segment cache/reset owner | repeated attempt reset | Do not add Noisy physical prefix; pose-only reset remains invalid | S1 state identity | S2 M resets reproduce same state | S4 reset hashes |
| Frozen old-policy M-attempt transaction | sampler, live sampler, live probe, storage | PPO batch and replay evidence | Retire first-policy plus later-search credit and immediate per-step update | S1 role/snapshot/order | S2 multi-Segment accumulator -> one update | S4 one transaction trace |
| Grouped actor-loss reduction | frontres_segment_ppo.py | actor optimizer | Retire flat valid-K-step-row mean and all priority/focal/M/K/evidence-count multipliers | S1 K-A row/formula/permutation/metamorphic | S2 storage metadata -> grouped loss | S4 policy-row mass diagnostics |
| Future-layout checkpoint identity | HSL/warmup/checkpoint owners | resume and Stage 3 load | Legacy 870D artifact must fail closed for v013 route | S1 schema/load rejection | S3 resume/connectivity | S4 loaded-layout report |

## Why The Work Is Split

Future-reference provenance changes the actor interface; transaction assembly
changes collector and storage lifecycle; grouped reduction changes the algorithm
owner. These are independent owner and acceptance boundaries. Formal runner
integration and live simulator identity need separate evidence, so they are not
combined with local implementation.

## Step Map

### Step 0 / 5: Contract, Figure, Architecture, Plan, And Matrix

Status: completed as documentation only.

Objective:

- activate the accepted v013/v004/v002 semantic owners;
- show the active method and the current code mismatch without claiming
  implementation;
- create this plan and the paired acceptance checklist.

Scope:

- contracts, registry/history, Concept Figure, Method-to-Code atlas, plan, and
  test matrix.

Non-scope:

- Python code, tests, training, checkpoints, and live runs.

Expected evidence:

- S0 T-doc/T-map registry and atlas consistency;
- no implementation claim.

Stop condition:

- a Concept Figure block, active contract, or architecture card disagrees on
  the method boundary.

### Step 1 / 5: Fixed Noisy Future-Context Interface

Status: Step 1A completed with a code-confirmed mismatch. Step 1B-S1 and
Step 1B-S2 completed the sampler lifecycle and command/reset/actor offline
connectivity. Checkpoint/HSL and live evidence remain separate gates.

Objective:

- materialize one fixed Noisy reference segment and append sparse positive
  future command references to the actual FrontRES actor input.

Scope:

- command-window construction and provenance;
- future-offset configuration and serialized actor layout;
- actor input, normalizer, HSL interface, and checkpoint schema;
- metadata that proves current/future/K-step execution references share one
  noisy_segment_hash.

Owner files/modules:

- source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py
  set_frontres_reference_window;
- source/whole_body_tracking/whole_body_tracking/tasks/tracking/flat_env_cfg.py;
- source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py;
- source/rsl_rl/rsl_rl/runners/frontres_runtime.py;
- source/rsl_rl/rsl_rl/runners/frontres_warmup.py;
- source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py;
- formal checkpoint owner and the focused observation/checkpoint tests.

Non-scope:

- Clean future input, perturbation labels, noise timing labels, a Noisy physical
  prefix before x_t, new root/contact channels, or a new policy head.

Expected evidence:

- S1 T-layout/T-provenance/T-hash/T-clean-isolation;
- S1 T-checkpoint/T-missing for old-layout rejection;
- S2 T-connect from formal segment reset through actor input and K rollout.

Stop condition:

- any actor-visible reference field is Clean, comes from a different Noisy
  segment, lacks a stable offset layout, or H is conflated with K.

#### Step 1A: Reference-Provenance Probe

Scope:

- trace the existing joint reference from command window through the current
  perturbation owner;
- add one minimal semantic contract probe at the existing motion-command test
  owner;
- record whether current command/current-window input is raw/Clean or
  perturbation-applied before wiring it into the actor.

Non-scope:

- no future actor prefix, config layout, normalizer, checkpoint, Segment
  transaction, PPO, or live simulator change.

Files:

- source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py,
  read-only owner audit;
- source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py,
  read-only perturbation audit;
- source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py,
  deterministic provenance probe.

Owner module:

- MultiMotionCommand.command and its reference-window path.

Core parameter path:

    raw joint reference
    -> joint_pos perturbation owner
    -> _gather_future_by_motion / command
    -> reference-window override
    -> candidate future actor context

Test class:

- core parameter path, deterministic fake command and explicit perturber.

Command:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python
    source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py

Expected result:

- the probe prints the exact command/reference shapes and proves whether
  command invokes joint perturbation; it must fail if a future actor input is
  falsely described as Noisy while the current command path bypasses the
  perturber.

Stop condition:

- classify the current future-reference source as fixed Noisy, raw/Clean, or
  unconfirmed. Only then choose the Step 1B materializer owner.

Observed result:

- current command/reference-window reads bypass the joint perturbation owner;
- RP specialist corruption is root roll/pitch while joint noise is disabled;
- cache noisy_variant adaptation repeats one state across K+1 frames.

The source is therefore neither a verified fixed Noisy future trajectory nor a
valid actor future context under FRS-METHOD-v014. Evidence:

    note/testing/evidence_ledger_v013_future_context_2026-07-19.md

The representation decision is resolved by Step 1B below: selection must
materialize one complete fixed Noisy sequence for the scenario. The concrete
materializer owner and carrier layout remain S1 implementation questions.

#### Step 1B: Fixed Noisy-Segment Lifecycle

Status: S1 lifecycle and S2 command/actor-route connectivity are completed
offline. S3 checkpoint/HSL and any live runtime remain later gates.

Decision:
- when a Segment is selected, sample one complete Noisy reference sequence
  before attempt 1 and bind it to that scenario;
- reuse that immutable sequence for all M attempts after every Clean x_t reset;
- use the same sequence for actor current/future context and K-step GMT
  execution; close the scenario only after M-attempt evidence aggregation.

Scope:
- move the existing corruption draw from attempt-time behavior to the
  segment-selection/reference-cache lifecycle;
- persist a scenario identity, provenance, coverage, and noisy_segment_hash;
- forbid retry-time perturbation draws and mixed/Clean reference fallbacks.

Non-scope:
- changing the local-repair x_t boundary, adding a Noisy physical prefix,
  changing M/PPO credit semantics, or choosing a new perturbation family.

Expected evidence:
- S1 T-lifecycle/T-immutability/T-hash/T-clean-isolation;
- S2 T-connect showing one selected scenario reaches all M actor windows and
  K-step executions unchanged.

Stop condition:
- any retry samples a new corruption realization, an actor-visible reference
  lacks the selected scenario identity, or one H/K frame comes from another
  sequence.

##### Step 1B-S1: Immutable Scenario Binding

Status: completed as a sampler-domain S1 boundary; S2 now consumes this owner
without transferring command, reset, actor, or PPO authority into it.

Objective:

- establish the local selection-time lifecycle before changing any command,
  actor, reset, PPO, or simulator path.

Scope:

- bind one materialized Noisy sequence to each base sampled source row;
- fan that immutable scenario out to all rows sharing its source index;
- retain scenario identity, provenance, coverage, and a content hash in a
  sampler-domain object.

Owner files/modules:

- source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py;
- source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_segment_lifecycle_contract.py.

Non-scope:

- MultiMotionCommand materialization, actual actor H layout, reference-window
  installation, physical reset, trial role/PPO changes, or live simulation.

Core parameter path:

    segment_id + source_index + K + H
    -> one injected materializer draw
    -> immutable Noisy sequence + scenario_id + noisy_segment_hash
    -> all M rows for that source

Expected evidence:

- S1 T-lifecycle/T-immutability/T-hash/T-coverage/T-clean-isolation;
- a meaningful toy fixture proves materializer calls equal selected base
  scenarios, not expanded trial rows.

Stop condition:

- one source index has more than one identity or hash, sequence mutation is
  externally observable, the factory is invoked per retry, Clean data enters
  the scenario payload, or coverage is shorter than K + H_max.

##### Step 1B-S2: Command-Owned Fixed-Tape Connectivity

Status: completed at S1/S2 offline connectivity; no live route was run.

Objective:

- make the S1 sealed scenario a real executable reference source for reset,
  current command, Actor H context, and K-step GMT execution.

Confirmed carrier and owner:

    MultiMotionCommand owns materialization and cursor state
    T_tilde_s [L, 65] = [q(29), dq(29), anchor_pos(3), anchor_quat(4)]
    L >= K + max(H); frontres_future_offsets is required and nonempty

Source-of-truth migration table:

| Semantic object | Active owner | Active consumers | Legacy path | Isolation rule | Implementation / integration proof | Live gap |
| --- | --- | --- | --- | --- | --- | --- |
| fixed Noisy tape | `MultiMotionCommand` | index reset, current command, actor H, GMT K | 29/58D `reference_window` | fixed-tape mode clears and outranks the legacy window | S1 carrier fixture / S2 selection-reset-actor-K contract | S4 real rollout |
| scenario identity/hash | sampler lifecycle | batch, reset request, diagnostics | per-expanded-row DR plan | one materialization per `source_index` | S1 lifecycle / S2 M-row propagation | S4 log |
| Clean x_t | reset owner | robot/controller dynamics | noisy preroll | no actor reference; no sampler call on reset | S2 reset fixture | S4 reset state |

Scope:

- parameterize the selection connector by required H and materialize one 65D
  tape per base source;
- install that sealed tape on all role rows, with `policy/candidate/noisy`
  executing it while `clean` remains the physical Clean baseline and every
  actor row receives the same Noisy H context;
- route current command/anchor and H reads through the command cursor, then
  fail closed when a legacy actor/normalizer layout is asked to consume it;
- add focused S1/S2 deterministic contracts.

Non-scope:

- a Noisy physical prefix, Clean actor future, noise labels/timing, a new policy
  head, checkpoint migration/HSL retraining, Double Segment transaction,
  grouped PPO, or any live run.

Owner files/modules:

- `commands.py` command tape materialization, installation, cursor, and legacy
  isolation;
- `frontres_segment_live_sampler.py` selection-time S1 binding;
- `frontres_segment_live_probe.py` reset-request and actor-observation bridge;
- `frontres_segment_stage1_env_hooks.py` role-aware tape installation;
- `frontres_runtime.py` actor-layout fail-closed bridge;
- focused command/connectivity contracts.

Expected evidence:

- S1 T-carrier/T-layout: ordered 65D materialization, copy/hash immutability,
  and K+H coverage;
- S2 T-connect/T-no-mixed-reference: one source hash reaches every M row,
  reset request, current command, H offset, and K frame;
- S2 T-legacy-isolation: no joint-only/static/re-sampled fallback while tape
  mode is active; old actor layout rejects rather than silently truncating.

Stop condition:

- an H lookup advances the K cursor; a Clean or static reference reaches actor
  input; a reset invokes scenario materialization; a Noisy role and its actor
  context disagree in hash; or implementation requires transaction/PPO/live
  semantics to continue.

### Step 2 / 5: Frozen-Policy Double Segment Transaction

Objective:

- collect multiple selected Segments and all M >= 2 policy-sampled attempts
  under one unchanged old policy before writing a PPO-ready transaction.

Scope:

- sampler trial expansion and role semantics;
- Clean x_t reset plus fixed Noisy scenario identity for every attempt;
- transaction accumulator, aligned storage metadata, and one update boundary;
- post-collection replay-evidence aggregation.

Owner files/modules:

- source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py;
- source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py;
- source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py;
- source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py.

Non-scope:

- best-action PPO selection, manual/search action credit, priority-weighted
  PPO, Gain formula changes, and long training.

Expected evidence:

- S1 T-role/T-order/T-snapshot/T-reset/T-metadata;
- S2 T-connect/T-single-step showing multiple Segments and all policy attempts
  reach one transaction and exactly one optimizer call.

Stop condition:

- any optimizer mutation occurs during collection, later policy samples are
  relabelled search, a repeated attempt gets a new Noisy scenario, or x_t
  cannot be reproduced.

#### Step 2-S1a: Declarative All-Policy Attempt Plan

Status: completed offline. This is a sampler-only semantic boundary, not a policy
snapshot or live transaction implementation.

Scope:

- construct an immutable row plan for multiple selected Segments;
- require at least two policy-attempt rows per selected Segment;
- preserve transaction and caller-declared policy-snapshot identities, source
  grouping, trial order, and per-Segment K;
- label every planned attempt `policy`.

Owner module:

- `source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py`.

Non-scope:

- parameter snapshot capture or verification;
- Clean reset, fixed Noisy tape lifecycle, command/actor route, storage,
  optimizer calls, replay update, and grouped PPO reduction.

Evidence:

- S1 deterministic plan contract: `S >= 2`, every group has `M_s >= 2`, all
  roles are `policy`, ordering is source-major then trial-major, and IDs/K are
  preserved without sampler-state mutation.
- focused sampler contract plus aggregate offline suite both pass; see
  `evidence_ledger_v013_future_context_2026-07-19.md` E-TX-1 through E-TX-3.

Stop condition:

- if proving the plan requires a live reset, a new Noisy materialization, or an
  optimizer mutation, stop and defer that work to Step 2-S1b/S2.

#### Step 2-S1b: Frozen Snapshot And Identity Carrier

Status: completed offline. This identity-connectivity boundary does not activate
the formal transaction route.

Objective:

- convert S1a's caller-declared `policy_snapshot_id` into an identity derived
  from the real pre-collection policy state;
- carry one transaction's policy/motion/segment/trial/K/Noisy identities through
  the existing batch, Clean-reset request, and independent storage carrier.

Scope:

- make `frontres_segment_live_sampler.py` capture and verify a deterministic
  fingerprint of `runner.alg.policy.state_dict()`;
- pre-bind the S1a plan and frozen snapshot to the selected batch before fixed
  Noisy materialization, so the tape lifecycle consumes the same transaction ID;
- after the sealed tape exists, finalize per-row motion/start/segment/source/
  trial/K/role/noisy-hash metadata;
- transport that metadata to the reset request and `FrontRESSegmentStorageBatch`,
  while explicitly keeping it out of the current PPO adapter.

Semantic owner and transport owners:

| Object | Owner | Consumer-only transport |
| --- | --- | --- |
| old-policy snapshot and transaction binding | `frontres_segment_live_sampler.py` | batch attributes |
| reset metadata | `frontres_segment_live_probe.py` | Clean reset request |
| stored identity tuple | `frontres_segment_storage.py` | storage batch; PPO remains non-scope |

Non-scope:

- changing legacy `sample_rollout_rows()` roles or selecting the S1a plan in
  the existing live loop;
- multi-Segment accumulation, optimizer suppression/step count, PPO metadata
  consumption, grouped loss, checkpoint/resume, or live execution;
- any new Noisy tape, Clean actor reference, Noisy physical prefix, label, or
  perturbation-timing input.

Expected evidence:

- S1 T-snapshot/T-mutation: a real `torch.nn.Module` state fingerprint changes
  under mutation and verification fails;
- S1 T-meta/T-reset/T-storage: one all-policy plan, one snapshot ID, and the
  same row metadata reach batch, Clean-reset request, and storage batch;
- S1 T-no-mixed-reference/T-legacy-isolation: a mismatched plan/snapshot/hash
  fails, and the legacy policy/search route is unchanged.

Completion evidence:

- a real `torch.nn.Linear` fingerprint is bound before materialization; a
  caller-declared foreign snapshot ID and an in-place parameter mutation both
  fail closed;
- one all-policy plan's transaction/snapshot/motion/segment/source/trial/K/Noisy
  hash metadata reaches the batch, Clean index-reset request, and storage batch;
- the storage carrier is intentionally absent from `to_ppo_batch`; S2 must
  still supply collection/accumulation semantics before any optimizer call.

Stop condition:

- stop if a genuine proof requires an actor clone or changes policy sampling,
  a new Noisy materialization, a PPO adapter change, an optimizer call, or a
  simulator run; those belong to S1b follow-up, S2, or Step 3.

#### Step 2-S2: Complete-Transaction Accumulator And Exact-One Update

Status: completed offline. This is a transaction-lifecycle boundary, not a
grouped-PPO implementation or formal live-route activation.

Objective:

- treat the one S1b-sealed storage batch `[B]`, where `B = sum_s M_s`, as one
  atomic frozen-policy transaction only when it covers `S >= 2` distinct
  Segments and `M_s >= 2` all-policy attempts for every source;
- prohibit an optimizer step while that complete transaction is being accepted,
  then permit one explicit update callback and prove its optimizer-step delta is
  exactly one.

Scope:

- add an offline transaction accumulator at the live-probe/storage boundary;
- consume, validate, and retain the existing S1b
  `FrontRESFrozenPolicyTransactionMetadata` object without rematerializing a
  tape, resampling a Segment, or reconstructing identity from row order;
- make collection fail closed on incomplete Segment/M coverage, non-policy
  roles, a changed old-policy fingerprint, changed Noisy identity, an early
  optimizer step, or a repeated finalization;
- hand the retained storage batch to one injected update callback only after
  collection closes, with an injected optimizer-step counter for deterministic
  S2 proof.

Owner files/modules:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`:
  transaction accumulator and exact-one-update gate;
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`:
  unchanged S1b storage carrier consumed as the atomic input;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`:
  deterministic transaction-lifecycle contract.

Non-scope:

- changing `run_frontres_segment_sampler_step()` or the legacy
  `frontres_segment_live_update_loop.py` immediate-update route;
- adding group metadata to `to_ppo_batch()`, grouped loss/reduction,
  checkpoint/resume, new config switches, live simulation, or long training;
- any new Noisy tape/materialization, Clean actor reference, Noisy physical
  prefix, perturbation label, or perturbation-timing input.

Expected evidence:

- S2 T-transaction-complete: one S1b metadata object with two distinct
  Segment groups and at least two policy rows per source is accepted as the
  sole transaction input;
- S2 T-no-early-step/T-no-mixed-reference: collection leaves the injected
  optimizer-step counter unchanged and rejects a mutated policy, mismatched
  storage identity, mixed Noisy hash, or non-policy attempt;
- S2 T-exact-one-update: closing the transaction calls the update callback once,
  increments the observed optimizer-step counter by exactly one, and forbids a
  second finalization.

Completion evidence:

- `FrontRESFrozenPolicyTransactionAccumulator` accepts exactly one complete
  S1b storage carrier, validates `S >= 2`, source-local `M_s >= 2`, all-policy
  roles, old-policy identity, and source-local Noisy hash/scenario identity;
- the deterministic live-probe contract uses an actual `torch.optim.SGD` on
  the fixture policy: collection performs zero calls to `step()`, finalization
  performs one, and zero/two-step callbacks fail closed;
- the real S1b sampler fixture reaches this accumulator with its sealed
  metadata. Focused contracts, source-linked Atlas contracts, and the 59-test
  aggregate suite pass. See `evidence_ledger_v013_future_context_2026-07-19.md`
  E-TX-7 through E-TX-9.

Stop condition:

- stop if transaction assembly requires a second independently materialized
  batch, modifying the S1b tape/metadata, passing identity into the current PPO
  adapter, changing the loss reduction, entering the legacy update loop, or a
  simulator run. Those are Step 3, Step 4, or live-sentinel work.

### Step 3 / 5: Grouped PPO Reduction And Loss Isolation

Objective:

- rebase the candidate reduction to v003: one PPO policy row per eligible
  attempt, with K only as return/evidence horizon, while retaining
  sign-preserving grouped scale and equal motion/Segment/attempt mass.

Scope:

- aligned motion/segment/trial policy-row metadata plus separate K-evidence
  count consumption;
- grouped_scale_only denominator and nested reduction;
- explicit loss-multiplier isolation and mass-share diagnostics.

Owner files/modules:

- source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py;
- source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py;
- focused Segment PPO algorithm and storage contract tests.

Non-scope:

- mean-centering default, focal advantage power, replay-priority weighting,
  new confidence/acceptance variables, or a changed paired Gain.

Expected evidence:

- S1 T-row/T-value/T-sign/T-permute/T-metamorphic/T-source;
- S2 T-connect from transaction storage to actor loss and one update.

Stop condition:

- a K execution step becomes an extra PPO row, a missing group field falls back
  to flat mean, a positive advantage flips sign, or priority/Gain/M/K/evidence
  count/focal terms multiply the active actor loss.

Status after K-A confirmation (2026-07-19):

- v002 passed only as an offline candidate whose final hierarchy was
  motion -> Segment -> attempt -> valid K-step row;
- K-A makes that final level semantically invalid for the active route. The
  old candidate tests are retained as historical loss-math evidence, not as
  acceptance evidence for v003;
- S0 white-box audit code-confirms that live storage already has the required
  physical representation: one first-step old-policy tuple plus K-step
  accumulated return evidence;
- Step 3-KA is a narrow S1 rebase of candidate storage, reducer, diagnostics,
  and tests. It must not enter the formal runner, checkpoint/resume, or a live
  simulator.

Required rebase evidence:

- a K-A row fixture with unequal motion/Segment/M and unequal K, where each
  eligible attempt has exactly one policy row;
- row permutation and sign/non-amplification over those policy rows;
- a metamorphic proof that extra/duplicated K evidence cannot add policy mass;
- fail-closed rejection of partial/missing policy-row metadata;
- static isolation of priority, Gain, M, K, and evidence-step count from actor
  loss multiplication.

Remaining gate:

- only after the v003 S1 rebase passes may Step 4-S2 select
  `grouped_scale_only` at a transaction-complete, exact-one-update boundary;
- the legacy `to_ppo_batch()` caller remains isolated. It must not become a
  formal route by default.

### Step 4 / 5: Formal-Route, Resume, And Diagnostic Integration

Objective:

- after the v003 S1 rebase, make the active v014 route the only formal Stage 3
  route and make legacy interface/transaction behavior fail closed.

Scope:

- Stage 3 configuration and runner wiring;
- warmup/checkpoint/resume versioning;
- transaction, provenance, H/K, and group-mass diagnostics;
- retirement/isolation coverage for the old first-policy/immediate-update path.

Owner files/modules:

- scripts/rsl_rl/train.py;
- source/rsl_rl/rsl_rl/runners/on_policy_runner.py;
- source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py;
- checkpoint, diagnostics, and formal-route contract owners.

Non-scope:

- changed deployment task, generic evaluator rewrites, policy-quality claim, or
  long-running optimization.

Expected evidence:

- S0 T-config/T-route/T-retirement (completed by Step 4-S0 audit);
- S1 T-K-A-row/T-storage/T-reducer (blocked pending Step 3-KA);
- S2 T-formal-connect/T-resume/T-diagnostic;
- S3 T-checkpoint/T-version.

Stop condition:

- formal training can choose a legacy actor layout or credit path, a K execution
  step becomes a PPO row, checkpoint resume loses future-layout/transaction
  identity, or diagnostics omit one required provenance/mass field.

### Step 5 / 5: Minimal Live Identity Sentinel

Objective:

- verify one real simulator transaction's reset, fixed Noisy context, frozen
  old-policy M attempts, one grouped update, and diagnostic identities.

Scope:

- minimum environment count and one bounded transaction only;
- artifact/log capture sufficient for the active checklist.

Non-scope:

- performance comparison, policy-quality conclusion, long training, curriculum
  tuning, new perturbation families, or historical accumulated-state recovery.

Expected evidence:

- S4 T-live/T-state/T-provenance/T-order/T-mass/T-frozen.

Stop condition:

- any reference hash, policy snapshot, reset state, optimizer count, or
  grouped-mass diagnostic differs from the active contract.

## Execution Rule

K-A semantic closure is documented. Before any code work, obtain explicit
approval for the narrow Step 3-KA rebase, then record its Step End Report,
evidence ledger, and checklist update. Do not enter formal-route wiring,
checkpoint/resume, or live execution automatically.
