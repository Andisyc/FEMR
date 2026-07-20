# FRS-v013 Future-Context Evidence Ledger

## Step 1A: Current Reference Provenance

Date: 2026-07-19

Scope:

- determine whether the current Segment Replay reference window can truthfully
  serve as actor-visible fixed Noisy future context;
- do not change production observation, command, sampler, or PPO behavior.

Evidence:

- E-FC-1: source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py,
  MultiMotionCommand.command -> _gather_future_by_motion -> reference-window
  override; source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py
  executed with the FrontRES Python environment.
- E-FC-2: source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/motion_perturbations.py,
  apply_joint_perturbation; scripts/rsl_rl/train.py RP specialist branch.
- E-FC-3: source/rsl_rl/rsl_rl/frontres/frontres_segment_dataset.py,
  _motion_from_noisy_variant; the same focused contract execution.

Observed command:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python
    source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py

Observed result:

- command.future_path had perturber_calls_after_command=0;
- joint_pos.owner_path had perturber_calls_after_joint_pos=1;
- noisy_variant.reference_adapter had frames=4 and all_frames_equal=True;
- command exited successfully with
  frontres_segment_motion_command_reference_contract: ok.

Facts:

- MultiMotionCommand.command gathers its joint reference through
  _gather_future_by_motion. That path does not invoke apply_joint_perturbation.
- RP specialist setup disables joint noise and enables root roll/pitch
  corruption. A raw joint command therefore does not encode the active RP
  artifact.
- The cache adapter builds reference from one noisy_state and repeats it across
  horizon_k + 1 frames. It is a static state payload, not a time-indexed future
  trajectory.

Decision:

- Do not wire the current command/reference_window payload into the actor as
  Future Noisy Context.
- The user confirmed that the replacement must sample one complete Noisy
  sequence at Segment selection and reuse that immutable sequence for all M
  attempts; see E-FC-4.

Open risks:

- The exact provenance of stored noisy_state is not enough to repair the
  missing temporal trajectory; even a genuinely noisy first state is repeated.
- A time-indexed joint-only window still cannot encode local-RP root/anchor
  corruption when joint noise is disabled.
- No live simulator conclusion is made by these deterministic probes.

Next:

- implement and prove the selected-scenario lifecycle before formal-route
  integration: one fixed sequence must reach current/future actor context and
  K-step execution unchanged across all M attempts.

## Step 1B: Fixed Noisy Segment Lifecycle Decision

Date: 2026-07-19

Scope:

- record the accepted lifecycle boundary for the fixed Noisy reference used by
  Future Context and Double Segment Replay;
- do not claim a code, integration, or live-runtime implementation.

Evidence:

- E-FC-4: user decision in this conversation: "对每段被抽样的动作片段都抽样一组固定的Noisy序列，严格管理其生命周期即可，在M次尝试中就使用该组扰动。"
- E-FC-5: FRS-METHOD-v013 `Fixed Noisy Segment`, which already requires one
  immutable R_tilde_s for all M_s attempts.

Facts:

- Existing source inspection proves the current reference route is not yet a
  verified fixed Noisy future trajectory (E-FC-1 through E-FC-3).
- The active v013 method already owns one fixed R_tilde_s per selected scenario;
  this decision makes its selection-time lifetime explicit rather than adding
  another learned variable.

Decisions:

- Sample one complete Noisy sequence when a Segment scenario is selected,
  before attempt 1.
- Bind it immutably to that scenario and reuse it after every Clean x_t reset
  for all M attempts, actor future context, and K-step execution.
- End its semantic binding after evidence aggregation; an immutable replay
  cache may retain it, but retry-time resampling or mutation is forbidden.

Open risks:

- The exact current owner interface, deterministic rematerialization behavior,
  and actor-window route are not code-confirmed for this decision.

Next:

- implement and prove the S1 lifecycle/provenance contract before any formal
  Stage 3 integration or live sentinel.

## Step 1B-S1: Immutable Scenario Binding

Date: 2026-07-19

Scope:

- implement only the sampler-domain selection-time lifecycle for an injected
  Noisy sequence;
- do not connect it to MultiMotionCommand, actor future input, reset, PPO, or a
  simulator run.

Evidence:

- E-FC-6: source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py,
  `FrontRESFixedNoisyScenarioLifecycle.bind_rows`; source/rsl_rl/rsl_rl/tests/
  frontres_fixed_noisy_segment_lifecycle_contract.py.

Observed command:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python
    source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_segment_lifecycle_contract.py

Observed result:

- passed with two base scenarios and five expanded rows;
- source 0 rows 0/1/2 shared one identity and hash, source 1 rows 3/4 shared
  another identity and hash;
- required coverage was [5, 5, 5, 4, 4] for the tested K/H layouts;
- external mutation, insufficient K + H coverage, Clean provenance payload,
  conflicting source K identity, and closed-scenario rematerialization were
  rejected by the deterministic contract.

Facts:

- the lifecycle owner creates at most one immutable sequence per
  `(transaction_id, source_index, segment_id)` identity and reuses it for all
  rows sharing that source index;
- the sealed sequence is cloned on ingestion and returned by copy, so retry
  consumers cannot mutate the stored value;
- closed scenarios retain immutable evidence but reject a new materialization
  under the same identity.

Limitations:

- the materializer is injected by the local test. This proves lifecycle
  semantics, not actual MotionPerturber or MultiMotionCommand behavior;
- actor current/H windows, GMT K-step reference execution, Clean x_t reset, and
  formal transaction credit remain unconfirmed.

Next:

- stop at the S1 boundary and obtain review before Step 1B-S2 command-tape and
  actor-route wiring.

## Step 1B-S2: Approved Carrier And Connectivity Boundary

Date: 2026-07-19

Decision:

- Dr. Cheng approved the command-owned carrier definition
  `[q(29), dq(29), anchor_pos(3), anchor_quat(4)]`, hence `[L, 65]` with
  `L >= K + max(H)`;
- `MultiMotionCommand` is the only runtime materializer/cursor owner; the S1
  lifecycle remains the once-per-source semantic binding owner;
- `frontres_future_offsets` is required and nonempty, with no implicit runtime
  default; its exact run value remains a later explicit configuration choice.

Scope:

- selection -> sealed carrier -> role-aware reset -> current/H actor route ->
  K-step command execution, proven only by deterministic S1/S2 contracts.

Non-scope:

- Noisy physical prefix, Clean actor future, perturbation labels/timing,
  checkpoint/HSL migration, transaction/PPO changes, and live execution.

Pre-implementation evidence:

- E-FC-7: white-box audit of `MultiMotionCommand`: legacy reference window is
  `[B,W,29|58]`, while current raw anchor caches are 3D/4D and are separately
  refreshed by `MotionPerturber`; therefore the legacy window cannot carry
  active local-RP future reference semantics.

Acceptance stop:

- stop immediately if the connectivity implementation requires a Clean future,
  an H/K conflation, a new physical state prefix, or a transaction/PPO change.

## Step 1B-S2: Offline Connectivity Result

Date: 2026-07-19

Scope completed:

- selection -> command-owned `[L,65]` tape -> role-aware Clean `x_t` reset ->
  tape-backed current reference/H reads/K execution -> actor layout guard;
- deterministic S1/S2 evidence only. No simulator launch, checkpoint/HSL
  migration, Double Segment transaction, or grouped PPO change was made.

Evidence:

- E-FC-8: `MultiMotionCommand.materialize_frontres_fixed_noisy_tape`,
  `set_frontres_fixed_noisy_tape`, cursor/H reader, and fixed-cache refresh;
  `frontres_segment_live_sampler.py`, `frontres_segment_stage1_env_hooks.py`,
  `frontres_segment_live_probe.py`, and `frontres_runtime.py`.
- E-FC-9: focused deterministic contracts:

      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_segment_lifecycle_contract.py
      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_motion_command_reference_contract.py
      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage1_env_hooks_contract.py
      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_fixed_noisy_actor_context_contract.py
      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py

- E-FC-10: aggregate offline regression:

      /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py

Observed result:

- all focused contracts above passed; the aggregate suite reported
  `contract_count=59`, `failed_count=0`, and
  `frontres_segment_all_contract_suite: ok`;
- one selected source invokes the materializer once, fans the identical tape and
  `noisy_segment_hash` to its M rows, and closes the scenario after evidence
  aggregation; an outer `finally` also closes it when rollout fails before
  evidence can be built;
- role-aware reset installs the same tape on all quartet rows without calling
  the live perturber; `policy/candidate/noisy` execute the tape, while `clean`
  restores the Clean dynamic/reference baseline but retains only the same
  Noisy actor-context tape;
- H reads are ordered, read-only `[B, |H|*65]` views; the command cursor alone
  advances once per K execution step. Current `q/dq` command and raw anchor
  cache read the same sealed tape;
- actor routing prepends only the H reads to the existing observation, rejects a
  missing selected scenario, legacy actor dimensions, and mismatched prefix
  normalizer statistics.

Limits retained:

- this is not S3 checkpoint/HSL evidence: a legacy 870D artifact is not
  accepted as a v013 future-layout artifact, and no new artifact was produced;
- this is not S4 evidence: no live simulator, physical trajectory, or training
  conclusion was run;
- Step 2 Double Segment transaction and Step 3 grouped PPO remain untouched.

Step End:

- Step 1B-S2 is closed at S1/S2 offline connectivity. The next authorized
  decision is whether to begin Step 2; do not start it automatically.

## Step 2-S1a: Declarative All-Policy Attempt Plan

Date: 2026-07-19

Scope completed:

- add a pure sampler-domain plan for multiple selected Segment groups with
  `M_s >= 2` policy attempts each;
- retain caller-declared `transaction_id` and `policy_snapshot_id`, source/trial
  order, and per-Segment K without mutating sampler state.

Evidence:

- E-TX-1: `source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py`,
  `FrontRESFrozenPolicyTransactionPlan` and
  `FrontRESSegmentSampler.plan_frozen_policy_transaction`.
- E-TX-2: deterministic test-first sampler contract in
  `source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py`.
- E-TX-3: aggregate offline regression and source-linked Atlas checks.

Observed commands:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_segment_sampler.py source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_formal_runtime_audit_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_policy_quality_atlas_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py

Observed result:

- the focused plan test passes for selected Segment groups `[1, 2]` with
  attempt counts `[3, 6]`: every one of the nine rows is `policy`, ordered by
  source then trial, with K `[16, 32]` retained per source;
- a second fixture upgrades base one-attempt segments `[0, 4]` to `[2, 2]` and
  rejects a single selected segment, duplicate groups, and empty transaction or
  snapshot identities;
- `py_compile`, both source-link contracts, and the aggregate suite passed;
  the latter reported `contract_count=59`, `failed_count=0`, and
  `frontres_segment_all_contract_suite: ok`.

Facts:

- the legacy `expand_rollout_trials` path remains unchanged: it still exposes
  `policy` then `search` evidence rows and therefore is not the formal v013
  route;
- the new planner is side-effect free with respect to sampler priority,
  staleness, and seen state; it has no runner, environment, tape, storage, or
  optimizer reference;
- `policy_snapshot_id` is only a caller-declared identity at this boundary. It
  is not evidence that an actual parameter snapshot has been captured or held
  fixed.

Limits retained:

- Step 2-S1b must bind that plan to a real frozen policy snapshot and propagate
  transaction/snapshot/motion/segment/trial/noisy-hash metadata through Clean
  reset and storage;
- Step 2-S2 must collect all selected groups before exactly one optimizer call;
- no Noisy tape lifecycle, Clean `x_t` semantics, actor route, PPO reduction,
  or live simulator path changed in this step.

Step End:

- Step 2-S1a is closed as an offline sampler-plan boundary. The next decision
  is whether to authorize Step 2-S1b; do not enter it automatically.

## Step 2-S1b: Frozen Snapshot And Identity Carrier

Date: 2026-07-19

Scope completed:

- capture an identity-only, deterministic fingerprint of the real
  `runner.alg.policy.state_dict()` before fixed Noisy materialization;
- bind the S1a all-policy plan to that derived snapshot identity, then reuse
  its transaction ID for one fixed Noisy lifecycle;
- carry the sealed transaction/snapshot/motion/start/segment/source/trial/K/
  role/Noisy-hash metadata through the selected batch, Clean index-reset
  request, and independent storage batch;
- at the S1b close, keep that metadata outside the then-current PPO adapter;
  later Step 3 owns any explicit loss-side consumption.

Evidence:

- E-TX-4: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py`,
  `FrontRESFrozenPolicySnapshot`,
  `capture_frontres_frozen_policy_snapshot`,
  `bind_frontres_frozen_policy_transaction`, and
  `finalize_frontres_frozen_policy_transaction_metadata`.
- E-TX-5: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py` and
  `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`: sealed carrier
  validation before Clean reset and policy verification before storage, with
  `FrontRESSegmentStorageBatch.transaction_metadata` intentionally absent from
  `to_ppo_batch`.
- E-TX-6: deterministic extensions of
  `frontres_segment_live_sampler_contract.py` and
  `frontres_segment_live_probe_contract.py`; source-link Atlas regression
  checks and the aggregate suite.

Observed commands:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_sampler.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_formal_runtime_audit_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_policy_quality_atlas_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py

Observed result:

- the focused sampler contract uses a real `torch.nn.Linear`: its snapshot ID
  is derived from a deterministic state hash, a foreign caller-supplied ID is
  rejected, and an in-place parameter mutation is rejected before storage;
- one all-policy two-Segment plan materializes exactly one tape per source and
  produces one transaction ID with source-local hash reuse across both trials;
- the focused probe contract proves the same metadata object reaches the Clean
  reset request and storage batch, calls its policy verifier before storage,
  rejects a mixed fixed-Noisy hash before reset, and confirms the PPO batch has
  no transaction metadata;
- syntax, focused storage, both Atlas contracts, and the aggregate suite pass;
  the aggregate result is `contract_count=59`, `failed_count=0`.

Facts:

- `Clean x_t` remains only the reset dynamics input. The S1b carrier contains
  provenance identities, not a Clean actor reference or a Noisy physical
  prefix.
- A pre-bound transaction reuses its ID in `_attach_fixed_noisy_scenarios`; the
  lifecycle remains the only source that materializes the tape, and reset does
  not resample or alter it.
- `FrontRESSegmentStorageBatch` retained the opaque immutable carrier at the
  S1b close. Step 3 later adds its own explicit, row-aligned candidate adapter;
  S1b itself did not authorize PPO consumption.

Limits retained:

- the existing legacy live loop still selects `policy` then `search` rows and
  can update immediately; the S1b carrier does not activate, replace, or
  relabel that path;
- S2 may consume the sealed full storage carrier offline, but does not itself
  wire it into grouped PPO, checkpoint/resume, or a simulator; Step 3 owns the
  separately bounded storage-to-loss adapter;
- actual M-attempt Clean reset repetition remains a separate reset/runtime
  proof, not an inference from the S1b/S2 offline carriers.

Step End:

- Step 2-S1b is closed at S1/S2 offline identity connectivity. Step 2-S2
  records the next bounded transaction gate below.

## Step 2-S2: Complete-Transaction Accumulator And Exact-One Update

Date: 2026-07-19

Scope completed:

- add a candidate-only accumulator at the live-probe/storage boundary;
- accept only one complete S1b storage carrier whose metadata covers at least
  two distinct Segment groups and at least two all-policy attempts per source;
- preserve the old-policy/noisy identity checks during collection, require zero
  observed optimizer steps before finalization, then require one observed step
  from one injected update callback;
- retain isolation from `to_ppo_batch()`, grouped PPO, legacy live-update loop,
  checkpoint/resume, and simulator execution.

Evidence:

- E-TX-7: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`,
  `FrontRESFrozenPolicyTransactionAccumulator` and
  `FrontRESFrozenPolicyTransactionResult`. The accumulator reads the S1b
  carrier only, validates source/Segment/M/role/hash identity, and does not
  call the current PPO adapter or loss owner.
- E-TX-8: `frontres_segment_live_probe_contract.py` starts from a failing
  missing-API test, then proves zero collection steps and exactly one actual
  `torch.optim.SGD.step()` on finalization; it also rejects an early step, a
  two-step callback, a non-policy attempt, a mixed source-local Noisy identity,
  and repeat finalization.
- E-TX-9: `frontres_segment_live_sampler_contract.py` passes the real S1b
  `FrontRESFrozenPolicyTransactionMetadata` and storage batch through the S2
  accumulator; the aggregate suite and both source-linked Atlas contracts
  remain green.

Observed commands:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_formal_runtime_audit_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_policy_quality_atlas_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py

Observed result:

- the initial S2 contract failed with the expected missing
  `FrontRESFrozenPolicyTransactionAccumulator` API, then passed after the
  minimal owner implementation;
- no optimizer callback is reachable while `append_storage_batch()` validates
  the sealed S1b carrier; an observed early count change fails closed;
- finalization invokes the supplied callback once and accepts only an observed
  step delta of one. The fixture uses a real SGD `step()` on the fixture policy;
  a zero/two-step callback fails and terminal state prevents a second update;
- actual S1b metadata reaches the same accumulator in the sampler contract;
  formal-runtime and policy-quality Atlas links pass, and the aggregate result
  is `contract_count=59`, `failed_count=0`.

Facts:

- S2 treats the complete S1b `[B]` carrier as one transaction; it does not
  combine separately sampled scenarios or alter the immutable tape.
- The accumulator itself contains no `to_ppo_batch()`,
  `compute_frontres_segment_ppo_loss`, or legacy single-update call. Its one
  callback is an explicit future integration seam, not a hidden PPO route.
- All policy attempts remain coverage rows even if ordinary `valid_mask` later
  excludes some rows; S2 does not choose a best attempt or alter loss mass.

Limits retained:

- the existing `run_frontres_segment_single_update()` trust-region retry loop
  may issue more than one physical `optimizer.step()` when it is invoked
  directly. S2 deliberately does not wire that legacy function; Step 3/4 must
  resolve its transaction-compatible update policy before formal routing;
- S2 does not execute M Clean resets, materialize a tape, enter grouped PPO,
  enable checkpoint/resume, or run a simulator/live job.

Step End:

- At the Step 2-S2 close, Step 3 was the next gated owner: grouped PPO metadata
  consumption and loss isolation.

## Step 3: Grouped PPO Reduction And Metadata Loss Consumption

Date: 2026-07-19

Scope completed:

- add candidate-only `grouped_scale_only` to the Segment PPO loss owner;
- consume sealed transaction metadata and explicit storage row indices at the
  storage-to-PPO adapter boundary;
- reduce actor, value, and entropy rows by equal motion -> Segment -> attempt
  -> valid-step mass, with no loss weight from priority, raw Gain, M, K, or a
  focal power;
- apply detached, sign-preserving `max(r_gs, r_txn)` advantage denominators;
- fail closed for missing, mismatched, or partial transaction metadata instead
  of treating a minibatch as an independent complete transaction.

Evidence:

- E-PPO-1: `frontres_segment_ppo.py` adds the sealed transaction row reader,
  hierarchical reducer, `grouped_scale_only` construction, and structured mass
  diagnostics on `FrontRESSegmentPPOResult`.
- E-PPO-2: `frontres_segment_grouped_ppo_contract.py` first failed because
  `FrontRESSegmentPPOBatch` had no transaction-metadata input, then passed a
  hand-computed two-motion/three-Segment/unequal-M/unequal-K fixture,
  permutation, sign/non-amplification, duplicate-step/equivalent-attempt mass
  invariance, missing/misaligned metadata, and partial-transaction rejection.
- E-PPO-3: the focused contract statically inspects the reducer and confirms
  no priority, Gain, focal, horizon, or trial-count term can multiply a row
  loss. It also proves the candidate adapter is absent from the legacy runner.
  Its reported valid-step masses sum to one.
- E-PPO-4: `frontres_segment_storage.py` preserves the opaque carrier and
  explicit row mapping only through `to_grouped_ppo_candidate_batch`; updated
  S1b/S2 fake-path tests verify the same object and `[0..B-1]` mapping at the
  candidate PPO batch while legacy `to_ppo_batch` remains metadata-free.

Observed commands:

    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py source/rsl_rl/rsl_rl/tests/frontres_segment_grouped_ppo_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_grouped_ppo_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_formal_runtime_audit_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_policy_quality_atlas_contract.py
    /Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py

Observed result:

- the initial grouped contract failed for the expected missing
  `transaction_metadata` batch API, then passed after the minimal algorithm and
  storage owner changes;
- the golden fixture reports `actor_loss=-0.382148862`,
  `expected=-0.382148870`, `motions=2`, `segments=3`, `attempts=4`,
  `valid_steps=7`, `step_mass_sum=1`, and `sign_flips=0`;
- scale evidence reports `r_txn=7.106335163`, low-segment RMS `1.0` with scale
  `7.106335163`, and high-segment RMS/scale `10.0`, so the low group is not
  amplified and no sign flips occur;
- the M/K metamorphic probe reports constant Segment mass `0.25`, source
  attempt mass `0.25`, duplicated valid-step mass `1/24`, and equivalent-attempt
  mass `1/12`;
- py_compile, formal-runtime Atlas, policy-quality Atlas, and the full Segment
  contract suite pass. The aggregate result is `contract_count=60`,
  `failed_count=0`, `total_marker_count=60`.

Facts:

- the old `scale_only` caller configuration remains unchanged, so this new
  reducer is not selected by the legacy runner;
- transaction-complete means every sealed metadata row appears exactly once,
  while ordinary invalid rows remain in the carrier but are excluded before
  group means are formed;
- Clean x_t, Noisy tape provenance, replay priority, and optimizer accounting
  retain their existing owners. This step neither resamples a tape nor calls an
  optimizer.

Limits retained:

- no formal runner, checkpoint/resume contract, grouped optimizer update, or
  simulator/live run was added;
- a formal minibatch route is intentionally unsupported: it must first define
  transaction-global group weights or use a complete transaction batch;
- the current live storage captures one first-step PPO tuple with a K-step
  accumulated return. It does not yet prove that one real attempt emits several
  stored valid PPO rows, so the offline valid-step fixture is loss-math evidence,
  not a live K-row claim;
- the legacy `run_frontres_segment_single_update()` remains unmodified and may
  still use its own trust-retry behavior. Step 4 must not route this candidate
  loss through it without an exact-one-update redesign.

Step End:

- Step 3 is closed at S1 plus offline storage-to-loss connectivity only. Step
  4 remains the next gated owner for formal configuration, update ordering,
  resume, and live diagnostics.

## Step 4-S1: K-A Semantic Closure

Date: 2026-07-19

Human-confirmed decision:

    one policy-sampled attempt
    -> one PPO policy row
    -> K-step executable-evidence return/advantage

K is not a sequence of PPO policy rows. H remains actor-visible future context;
Clean x_t remains only the repeatable dynamics reset; the fixed Noisy tape
remains unchanged across all M attempts.

Code-confirmed S0 fact:

- the current live storage captures one first-step old-policy tuple plus a
  K-step accumulated return/advantage. This is the physical K-A representation;
  it is not proof of formal transaction connectivity.

Contract consequence:

- FRS-METHOD-v014, FRS-TRAIN-v005, and FRS-PPO-v003 supersede the v013/v004/v002
  active set;
- the v002 offline candidate formula and tests used a final valid-K-step-row
  reduction. They are historical loss-math evidence only and cannot certify the
  active v003 single-policy-row contract;
- Step 3-KA is therefore reopened as a narrow candidate storage/reducer/test S1
  rebase. Step 1B tape provenance and Step 2 frozen-policy transaction evidence
  are retained unchanged.

No code, test, simulator, optimizer, checkpoint, or live run was executed for
this semantic closure. Formal route, checkpoint/resume, and live work remain
blocked pending explicit approval of Step 3-KA and its S1 evidence.
