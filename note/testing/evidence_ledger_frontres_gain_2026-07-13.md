# Evidence Ledger: FRS-GAIN-v001 Integration

Date: 2026-07-13
Scope: Stage 3 Segment Replay paired Gain implementation and formal policy-row
return connectivity.
Contract: `note/frontres_core/contracts/active/reward/FRS-GAIN-v001-style-physics-repair.md`

## E1: Shared Gain Owner

Evidence:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
- `source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`

Facts:
- Style compares Repaired and Noisy body positions against the same Clean
  positions.
- Physics has paired success/fall, survival, ZMP/support margin, and a
  documented foot-height contact-consistency proxy.
- Repair cost uses full-6D action magnitude and temporal action change.
- Missing root orientation, ZMP/support, contact, or unavailable temporal data
  remains NaN and is formatted as `UNCONFIRMED`; it is not silently zeroed.

Observed result:
- `frontres_gain_components_contract.py`: PASS.

Limitation:
- Real IsaacLab population of the captured runtime fields remains an S4
  question. Contact consistency is a foot-height support proxy, not force
  sensor data.

## E2: Formal Policy-Row Connectivity

Evidence:
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`

Facts:
- Live capture stores per-step full-6D actions.
- Paired policy rows use shared `gain_total` in `storage.rewards`.
- K-step return construction uses per-step Gain evidence for policy rows.
- Legacy `repair_score_accum` and generic environment reward are not used for
  those formal policy rows when Gain evidence is present.

Observed result:
- `frontres_segment_gain_connectivity_contract.py`: PASS.
- Existing `frontres_segment_live_probe_contract.py`: PASS.

## E3: Configuration And Diagnostics

Evidence:
- `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`

Facts:
- Gain weights and named scales are explicit configuration fields.
- Live probe exposes a decomposed Gain block with source, Style, Physics,
  repair cost, total, and component diagnostics.
- Missing component values are rendered as `UNCONFIRMED`.

Open risks:
- Sampler priority still consumes the legacy score summary.
- Periodic and sequence evaluation still need formula identity with the shared
  Gain owner.
- S4 live evidence is not yet available.

Next:
- Revalidate paired K-step aggregation with the accepted component and action
  mask semantics, then connect sampler evidence, periodic eval, and sequence
  eval to `frontres_gain.py`.

## E4: Step 1 Style Completion

Evidence:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_capture_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`

Facts:
- Root orientation uses Clean/Repaired/Noisy root quaternion roles and a
  geodesic error difference against Clean.
- Capture reads `anchor_quat_w_original` for Clean and paired
  `robot_anchor_quat_w` rows for Repaired/Noisy.
- The full style fixture, root-role capture fixture, live probe regression, and
  aggregate suite pass.

Observed result:
- Gain S1: PASS.
- Root orientation capture S1: PASS.
- Live probe regression: PASS.
- Aggregate suite: `44/44`, failed `0`.

Limitation:
- This proves deterministic code and offline route wiring. It does not prove
  that a real IsaacLab run populates finite root-orientation values.

Decision:
- Step 1 / 11 is complete. Step 2 / 11 is complete for S1/S2 offline
  implementation and connection. Step 3 / 11 is complete offline. Step 4 /
  11 is next and has not started.

## E5: Step 2 Physics Completion

Evidence:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_reward_window.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_capture_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`

Facts:
- Physics Gain averages paired success, survival, ZMP/support margin, and
  contact consistency without importing generic environment reward.
- ZMP uses the existing balance/capture-margin owner for train and baseline
  rows with the same quartet offsets.
- Contact consistency compares Clean foot support state with repaired/noisy
  robot foot support state. It is a height-based support proxy, not a contact
  force measurement.
- Missing Physics evidence remains NaN and is rendered as `UNCONFIRMED`.

Observed result:
- Gain component contract: PASS.
- Motion-quality/contact pairing contract: PASS.
- Gain connectivity and live-probe contracts: PASS.
- Aggregate suite: `44/44`, failed `0`.

Limitation:
- No IsaacLab S4 sentinel was run in this step. Real contact/ZMP population
  and metric diversity remain unconfirmed until a live run.

Decision:
- Step 2 / 11 is complete for S1/S2 offline implementation and connection.
- Step 3 / 11 is next and has not been started.

## E6: Step 3 Repair Cost Completion

Evidence:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`

Facts:
- Repair Cost reads the post-override transition action, not the raw policy
  action, so baseline and Clean rows retain their executed zero action.
- The full six Delta SE dimensions are used for magnitude and temporal cost.
- The action-valid mask is `horizon` plus `not done before this step`; the
  action that causes done remains included because it was executed.
- Clean-row norm/temporal/cost are exposed separately and are unavailable when
  no valid Clean action row exists.

Observed result:
- Repair component contract: PASS.
- Executed-action live-probe contract: PASS.
- Gain connectivity contract: PASS.
- Aggregate suite: `44/44`, failed `0`.

Limitation:
- This is a pure tensor and fake-runner boundary. It proves action source,
  shape, masking, and aggregation, but not real simulator action diversity.

Decision:
- Step 3 / 11 is complete for S1/S2 offline implementation and connection.
- Step 4 / 11 is next and has not been started.

## E7: Step 4 Paired K-Step Gain Completion

Evidence:
- `source/rsl_rl/rsl_rl/frontres/frontres_gain.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`

Facts:
- Mixed per-row K and done masks are applied before Repair Cost and temporal
  component aggregation.
- Quartet rows preserve their own Clean/Repaired/Noisy pairing under row
  permutation; the hand fixture checks the resulting `gain_total` by role.
- The total formula uses the same named Style, Physics, and Repair components
  with the configured weights.
- Storage connector tests preserve the per-step Gain trace and do not replace
  it with the legacy environment/RP score.

Observed result:
- Mixed-K/component golden fixture: PASS.
- Segment storage contract: PASS.
- Gain connectivity and live-probe contracts: PASS.
- Aggregate suite: `44/44`, failed `0`.

Limitation:
- This is offline evidence. The real runtime still needs S4 proof that
  per-step component values are finite and non-stale across actual rollouts.

Decision:
- Step 4 / 11 is complete for S1/S2 offline implementation and connection.
- Step 5 / 11 is next and has not been started.

## E8: Step 5 Formal Training Return Completion

Evidence:
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_algorithm_contract.py`
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`

Facts:
- With `gain_config` active, formal policy rows use paired `gain_total` for
  storage rewards and `gain_steps` for K-step returns.
- `compute_returns_and_advantages()` receives the same per-row horizon and done
  trace, then produces returns and advantages consumed by the PPO batch.
- Policy-row validity survives storage and `to_ppo_batch()`; baseline rows are
  retained as evidence rows but are excluded from PPO policy updates.
- If formal Gain evidence is unavailable, the route raises instead of silently
  reusing legacy `repair_score`.

Observed result:
- Formal Gain -> storage -> PPO batch contract: PASS.
- Legacy fallback rejection contract: PASS.
- Storage and PPO contracts: PASS.
- Aggregate suite: `44/44`, failed `0`.

Limitation:
- This proves the offline formal route only. It does not prove a real live
  rollout produces finite Gain traces or that long-run learning improves.

Decision:
- Step 5 / 11 is complete for S1/S2 offline implementation and connection.
- Step 6 / 11 is partial/blocked: formal Gain source selection and
  post-update diagnostic isolation pass, but legacy score consumption remains
  in sampler priority/state logic.

## E9 - Step 6 Sampler Priority Boundary (2026-07-13)

- `build_live_sampler_evidence()` now selects finite
  `gain_total_per_sample` when `gain_source=FRS-GAIN-v001`, even if the
  summary also contains a conflicting legacy `gain_over_noisy_per_sample`.
- The sampler contract test proves that extreme
  `ppo_post_update_distribution_kl_mean`, post-update ratios, and
  `ppo_param_delta_l2` do not change priority when formal Gain evidence is
  unchanged.
- Verification passed:
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
  aggregate `frontres_segment_all_contract_suite.py` reported
  `contract_count=44 failed_count=0 total_marker_count=44`;
  `git diff --check`; Python compile of the changed sampler and contract test.
- Remaining blocker: `frontres_segment_sampler.py` still reads
  `score_noisy/score_repaired` in `_learning_value()` and segment state
  transitions. Therefore Step 6 is partial/blocked, not accepted as a
  Gain-only priority contract. No PPO formula or live route was changed.

## E10 - Single Active Gain Owner Decision (2026-07-13)

- Confirmed design decision: `frontres_gain.py` is the only active owner of
  paired Style/Physics/Repair Gain calculation.
- `gain_total` and its component decomposition must feed PPO reward, sampler
  evidence/priority, diagnostics, periodic evaluation, and sequence evaluation.
- The former family-specific executability score is legacy and must not remain
  active through sampler difficulty heuristics, fallback reward paths,
  diagnostics, or evaluation.
- Step 6 is ready for ordered implementation gates 6A evidence owner, 6B
  sampler state migration, and 6C cross-consumer acceptance. This entry records
  the decision only; no Python implementation claim is made.

## E11 - Step 6A Formal Sampler Evidence Boundary (2026-07-13)

- `build_live_sampler_evidence()` now requires
  `gain_source=FRS-GAIN-v001`, finite per-row total Gain, and finite
  Style/Physics/Repair component vectors with matching row counts.
- Missing or non-finite formal Gain now fails closed; generic environment
  reward and legacy score differences cannot construct active sampler Gain.
- The evidence payload carries canonical total/component fields and retains
  old score fields only as compatibility data for the pending 6B migration.
- The S2 closed-loop fixture supplies one hand-checkable formal row
  (`0.25 + 0.15 - 0.05 = 0.35`) because its fake lifecycle has no simulator
  motion tensors; this is connector evidence, not S4 physics evidence.
- Verification passed:
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_closed_loop_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_pseudo_suite.py`;
  aggregate `frontres_segment_all_contract_suite.py` reported
  `contract_count=44 failed_count=0 total_marker_count=44`;
  Python compile and `git diff --check` passed.
- Confirmed: Step 6A implementation and S2 offline connector boundary.
- Open: `frontres_segment_sampler.py` still consumes compatibility
  `score_noisy/score_repaired` in useful/state logic; Step 6 remains
  partial/blocked until 6B and 6C are executed.

## E12 - Step 6B Gain-Only Sampler Decisions (2026-07-13)

- `frontres_segment_sampler.py` now traces canonical
  `FrontRESSegmentRolloutEvidence.gain_total` through aggregation, useful
  value, state transitions, priority updates, and frontier horizon selection.
- `score_noisy`, `score_repaired`, and `gain_over_noisy` remain compatibility
  fields but are not read by active sampler decisions; aggregate trial score
  fields are retained as unavailable compatibility diagnostics.
- Solved, hopeless, positive, and delayed-regret transitions now use Gain,
  validity/reset, fall, and effective horizon; contact gates useful value.
- A differential fixture changed legacy scores from zero to one while keeping
  canonical Gain fixed; priority and segment state remained identical:
  `[probe sampler_gain_only] clean_priority=[0.04, 0.0]`
  and `poisoned_priority=[0.04, 0.0]`, states `[1, 5]` in both paths.
- Frontier budget selection no longer uses `last_oracle_gap`; it uses active
  success/fall evidence and trial count while preserving K curriculum behavior.
- Verification passed:
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sampler_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
  closed-loop and Stage 3 pseudo suites;
  aggregate `frontres_segment_all_contract_suite.py` reported
  `contract_count=44 failed_count=0 total_marker_count=44`;
  Python compile and `git diff --check` passed.
- Confirmed: Step 6B implementation and S2 offline connectivity.
- Open: Step 6C must audit every currently wired consumer; real sampler
  distribution and simulator behavior remain unconfirmed S4 evidence.

## E13 - Step 6C Cross-Consumer Gain Owner Audit (2026-07-13)

- Audit boundary: `frontres_gain.py` -> live capture summary -> formal PPO
  storage, sampler evidence/priority/state, diagnostics, periodic eval, and
  sequence eval.
- Confirmed formal consumers: `_segment_storage_rewards()` and
  `_segment_storage_reward_steps()` call the shared paired Gain path when the
  active configuration is present; Step 6A/6B sampler evidence and decision
  tests remain passing.
- Blocked consumer: `frontres_segment_live_probe.py` lines 2679-2750 still
  constructs train-effect `score_*` fields from `_paired_score_summary()`;
  `_paired_gain_summary()` is additive and does not replace those fields.
- Blocked consumer: `frontres_segment_live_training.py` lines 1215-1373 use
  `capture.reward_accum` in `_offline_eval_score_summary()` and
  `_offline_eval_per_motion_summary()` for periodic/sequence repaired/noisy
  values and gain, instead of the shared Gain owner.
- Blocked consumer: `frontres_segment_diagnostics.py` lines 60-63,
  90-107, and 180-195 still read/format legacy score fields.
- False-positive test evidence: the current pseudo training and sequence tests
  pass, but their fixtures assert `reward_accum`/`score_*` behavior and
  contain no Gain-owner identity or legacy-score poisoning test for these
  consumers.
- Commands passed: Gain connectivity, sampler, live sampler, live-training
  pseudo, sequence-eval, diagnostics, and aggregate contract suite; aggregate
  reported `contract_count=44 failed_count=0`.
- Conclusion: Step 6C is `partial/blocked`; do not treat the new Gain route as
  cross-consumer accepted. The next safe implementation boundary is to migrate
  diagnostics, periodic eval, and sequence eval in their ordered Step 7-9
  steps, then add owner/isolation tests before live training.

## E14 - Step 7 Canonical Train Diagnostics Migration (2026-07-13)

- Boundary: `capture` -> `_paired_gain_summary` -> live probe summary ->
  update-loop aggregation -> train-effect formatter.
- Changed active diagnostic keys to `gain_style_mean`, `gain_physics_mean`,
  `gain_repair_cost_mean`, `gain_total_mean`, and `gain_total_pos_frac`.
- Removed old `score_noisy_mean`, `score_repaired_mean`, `score_gain_mean`, and
  `score_gain_pos_frac` from the active live summary/update-loop diagnostic
  path. Per-row legacy score vectors remain only as explicitly documented
  sampler compatibility evidence.
- Missing Gain components now render as `UNCONFIRMED` instead of numeric zero.
- Regression evidence: diagnostics poisoning fixture changes legacy score
  fields to extreme values without changing canonical train-effect output;
  live-probe connectivity fixture changes `reward_accum` to `-999` without
  changing `gain_total_mean` and asserts `score_gain_mean` is absent.
- Verification passed:
  `frontres_segment_diagnostics_contract.py`,
  `frontres_segment_live_update_loop_contract.py`,
  `frontres_segment_live_probe_contract.py`,
  `frontres_segment_gain_connectivity_contract.py`,
  `frontres_segment_live_training_pseudo_contract.py`, and aggregate suite
  `contract_count=44 failed_count=0`; Python compile and `git diff --check`
  passed.
- Status: Step 7 is `partial/offline`; raw ZMP/contact diagnostic exposure and
  real S4 population remain open. Step 8/9 periodic/sequence legacy consumers
  remain blocked by E13.

## E15 - Step 8 Periodic Evaluation Gain Migration (2026-07-13)

- Boundary: independent sampler -> eval batch/reset -> paired capture ->
  `_capture_paired_gain` -> periodic summary -> periodic formatter.
- Periodic evaluation no longer calls `_offline_eval_summary`; its accepted
  Gain fields come only from `FRS-GAIN-v001` Style / Physics / Repair / total
  components. `reward_accum` is not read for periodic accepted Gain.
- Missing Gain evidence is represented as `gain_source=UNCONFIRMED` and the
  formatter prints `UNCONFIRMED`, while motion-quality diagnostics remain
  independently visible.
- Regression evidence: periodic pseudo test changes `reward_accum` to extreme
  values while injecting canonical Gain and confirms the reported total is
  unchanged; the all-fall fixture confirms missing Gain is not converted to 0.
- Verification passed:
  `frontres_segment_diagnostics_contract.py`,
  `frontres_segment_live_training_pseudo_contract.py`, aggregate
  `contract_count=44 failed_count=0`, Python compile, and `git diff --check`.
- Status: Step 8 is `partial/offline`; S2 periodic routing is covered, real S4
  component population remains open, and sequence evaluation is intentionally
  deferred to Step 9.

## E16 - Step 9 Sequence Evaluation Gain Migration (2026-07-13)

- Boundary: sequence item capture -> shared Gain owner -> per-motion grouping
  -> sequence aggregate -> item/aggregate/differential logs.
- Sequence evaluation no longer calls a reward-derived score summary. Item and
  per-motion summaries expose `FRS-GAIN-v001` Style / Physics / Repair / total
  fields; aggregate averaging uses finite numeric values only and preserves
  `UNCONFIRMED` when evidence is unavailable.
- The real/zero-policy differential compares canonical `gain_total_mean`; the
  former `reward_accum` is only printed as labeled raw debug data and is not a
  metric source.
- Regression evidence: sequence pseudo captures inject canonical Gain while
  carrying legacy reward values; per-motion Gain remains role-scoped and the
  aggregate preserves component identity. Sequence planner, reset/preroll,
  motion grouping, action visibility, and differential tests remain passing.
- Verification passed: sequence eval contract, live-training pseudo contract,
  aggregate `contract_count=44 failed_count=0`, Python compile, and
  `git diff --check`.
- Status: Step 9 is `partial/offline`; Step 6C cross-consumer S2 acceptance is
  closed, while S4 real Gain population and Step 10 persistence remain open.

## E17 - Step 10A Formal Checkpoint/Resume Audit (2026-07-13)

- Boundary: `OnPolicyRunner.save/load` ->
  `source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py` -> model/std,
  optimizer, normalizer, sampler, and Gain configuration identity.
- Owner finding: `frontres_segment_checkpointing.py` is only imported by
  `frontres_segment_checkpoint_contract.py`; it is not the formal runner save/
  load owner. Earlier S3 evidence was therefore insufficient for the active
  path.
- The formal owner now persists `frontres_gain_config` with contract id
  `FRS-GAIN-v001` and the named Style/Physics/Repair scales.
- Full FrontRES resume rejects a checkpoint with missing or mismatched Gain
  identity; explicit non-full Stage 2 -> Stage 3 initialization is allowed and
  emits a warning because it intentionally uses the current Stage 3 config.
- Regression probes:
  `[probe step10a] gain_config_mismatch_rejected` and
  `[probe step10a] missing_gain_config_rejected`.

Correction recorded by Step 10C-C1: the detached
`frontres_segment_checkpointing.py` helper and its focused two-head migration
test were subsequently deleted; the commands below are historical E17 evidence,
not current runnable paths. Current checkpoint evidence is the formal
`frontres_checkpointing.py` contract exercised through the active Segment suite.
- Verification passed:
  `frontres/bin/python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py`;
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`;
  aggregate `contract_count=44 failed_count=0 total_marker_count=44`;
  `git diff --check`.
- Status: Step 10A is `partial/offline`; formal S3 contract coverage is
  confirmed, while actual server artifact resume, S4 runtime persistence, and
  Step 11 live acceptance remain open.

## E18 - Current Note State Reconciliation (2026-07-13)

- Reconciled current views after the E14/E15/E16 migrations and E17 checkpoint
  audit.
- Current status: Step 6A/6B/6C is `completed/offline` for S2; Steps 7/8/9 are
  `partial/offline` because real S4 population remains open; Step 10A is
  `partial/offline`; Step 11 is not started.
- E13 remains unchanged as historical evidence of the pre-migration blocked
  state and is not a current blocker.
- Updated current plan, checklist, control board, and Architecture view to
  reference E14/E15/E16/E17 instead of treating E13 as active state.

## E19 - Method-Code Alignment Audit Baseline (2026-07-13)

- Audit boundary: active contracts -> Stage 3 config/entrypoint -> perturbation
  and K curriculum -> full-6D actor/action -> rollout/log-prob -> storage/
  returns -> PPO -> sampler/Gain -> checkpoint/eval/diagnostics.
- Code-confirmed: formal sampler budget planning supports `8/16/32/64`; active
  algorithm config exposes `frontres_segment_max_horizon_k=64`; live capture and
  storage returns consume per-row horizon vectors.
- Code-confirmed: active G1 FrontRES preset sets
  `num_task_corrections=6`, `task_conf_dim=0`, disables split acceptance,
  authority, and state-router branches; live Segment storage/PPO require 6D.
- Blocker found: `FrontRESActorCritic.update_distribution()` puts bounded
  `tanh` outputs in `distribution.mean`; `act()` samples from that Normal and
  applies `tanh` again, while Segment log-prob reconstructs raw actions with
  `atanh` and evaluates them against the bounded mean. The sampled action,
  stored mean, and log-prob are not proven to share one distribution space.
- Gap found: live Segment advantage mode defaults through an implicit
  `getattr(..., "scale_only")` fallback rather than an explicit active
  algorithm config field.
- Existing tests cover fake policy distribution formulas and full-6D/K
  contracts, but do not instantiate the actual FrontRES actor to prove the
  raw/bounded distribution identity.
- Status: Step 10B is `partial/blocked`; no live test or long training should
  start until the direct Delta SE PPO distribution parameterization is confirmed
  and its actual-policy contract is added.

## E20 - Current-Code Method Alignment Recheck (2026-07-13)

- Rechecked the previous E19 blocker against the current source after the
  active-design updates; E19 remains a historical audit observation, not the
  current state.
- Actual policy path is now code-confirmed as one raw Gaussian distribution:
  `FrontRESActorCritic.update_distribution()` stores raw actor logits as
  `distribution.mean`; `act()` samples once and applies one bounded `tanh`; the
  actor and Segment log-prob helpers invert that transform with `atanh` and the
  matching Jacobian.
- Added `frontres_actual_policy_distribution_contract.py`, which exercises the
  actual actor methods without a simulator and checks mean space, action bounds,
  manual transformed log-prob, and per-dimension log-prob summation.
- Made `frontres_segment_advantage_normalization="scale_only"` explicit in the
  algorithm config and Stage 3 preset; made `frontres_segment_max_horizon_k=64`
  explicit in the Stage 3 preset. Stage-entrypoint pseudo coverage now asserts
  both values.
- Verification passed: actual-policy contract; Stage 3 entrypoint pseudo
  contract; Python compile; `git diff --check`; aggregate
  `contract_count=45 failed_count=0 total_marker_count=45`.
- Current status: Step 10B is `partial/offline`, not blocked on a semantic
  decision. Offline method/code alignment is confirmed for K, full-6D action,
  policy distribution, explicit advantage mode, and Gain consumers.
- Remaining limitation: no S4 live evidence yet for finite/diverse action,
  per-row K distribution, raw motion/ZMP/contact metrics, or long-run learning;
  live test remains intentionally paused.

## E21 - Isolated Retired Authority Cleanup (2026-07-13)

- Boundary: package export/import surface -> production caller scan -> dedicated
  test inventory.
- `frontres_authority_space.py` and `frontres_authority_event.py` had no
  production callers beyond `frontres/__init__.py`; their only remaining
  references were dedicated tests and historical notes.
- Removed those two modules, their package exports, and three dedicated tests.
  The perturbation runtime method `frontres_authority_event_state()` was not
  removed because it is still read by the connected generic rollout helper;
  that path is explicitly outside this bounded step.
- Verification passed: import scan found no active source reference to the
  removed modules; package Python compile; aggregate
  `contract_count=45 failed_count=0 total_marker_count=45`; `git diff --check`.
- Status: Step 10C-A completed offline. Connected acceptance/rho/authority
  code remains a separate pending deletion/migration step and is not claimed
  removed.

## E22 - Direct Retired-Path Cleanup (2026-07-13)

- Removed the retired `stage2_acceptance` entrypoint, fake Segment Replay
  connector/projector, authority rollout action/event bridge, authority return
  and target modules, transition alpha/rho payload modules, authority policy
  head/config fields, and dedicated tests.
- Removed the generic runner's authority/acceptance payload write and
  authority-return diagnostic path; formal Stage 3 continues through the
  dedicated Segment runner and full-6D Delta SE(3) PPO.
- Verification passed: Stage 3 pseudo entrypoint, stage entrypoint contract,
  Python compile, `git diff --check`, and aggregate
  `contract_count=41 failed_count=0 total_marker_count=41`.
- Remaining cleanup is explicitly bounded: generic `RolloutStorage` fields,
  old acceptance/structured-rho compatibility in `FrontRESUnified`, and
  legacy diagnostic formatter helpers still require removal or an unreachable
  path proof; they are not claimed deleted by E22.

## E23 - Retired Compatibility Surface Cleanup (2026-07-14)

- Boundary: retired design symbols -> config/defaults -> package exports ->
  negative entrypoint tests -> Segment contract suite.
- Removed the remaining generic PPO optimizer branch for the retired
  `acceptance_actor` surface, stale `ppo_hrl` defaults, unused oracle/floor
  configuration, the unused oracle module/export, and the obsolete Stage 2
  authority wrapper and dedicated authority/legacy tests.
- Updated the active comments and negative entrypoint contract so deletion is
  represented as absence, rather than requiring a legacy wrapper to remain.
- Evidence: `frontres_segment_all_contract_suite.py` passed with
  `contract_count=40 failed_count=0 total_marker_count=40`; targeted full-6D,
  Segment PPO, diagnostics, Stage 3 pseudo, and stage-entrypoint contracts
  passed; `py_compile` and `git diff --check` passed.
- Static residual search now finds only negative assertions and historical
  contract-status text for retired names; no active production caller remains
  for the removed compatibility surface.
- Limitation: no S4 live runtime or checkpoint-artifact resume was run in this
  cleanup step.

## E24 - Common Runner Retired-Field Sweep (2026-07-14)

- Boundary: common runner logging/update injection -> active Stage 2 HSL and
  Stage 3 Segment Replay consumers.
- Removed the unused `rho_advantage`, `alpha_groundtruth`, and
  `alpha_groundtruth_mask` log payload entries. Removed the common-runner
  `oracle_mix` injection and the two unused oracle-mixing config thresholds.
  The standalone `MOSAIC` algorithm implementation remains unchanged.
- Production residual scan is empty for retired acceptance/authority/rho,
  active-dimension, actor-takeover, oracle/floor, and old PPO-actor symbols;
  remaining matches are negative regression assertions only.
- Verification passed: full source/script `compileall`,
  `frontres_segment_all_contract_suite.py` with `40/40`, Stage entrypoint
  contract, Stage 3 pseudo entrypoint contract, and `git diff --check`.
- Limitation: this is offline evidence only. No S4 live runtime or checkpoint
  artifact resume was run.

## E25 - Pre-Fall Style Evidence Mask Correction (2026-07-15)

- Live evidence: `formal_runtime_audit_20260715.txt` reached finite task-space
  application, frozen GMT execution, paired body/ZMP/contact capture, Physics
  Gain, and Repair Cost. `AUDIT-GAIN-01` then reported all-NaN Style Gain.
- Root cause: `_capture_paired_gain()` used final `~done_any` as a row-level
  Style mask. Any fall within K erased all valid pre-fall Style frames. This
  duplicated terminal semantics already owned by Physics Gain and storage PPO
  eligibility, and violated the contract's per-step K/done accumulation rule.
- Fix: body and root-orientation Style now share the executed per-step
  `horizon & not_done_before_step` mask. Post-fall frames are excluded, while
  finite pre-fall evidence remains available for diagnostics and replay.
- PPO semantics are unchanged: `build_live_segment_storage()` still excludes
  terminal policy rows through its independent `valid_mask`.
- Regression: a K=3 pseudo trajectory falls on step 2 and places extreme
  contamination at step 3. Canonical Style MPJPE remains finite and uses only
  steps 1-2; the same policy row remains invalid for PPO.
- Verification passed: Gain component contract, Gain connectivity contract,
  live probe contract, formal runtime audit contract, Runtime Atlas owner-path
  check, Python compile, and aggregate `contract_count=44 failed_count=0`.
- The adjacent audit-only defect `max_horizon_k=missing` was caused by printing
  before the Stage 3 preset assignment. The probe now emits after the value is
  finalized, with a source-order regression assertion.
- Status: S1/S2 correction confirmed. Current source requires the same official
  tiny formal-route rerun before S4 Gain/return or downstream rows can pass.

## E26 - Third Formal Audit Zero-Valid Instrumentation Failure (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715.txt`, 650 lines, timestamp
  2026-07-15 15:46:01.
- Confirmed live: the pre-fall Style correction works. Style, Physics, Repair,
  total Gain, returns, and advantages are finite on the official Stage 3 route.
- The two policy rows were terminal/ineligible, so production Segment PPO
  returned its contract-defined no-step result with `valid_count=0`.
- Root cause of the crash: `frontres_formal_runtime_audit.py` imposed an
  audit-only `valid_count > 0` assertion. This was not a production invariant
  and changed formal-training control flow when audit mode was enabled.
- Fix: the audit now records `valid` and `update_observed`; zero valid rows emit
  `update_observed=0` and remain unconfirmed PPO evidence without raising.
- Regression: a zero-valid critic-warmup result previously reproduces the exact
  assertion and now emits `AUDIT-PPO-01 valid=0 update_observed=0`.
- Deployment identity warning: startup `train.py` reported
  `max_horizon_k=missing`, whereas the runner reported 64 and current local
  `train.py` prints after assignment. The server run therefore did not match
  the full local worktree; no S4 PASS is promoted.
- Next live boundary: synchronized rerun3 with 32 environments, requiring at
  least one eligible policy row before PPO update/trust evidence can close.

## E27 - Fourth Formal Attempt Used Wrong Cost Configuration (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715.txt`, 767 lines, timestamp
  2026-07-15 16:09:19.
- The audit-only zero-valid correction works live:
  `AUDIT-PPO-01 valid=0 update_observed=0` emitted without an assertion.
- The production fail-fast guard then correctly rejected `update_count=0`.
  Existing pseudo contracts require this behavior; it is not a bug to remove.
- The run was not the governed rerun3: PhysX reported 8 envs and the quartet
  contained only two policy rows. The locked command requires 32 envs and eight
  policy rows.
- Both policy rows fell and carried negative total Gain. With no eligible row,
  PPO loss, gradient, parameter delta, and post-update KL are intentionally
  unobserved.
- Deployment remained mixed: startup `train.py` printed
  `max_horizon_k=missing`, while the runner printed 64. Current local script
  prints after assigning 64.
- Decision: preserve all production guards and method semantics. Synchronize
  the complete worktree and execute the exact 32-env rerun3 command before any
  S4 or long-training decision.

## E28 - Rerun3 Full-Quartet Termination Boundary (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 767 lines,
  timestamp 2026-07-15 16:24:31.
- Rerun3 used 32 envs and quartet `8/8/8/8`; the prior row-budget concern is
  closed for this run.
- All 32 rollout rows terminated within K=8. Policy eligibility was 0/8,
  `evidence.fall_count=8`, and the production zero-update guard fired.
- Gain remained finite but negative on average; this cannot be interpreted as
  a PPO update failure because no policy row was eligible.
- Code-supported mismatch: index reset accepts eight sampled rows and writes
  robot state only to env ids 0..7. Quartet reference indices are synchronized
  by the command owner, but no equivalent full-quartet dynamic-state write is
  visible at the reset owner.
- Additional lifecycle risk: `episode_length_buf` is randomized before reset
  and is not reset by the index hook. The log lacks timeout/terminated
  decomposition, so the first live contradiction is not yet localized between
  stale episode lifecycle and invalid quartet dynamic state.
- Decision: do not change PPO, valid masks, Gain, actor bounds, or fail-fast
  guards. First add a role-aware per-step done/timeout/survival probe, then fix
  the earliest contradicted reset invariant with a quartet reset regression.

## E29 - Reset Lifecycle Step A Offline Closure (2026-07-15)

- Added `AUDIT-RESET-LIFECYCLE-01` to the formal rollout owner and the Runtime
  Atlas as the 21st owner card.
- Captured objects are: role-aware episode length before/randomized/after
  reset, quartet root/joint pair error, and per-step done/timeout/physical
  termination/alive/survival with first-done step.
- The pseudo contract uses eight rows in a 2/2/2/2 layout. It distinguishes one
  policy timeout from candidate/Noisy physical terminations and detects
  intentional Noisy root and Clean joint mismatches.
- Verification: `frontres_formal_runtime_audit_contract: ok`,
  `frontres_segment_live_probe_contract: ok`, Atlas viewer import passed, Python
  compilation passed, diff check passed, and aggregate contracts passed 44/44.
- Evidence level: S2 insertion and semantic-contract PASS; S4 remains pending.
  No reset behavior was changed. The next live log must select the earliest
  contradicted owner before a repair is implemented.

## E30 - Reset Lifecycle Live Localization (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 776 lines,
  timestamp 2026-07-15 17:19.
- The official 32-env route used quartet `8/8/8/8`. Index reset wrote only env
  ids 0-7.
- After reset, Candidate/Noisy/Clean joint-state max error versus Policy was
  `8.43462/8.34135/6.73203`, proving the quartet dynamic states are not paired.
- The initial root metric compared world positions across different env origins
  and is therefore invalid as pairing evidence. The corrected probe subtracts
  `scene.env_origins`; root-local evidence is pending another tiny rerun.
- At rollout step 0 every role produced eight dones. Every timeout count was
  zero and every physical-termination count was eight. First-done step was zero
  for all 32 rows.
- Conclusion: the immediate zero-valid failure is not timeout and is upstream
  of PPO. The first proven contract violation is the quartet dynamic-state
  reset boundary. The next bounded change must reset all four role rows from
  the sampled policy-row state before applying role-specific corruption/repair.

## E31 - Quartet Reset Offline Repair (2026-07-15)

- Runner connector now establishes pair layout before reset and passes explicit
  Policy/Candidate/Noisy/Clean env IDs with each sampled index-reset request.
- Adapter reset expands sampled motion/frame to all role rows and writes motion
  group, root pose/velocity, joint pose/velocity, command correction reset, and
  zero episode age. The returned reset-success tensor remains sample-sized.
- Eight-row golden evidence: frame, origin-relative root x, and first joint are
  all `[3,4,3,4,3,4,3,4]`; episode ages are all zero; DR scale is
  `[0.5,1.0,0,0,0,0,0,0]` and local-rp mask is true only for Policy rows.
- Focused contracts passed for stage1 env hooks, live-probe connector, sequence
  eval isolation, and live-training validation/fail-fast behavior.
- Full aggregate verification passed `44/44`; formal audit contract, Atlas
  viewer, Python compilation, and diff check also passed.
- Evidence level: S1/S2 integrated-offline PASS. S4 requires the corrected
  32-env formal live rerun; PPO/Gain semantics were not modified.

## E32 - Quartet Reset Live Closure And Termination Boundary (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 777 lines,
  timestamp 2026-07-15 17:49.
- Deployment identity is proven by `role_env_ids` covering Policy `0..7`,
  Candidate `8..15`, Noisy `16..23`, and Clean `24..31`.
- Reset repair succeeded live: all episode ages are zero; origin-relative root
  mismatch is at numerical-noise scale (`<=1.90735e-6`); all joint mismatches
  are zero.
- The same final exception remains because every aligned role physically
  terminates at step 0. Timeout counts are zero, so this is a new downstream
  boundary rather than failure of the reset patch.
- A role-aware active-term probe now reads current-step masks from IsaacLab
  `TerminationManager.get_term()`. Offline contract preserves term names and
  role counts. S4 term identity remains pending; termination thresholds and
  PPO/Gain paths are unchanged.
- Post-insertion verification passed: aggregate contract suite `44/44`, formal
  runtime audit contract, Runtime Atlas import with 62 owner paths, Python
  compilation, and `git diff --check`.

## E33 - Step-0 Termination Owner Is Anchor Position (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 777 lines,
  timestamp 2026-07-15 18:05.
- `AUDIT-RESET-LIFECYCLE-01` at rollout step 0 reports
  `anchor_pos={policy:8,candidate:8,noisy:8,clean:8}`.
- Every other active term is zero: `motion_end`, `time_out`, `anchor_ori`, and
  `ee_body_pos`. The per-term union matches the returned physical termination
  count exactly.
- Code ownership: active G1 config routes `anchor_pos` to
  `bad_anchor_pos_z_only()`, which compares `command.anchor_pos_w[:,2]` and
  `command.robot_anchor_pos_w[:,2]` against threshold `0.5 m`.
- Confirmed: quartet reset and active-term identity are live evidence.
- Unconfirmed: whether the excessive z error originates from stale cached
  reference/time-step state, reference anchor z, robot torso z, or update
  ordering. No termination or training behavior was changed.

## E34 - Anchor-Z Owner Probe Insertion (2026-07-15)

- `AUDIT-ANCHOR-Z-01` is inserted directly in
  `terminations.py::bad_anchor_pos_z_only()` and remains `PENDING_LIVE`.
- Captured per quartet role: final world-frame reference z, robot torso z,
  signed/absolute error, returned mask, threshold, clean/raw/correction z,
  command time step, and motion index.
- The original mask expression remains
  `(error > threshold) | torch.isnan(error)` and the probe returns that same
  tensor without a clamp, skip, fallback, or threshold change.
- Verification: formal runtime audit contract PASS, Python compilation PASS,
  Runtime Atlas rebuilt with 22 owner cards, viewer/import check PASS with 62
  repository owner paths, aggregate contract suite `44/44`, and
  `git diff --check` PASS.
- Evidence level: S2 insertion/integration PASS. S4 value provenance remains
  pending one tiny official-route formal run.

## E35 - Stale Sampled-Frame Command Cache Causes Step-0 Done (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 779 lines,
  timestamp 2026-07-15 22:02.
- First `AUDIT-ANCHOR-Z-01`: clean reference z mean is approximately
  `0.79 m` and robot z mean is approximately `0.79 m`, while raw/final
  reference z remains near `0..0.03 m`.
- This produces per-role absolute-error means near `0.776 m` and minima above
  `0.543 m`; all rows therefore exceed the configured `0.5 m` threshold.
- Quartet motion indices and sampled time steps are identical across roles.
  Policy/Candidate z correction is only about `-0.001 m` mean and cannot
  explain the error. Clean and Noisy corrections are zero.
- Second owner call: raw/clean reference z agree around `0.78..0.80 m`, and
  maximum absolute error is below `0.014 m`. The first termination has already
  invalidated the K-step rollout by then.
- Root cause boundary: index reset does not initialize the command's current
  sampled-frame perturbation cache before the first termination evaluation.
  This is an integration defect, not a threshold, reset-state, perturbation,
  FrontRES actor, Gain, valid-mask, or PPO defect.

## E36 - Sampled-Frame Command Cache Fix Integrated Offline (2026-07-15)

- Production owner: `commands.py::MultiMotionCommand.`
  `refresh_frontres_reference_cache_current_frame()` builds current-frame
  perturbed position/quaternion cache, supervised target, vertical projection,
  and quartet synchronization without changing `time_steps`.
- Connector: `frontres_segment_stage1_env_hooks.py::`
  `apply_frontres_segment_index_reset()` calls the owner after sampled
  motion/frame and role perturbation setup, before robot write and first env
  termination.
- Eight-row golden evidence: frame identity stays `[3,4,3,4,3,4,3,4]`, cache
  refresh count is one, cache z is `[1,1,1,1,1,1,1,1]`, episode age remains
  zero, and perturbation scale remains Policy-owned.
- Production-source evidence: refresh contains no `time_steps +=`, performs one
  position draw, one quaternion draw, and one quartet sync; `_update_command()`
  advances first and calls the refresh owner exactly once.
- Focused stage1 hook, live-probe, sequence-eval, live-training, PPO-boundary,
  and formal-audit contracts pass. Python compilation passes.
- Aggregate contract suite passes `44/44`; Runtime Atlas viewer/import checks
  62 repository owner paths and `git diff --check` passes.
- Evidence level: S1/S2 integrated-offline PASS. S4 first-step anchor and
  nonzero-update evidence remain pending.

## E37 - Sampled-Frame Cache Fix Integrated Live (2026-07-15)

- Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 803 lines,
  timestamp 2026-07-15 22:32.
- First `AUDIT-ANCHOR-Z-01`: raw, clean, and final reference z align with robot
  z; maximum absolute error is `0.020011 m`, far below threshold `0.5 m`.
- All active termination masks are zero for all four roles through K=8;
  survival reaches eight and first-done indices remain `-1`.
- Formal training continues through storage (`size=32`, `valid=8`), one
  critic-only PPO update (`actor_weight=0`, `KL=6.008e-05`, trust accepted),
  sampler update, diagnostics, and checkpoint save.
- Saved payload: model, optimizer, observation normalizer, sampler, canonical
  Gain config, and warmup state are present in `model_1.pt`.
- Method-quality observation: Gain Total is `-0.006978` with positive Style
  `0.002784`, positive Physics `0.001231`, and Repair Cost `0.073285`; this
  single critic-only batch does not evaluate actor learning quality.
- Evidence level: S4 integrated-live PASS for sampled-frame cache, quartet
  reset, K=8 survival, first formal update, and checkpoint boundary.
- Remaining formal-audit gaps are diagnostic, not execution blockers:
  `KROLLOUT.reset_ok`, `APPLY.delta_norm`, `PAIR.roles`, and `RETURN.rewards`
  are still `missing`; their owner rows remain partial.

## E38 - Four Formal Diagnostic Fields Closed Offline (2026-07-15)

- Root cause: the consolidated audit read three retired summary aliases
  (`reset_success_count`, `delta_se_norm`, `trial_roles`) instead of the live
  owners (`segment_reset_success_frac`, `motion_delta_se_norm`,
  `trial_role_counts`).
- Segment storage already owned the canonical rollout reward, but `_batch()`
  omitted it from `FrontRESSegmentStorageBatch`. The diagnostic batch now
  retains the indexed reward alongside returns and advantages.
- `to_ppo_batch()` remains unchanged and does not forward reward into the PPO
  loss contract. Reset/cache, Gain calculation, PPO math, and update control
  flow are unchanged.
- Pseudo-sample contracts prove all four owner rows contain real values and no
  `missing`; the storage contract proves reward row identity is preserved.
- Evidence level: S1/S2 integrated-offline PASS. S4 actor-warmup formal rerun
  remains required to confirm the fields on the official runtime path.

## E39 - Actor-Warmup Formal Sentinel (2026-07-15)

- Raw evidence: `formal_runtime_audit_actor_warmup_20260715.txt`, 32 envs,
  two iterations, critic warmup 1 and actor warmup 2.
- Iteration 0 is `critic_only` with actor weight 0. Iteration 1 is
  `actor_warmup` with actor weight 0.5, seven valid policy rows, finite loss,
  post-update KL `0.005442`, and accepted trust-region update.
- The four E38 fields are live-populated: reset success `1.0`, applied Delta
  SE(3) norm `0.016680`, role counts policy 8/baseline 24, and finite
  reward/return/advantage tensors.
- Actor parameters change (`residual_actor.0.weight` first); frozen GMT remains
  outside the optimizer. The second rollout Gain is negative, but it precedes
  that iteration's optimizer step and is not evidence of post-update quality.
- Evidence level: S4 PASS for Step G wiring and E38 diagnostic closure.

## E40 - Perturbation Audit Aliases Closed Offline (2026-07-15)

- Root cause 1: the formal task algorithm config lacked
  `frontres_segment_max_horizon_k` and
  `frontres_segment_advantage_normalization`, although the mirrored rsl_rl
  config defined both. `_set_if_present()` therefore skipped the preset and
  runtime happened to use algorithm defaults.
- Root cause 2: the formal rollout summary did not retain the perturbation
  family/strength already consumed by the index-reset request.
- The task config now owns `64` and `scale_only`. Reset summary now records
  actual family counts and strength min/mean/max; no-perturbation reset requests
  produce empty/zero diagnostics without failing.
- Focused entrypoint, formal-audit, and live-probe contracts pass, including a
  three-row `local_rp` fixture with strengths `[0.25, 0.5, 0.75]`.
- Evidence level: S1/S2 integrated-offline PASS. One S4 rerun remains required
  for the corrected compact perturbation fields.

## E41 - Perturbation Audit And Compact Formal Route Closed Live (2026-07-16)

- Raw evidence: refreshed `formal_runtime_audit_actor_warmup_20260715.txt`,
  run directory timestamp 2026-07-15 23:48, reviewed 2026-07-16.
- Both startup and runner probes report `specialist_mode=rp`,
  `perturbation_channels=rp`, `dr_scale=1.25`, and `max_horizon_k=64`.
- The actor-warmup rollout reports `family_counts={'local_rp': 8}` with
  strength min `0.151714`, mean `0.858859`, and max `1.313188`.
- The log contains no audit `missing`, traceback, NaN, or Inf. Reset success is
  `1.0`; seven of eight policy rows are valid; actor-warmup post KL is
  `0.005442` and the trust-region update is accepted.
- `model_2.pt` saves model, optimizer, observation normalizer, sampler, Gain
  config, and warmup identity.
- Method-quality boundary remains open: the second rollout has total Gain
  `-0.325550`, Physics Gain `-0.281932`, repair cost `0.115193`, one fall, and
  repaired MPJPE `0.095913` versus noisy MPJPE `0.072490`. This rollout is
  collected before its actor update, so it does not measure post-update policy
  improvement.
- Evidence level: S4 PASS for Step H and compact formal-route diagnostics;
  method-quality acceptance remains pending Step I.

## E42 - Same-Run Checkpoint Pair Verified (2026-07-16)

- Raw evidence: `formal_actor_pair_20260716.txt`, local `model_1.pt`, and local
  `model_2.pt`.
- `model_1.pt` is `iter=1` and `model_2.pt` is `iter=2`; both contain model,
  optimizer, observation normalizer, privileged normalizer, sampler, Gain, and
  warmup payloads. Both use `FRS-GAIN-v001`, six-dimensional `std=0.01`, and
  learning rate `1e-6`.
- The nested model state contains 19 common tensors. Between the checkpoints,
  18 tensors change: all eight residual-actor tensors change, while the policy
  std remains unchanged. Optimizer state grows from 10 to 18 entries, matching
  the actor entering optimization after critic-only warmup.
- This proves the pair is suitable for a controlled same-sequence comparison.
  It does not yet prove which checkpoint has better repair quality.
- Evidence level: S2 checkpoint identity/parameter-delta PASS; Step I fixed
  offline evaluation remains pending.

## E43 - Fixed-Sequence Eval Reproducibility Guard (2026-07-16)

- The checkpoint loader restores the Segment sampler before the official
  sequence evaluator runs. That replay history is valid training-resume state,
  but it is not a valid selector for comparing `model_1.pt` and `model_2.pt`.
- Sequence offline eval now rebuilds the sampler's initial state with an
  explicit `--frontres_segment_sequence_eval_seed` before selecting candidate
  motions. The policy, observation normalizer, environment, and Gain contract
  are unchanged; only evaluation sampler history is ignored.
- The evaluator prints `reset=1`, the seed, and
  `checkpoint_replay_state=ignored`. It still prints per-sequence motion IDs,
  reset frames, preroll, and evaluation boundaries for the human comparison.
- Offline contracts pass: sampler reset pseudo-sample, sequence evaluator,
  Stage 3 launch command, live sentinel, and Stage 1/2/3 entrypoint contracts.
  Python compilation also passes for all changed modules.
- The actual IsaacLab evaluation is not run on this Mac because the server
  motion/cache paths under `/hdd1/cyx` are unavailable. Step I remains pending
  until both checkpoints produce logs with identical `motion_ids` and
  `reset_frame/preroll` fields.
- Evidence level: S1/S2 evaluation reproducibility PASS; S4 checkpoint quality
  comparison pending.

## E44 - Eval-Only Warmup Guard Fix (2026-07-16)

- `eval_model1_fixed.txt` failed before sequence rollout because the command
  set `STAGE3_IS_FULL_RESUME=True`. The checkpoint stores the short audit-run
  warmup config `critic=1, actor=2`, while the runtime training defaults are
  `critic=200, actor=500`.
- This is a mode mismatch, not a bad checkpoint. Warmup equality is a
  training-resume invariant; it is irrelevant after the runner enters an
  offline evaluation-only route.
- Checkpoint loading now skips only this warmup equality guard when either
  Stage 3 offline-eval flag is active. Actor weights, observation normalizer,
  policy std, sampler payload, and FRS-GAIN-v001 validation remain active.
  Formal training resume still raises on the same mismatch.
- Python compilation and the full live-sampler checkpoint contract pass after
  the fix. The pair quality comparison remains pending a rerun of both eval
  commands.
- Evidence level: S1/S2 load-boundary fix PASS; S4 fixed-sequence quality
  comparison pending.

## E45 - Sequence Eval Seed Forwarding Fix (2026-07-16)

- The refreshed log confirmed E44 was active: `eval_only=True` and the warmup
  guard was skipped, so the checkpoint loaded successfully.
- The next failure was a missing API edge: `train.py` passed `sampler_seed`,
  but `OnPolicyRunner.run_frontres_segment_sequence_offline_eval()` did not
  accept or forward it to the sequence-eval owner. No rollout had started yet.
- The runner wrapper now forwards `sampler_seed` end to end. Python
  compilation, sequence-eval, Stage 3 entrypoint, and explicit three-layer
  seed-wiring contracts pass.
- The pair quality comparison remains pending a fresh server rerun. The
  previous `eval_model1_fixed.txt` contains no evaluation metrics because it
  stopped at this wrapper boundary.
- Evidence level: S1/S2 integration fix PASS; S4 fixed-sequence quality
  comparison pending.

## E46 - Fixed-Sequence Perturbation Curriculum Fix (2026-07-16)

- Both new eval logs completed without traceback. The two motion IDs,
  `reset_frame=0`, `preroll_steps`, and `eval_start_frame` matched exactly.
- They were nevertheless not a strict paired comparison: `model_1` used
  `seq_idx=100000` and strengths `[0.921171, 1.313188, 1.085961, 0.318131]`,
  while `model_2` used `seq_idx=200000` and strengths
  `[0.286224, 0.542230, 0.982733, 1.183559]`.
- Root cause: the sequence evaluator's perturbation plan inherited both
  `current_learning_iteration` through `seq_idx` and curriculum progress.
  Thus checkpoint iteration leaked into the supposedly fixed corruption.
- Sequence eval now uses the explicit eval seed for `seq_idx` and fixes
  perturbation curriculum progress at `1.0`; normal training still uses its
  iteration-based curriculum. Contracts prove equal family/strength plans
  across different checkpoint iterations.
- The old metrics are diagnostic only, not a valid checkpoint ranking. A new
  pair run is required before Step I quality acceptance.
- Evidence level: S1/S2 perturbation-control fix PASS; S4 paired quality
  comparison pending.
