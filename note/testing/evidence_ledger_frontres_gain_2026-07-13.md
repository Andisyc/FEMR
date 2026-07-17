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

## E47 - Valid Same-Condition Checkpoint Comparison (2026-07-16)

- Raw evidence: refreshed `eval_model1_fixed.txt` and `eval_model2_fixed.txt`.
  Both runs completed without traceback, reset all four rows successfully, and
  report the same two motion IDs, reset frames, preroll steps, eval starts,
  `local_rp` family, and perturbation strength vector
  `[1.248498, 0.477477, 0.509009, 1.081832]`.
- Motion `CMU/74/...`: both checkpoints remain at 25% success, 75% fall, and
  survival `76.5`. Model 2 changes total Gain only from `16.221064` to
  `16.232580`; repaired MPJPE changes `0.140732 -> 0.140054`. This is a
  negligible policy-quality change on this motion.
- Motion `BioMotionLab_NTroje/...`: model 2 improves success `75% -> 100%`,
  fall `25% -> 0%`, survival `113.5 -> 120.0`, repaired MPJPE
  `0.040587 -> 0.038695`, velocity error `0.004418 -> 0.002475`, and
  acceleration error `0.003281 -> 0.001510`. Its total Gain changes from
  `6.747443` to `-0.022432`, showing that the current Gain decomposition and
  survival outcome are not perfectly aligned on this sample.
- Aggregate: model 2 success `50% -> 62.5%`, fall `50% -> 37.5%`, survival
  `95.0 -> 98.2`, while total Gain decreases `11.484253 -> 8.105074` and
  positive-gain fraction changes `100% -> 50%`. The result is mixed, not a
  universal improvement claim.
- Evidence level: S4 valid paired comparison PASS; broad method-quality
  acceptance remains pending a larger fixed-sequence set and Gain/survival
  alignment review.

## E48 - Gain Component Audit Boundary (2026-07-16)

- Active Gain owner is `frontres_gain.py`: Style is the mean of normalized
  Clean-vs-Repaired improvements for MPJPE, velocity, acceleration, and root
  orientation; Physics is the mean of available paired success, survival, ZMP,
  and contact improvements; Repair Cost is the executed full-6D norm/temporal
  cost. The composition is
  `style_weight * style + physics_weight * physics - repair_weight * cost`.
- The E47 logs prove a real semantic tension but not a formula defect: model 2
  improves survival and dynamics on the second motion while its total Gain is
  negative. The logs only expose aggregate Physics, not its success/survival/
  ZMP/contact components, so attribution from E47 alone would be speculation.
- Diagnostics now expose all four Physics subcomponents in the live and
  sequence-eval Gain lines. No Gain weights or formulas were changed.
- Python compilation and sequence-eval, live-probe, and diagnostics contracts
  pass. A new paired eval is required to determine whether the conflict is
  systematic or a two-motion sample effect.
- Evidence level: S1/S2 Gain-owner and diagnostic audit PASS; S4 component
  correlation and method-quality decision pending.

## E49 - Gain Audit Rerun Boundary (2026-07-16)

- Four pulled logs were inspected. The files named `seed20260716` and
  `seed20260717` all print the internal runtime seed `20260716`, the same
  `seq_idx=2026071600000`, and the same eight motion IDs. Thus only one matched
  seed was actually run; the second seed is missing, not a second result.
- The valid `20260716` pair improves aggregate survival from `101.3` to
  `105.1` and reduces fall from `31.2%` to `25.0%`, but total Gain decreases
  from `-0.510026` to `-0.946795`. This is a valid mixed result, but it cannot
  complete the two-seed variance audit.
- All four logs print `physics_components=(success=UNCONFIRMED,
  survival=UNCONFIRMED, zmp=UNCONFIRMED, contact=UNCONFIRMED)`. The local
  diagnostic patch is therefore not present in this server run; aggregate
  Physics cannot yet be decomposed.
- Status: Step I-B partial. Next action is to synchronize the two diagnostic
  owner files, run the true `20260717` pair, and verify the log header itself
  before comparing results.

## E50 - Gain Component Diagnostic Alias Fix (2026-07-16)

- The local source audit found a concrete formatter boundary defect. The
  sequence/periodic summary builder created internal keys such as
  `gain_physics_success_gain_mean`, while `_format_eval_gain_line()` read
  `gain_physics_success_mean`. The same mismatch affected per-motion rows.
- This explains the observed combination of finite aggregate `gain_physics`
  and `physics_components=(... UNCONFIRMED ...)` without changing the Gain
  formula or proving that the underlying Physics tensors were missing.
- `frontres_segment_live_training.py` now centralizes the internal-to-public
  Gain component aliases for sequence, periodic, and per-motion summaries.
  The sequence contract includes a finite four-component pseudo Gain fixture
  and asserts the rendered success/survival/ZMP/contact values.
- Fresh local verification: both modified Python files compile and `git diff
  --check` passes. The targeted torch contract could not run on this machine
  because the available Python environments do not provide `torch`.
- Evidence level: code-confirmed diagnostic root cause; S1/S2 static/compile
  verification; S4 live component population and freshness remain pending.

Next:
- Upload the two modified source/test files, rerun the same matched sequence
  evaluation with `--frontres_formal_runtime_audit`, and require both
  `[AUDIT-GAIN-01]` output and finite `physics_components` values before using
  Gain to rank checkpoints.

## E51 - Live Gain Component Population And Unit Boundary (2026-07-16)

- All four refreshed matched-evaluation logs set `formal_runtime_audit=True`
  and emit `AUDIT-PAIR-EVIDENCE-01` plus `AUDIT-GAIN-01`. The captured Style,
  ZMP, contact, repair cost, and total Gain tensors are finite. Sequence logs
  now print finite `success`, `survival`, `zmp`, and `contact` components.
- Reset/evaluation oracles are true for the sampled rows: frame-zero reset,
  evaluation start frame, motion identity, RP-only family, role alignment,
  metric shapes, reset success, and summary motion alignment. No Python fatal
  marker appears in the four logs.
- The two checkpoint pairs are now genuinely matched by seed. Seed 20260716
  changes model 1 -> model 2 from success `68.8%` to `75.0%` and survival
  `101.3` to `105.1`, but total Gain falls `-0.510026` to `-0.946795` and
  positive fraction falls `87.5%` to `62.5%`. Seed 20260717 keeps survival at
  `115.6`, while total Gain is nearly unchanged (`0.011984` to `0.011943`)
  and positive fraction falls `75.0%` to `50.0%`. This is mixed checkpoint
  behavior, not a universal improvement claim.
- The negative Gain is not evidence that Physics fields are missing. It is a
  paired result: `physics_survival_gain` compares Repaired survival against
  Noisy survival, not Repaired survival against zero. In seed 20260716 the
  printed survival component is `-2.5` for model 1 and `-4.0` for model 2.
- New code-confirmed risk: `_capture_paired_gain()` passes raw survival steps
  into `compute_segment_gain()`, while the per-step training path passes
  survival divided by the current rollout step into
  `compute_segment_gain_step()`. The active contract requires shared units and
  scales across training and evaluation; this normalization boundary is not
  yet contract- or live-confirmed.
- Evidence level: S4 paired Physics/Style/Repair population and diagnostic
  freshness PASS; training-return/evaluation unit alignment remains partial.

Next:
- Audit survival units and K aggregation offline and through the formal
  training-return route. Do not silently add a scale or change the Gain
  formula until the accepted contract decides whether survival is raw horizon
  difference or normalized horizon quality.

## E55 - Normalized Survival Gain Alignment (2026-07-16)

Scope:
- Activated `FRS-GAIN-v002-style-physics-repair.md`, superseding v001.
- `frontres_gain.py` now owns `survival_quality = survival_steps / K` and
  rejects missing or shape-incompatible K as non-finite evidence.
- Final paired Gain consumes cumulative raw survival steps with per-row K.
- Per-step Segment PPO Gain consumes the current alive increment with the same
  K, so its survival component sums to the final K-normalized difference.
- Evaluation keeps raw `mean_survival_steps` separate and prints repaired,
  noisy, and Gain survival quality. Long sequence survival remains separate.

Commands and results:
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
  PASS: `k1_quality_gain=1.000000`, `k4_quality_gain=0.500000`,
  `k8_quality_gain=0.250000`, per-step `[0, 0, 0.25, 0.25]`, sum `0.5`.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`
  PASS.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py`
  PASS; active diagnostic source is `FRS-GAIN-v002`.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
  PASS after updating the log contract to show
  `survival_quality=(repaired=... noisy=... gain=...)`.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py`
  PASS.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`
  PASS.

Evidence level: S1/S2 contract and consumer connectivity confirmed offline.
Limitation: no new IsaacLab formal runtime was run in this step; live
population, mixed-K behavior, sampler evidence, and 120-step sequence output
remain S4 evidence boundaries.

Next:
- Run one minimal formal Stage 3 sentinel with mixed effective K and inspect
  raw survival steps, repaired/noisy survival quality, per-step return, and
  `gain_source=FRS-GAIN-v002` before any training decision.

## E56 - Formal Audit v002 Instrumentation (2026-07-16)

Scope:
- Live capture now retains the already-computed per-step
  `physics_survival_gain` beside `gain_steps`.
- `AUDIT-GAIN-01` prints raw policy-row survival steps, policy-row effective K,
  repaired/noisy survival quality, normalized survival Gain, per-step survival
  Gain sum, final survival Gain mean, and their absolute difference.
- `AUDIT-RETURN-01` prints the same K and survival Gain trace beside Gain
  steps, storage rewards, returns, and advantages.
- Runtime Atlas GAIN/RETURN cards now explain these v002 objects and mark the
  previous E37 live status as `stale-rerun-required`.

Verification:
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_formal_runtime_audit_contract.py`
  PASS; the fixture reports v002 fields and step-sum error below `1e-5`.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`
  PASS.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
  PASS.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py`
  PASS.
- `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py`
  PASS.
- Runtime Atlas generator:
  `/Users/chengyuxuan/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node note/architecture/auxiliary/atlas_app/build_formal_runtime_audit.mjs`
  PASS; generated 22 owner cards.

Evidence level: S1/S2 code and formal-route contract-confirmed. No live
IsaacLab run was started in this step. The C v002 S4 gate remains open until a
fresh official `MODE=train` run emits these fields with
`gain_source=FRS-GAIN-v002`.

## E52-SIDE-PROPOSAL - Side-Conversation Survival Design Proposal (2026-07-16)

- This entry records a temporary side-conversation result for planning in the
  main conversation. It is not an active contract decision and does not
  authorize a code change.
- The external-code review confirms that Level Replay supplies prioritized
  level-score aggregation, not a survival-Gain formula. Its score functions
  operate on rollout-derived quantities and partial scores are merged with
  step-count weighting. The official reference is
  `https://github.com/facebookresearch/level-replay`.
- The local MOSAIC reference uses normalized `frontres_survival_rate` for
  quality diagnostics and normalizes episode length by a reference horizon for
  frontier scoring. FEMR's own external reuse map says K-step repair reward is
  FEMR-specific and should not be copied from Level Replay.
- Planning candidate: preserve raw survival steps as a diagnostic, define a
  K-normalized survival quality for paired Physics Gain, and keep long-sequence
  survival separate from short K-step Gain. This remains a proposal until the
  main conversation confirms the semantic choice.
- Next main-conversation action: decide raw versus normalized semantics, then
  create the smallest deterministic K=1/4/8 contract before considering any
  live run or training.

## E52-OFFLINE-PROBE - Survival Unit And K Aggregation Offline Probe (2026-07-16)

- Added `test_survival_unit_and_k_aggregation_probe()` to the existing Gain
  component contract. It reuses `compute_paired_physics_gain()` and prints the
  same paired survival evidence as raw steps, fixed-K normalized steps, and
  per-step normalized values over `K=4`; it does not change the Gain owner or
  formula.
- The hand-checkable fixture is `repaired=[1,2,3,4]` and
  `noisy=[1,2,2,2]`. Its expected values are raw final delta `2.0`, fixed-K
  delta `0.5`, per-step deltas `[0,0,1/3,1/2]`, and per-step sum `5/6`.
  These are contract expectations, not live observations yet.
- Fresh local verification: the modified contract compiles and `git diff
  --check` passes. Executing the contract is blocked on this machine because
  no available Python environment provides `torch`.
- Evidence level: code-confirmed probe path and static verification;
  contract execution and formal training-return comparison remain pending.

Next:
- Run the Gain contract in the server `mosaic` environment, then compare its
  output with `AUDIT-RETURN-01` from one minimal formal Stage 3 update. Do not
  change survival units before those two paths are visible side by side.

## E53 - Local Survival Unit Probe Execution (2026-07-16)

- The local `./frontres` environment is usable: Python `3.13`, torch `2.12.1`,
  and the repository `rsl_rl` package import successfully.
- Fresh command:
  `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py`
- Observed result: `raw_delta=2.000000`, `fixed_k_delta=0.500000`,
  `per_step_delta=[0.0, 0.0, 0.3333333134651184, 0.5]`,
  `per_step_sum=0.833333`, followed by
  `frontres_gain_components_contract: ok`.
- This contract confirms the numerical difference between raw, fixed-K, and
  per-step normalized survival paths. It does not select the method's accepted
  survival unit and does not prove the formal PPO return route uses the same
  path.
- Evidence level: contract-confirmed offline numerical audit; formal
  training-return comparison remains open.

## E54 - Survival Design Document Governance Review (2026-07-16)

- The side-conversation document
  `plans/survival_gain_unit_alignment_20260716.md` is correctly classified as
  a proposal. It does not alter `FRS-GAIN-v001` and does not authorize code or
  training changes.
- Its mature-practice claim is currently `note-confirmed`: the note records
  normalized survival diagnostics in MOSAIC and distinguishes Level Replay's
  score aggregation from FEMR's survival-Gain semantics. Those external
  claims still require independent reference review if they become the basis
  for an active contract.
- The evidence ledger previously contained two `E52` headings. They are now
  disambiguated as `E52-SIDE-PROPOSAL` and `E52-OFFLINE-PROBE`; `E53` remains
  the local executed contract result.
- No Concept Figure or active contract update is appropriate before the main
  conversation confirms raw versus normalized survival semantics and the
  effective-K owner.

## E57 - Formal Audit Horizon Scope Regression Fix (2026-07-16)

- The first v002 formal-route attempt reached
  `frontres_segment_live_probe.py::_run_live_rollout_capture` and failed before
  the per-step Gain call because the new step-Gain instrumentation referenced
  `capture.horizon_k` before the `capture` object was constructed.
- The owner already had the effective per-row window in the local
  `horizon_k` tensor. The fix uses `horizon_k[:n_pair]` for the step-Gain call;
  no K assignment, Gain formula, action representation, or PPO behavior was
  changed.
- Regression coverage now checks that the pre-capture owner reads the local
  horizon. Fresh local results: Python compilation, live probe contract,
  formal audit contract, Gain component contract, and `git diff --check` all
  pass.
- Evidence level: code-confirmed and contract-confirmed. The formal `MODE=train`
  runtime gate remains open and must be rerun after deployment.

## E58 - Formal Actor Warmup Runtime Probe (2026-07-16)

- Fresh log: `formal_runtime_audit_actor_probe_20260716.txt`.
- The official route reached iteration 201 with `phase=actor_warmup`,
  `actor_weight=0.002`, `valid=8`, and `update_observed=1`.
- The first actor-weighted update produced `post_mean_delta_l2=8.528e-04`,
  `post_mean_delta_max=0.001053`, `post_kl=0.004131`, and
  `trust_accepted=1`. GMT remained frozen with
  `gmt_trainable=0` and `gmt_in_optimizer=0`.
- `AUDIT-GAIN-01` and `AUDIT-RETURN-01` remained populated with finite v002
  survival fields, rewards, returns, and advantages. The final checkpoint
  `model_201.pt` saved with model, optimizer, normalizer, sampler, Gain config,
  and warmup payloads.
- The full-quartet `gain_steps` diagnostic still reports `finite=0` because
  non-policy quartet rows are intentionally NaN; the policy-row v002 survival
  trace and PPO tensors are finite. This is a diagnostic readability gap, not
  evidence of invalid policy rewards.
- Evidence level: runtime-confirmed for the first actor-warmup update and
  v002 return wiring. It does not prove long-run actor quality or final policy
  superiority.

## E59 - Local Actor Checkpoint Pair Audit (2026-07-16)

- Local artifacts: `model_200.pt` (892M, `iter=200`) and `model_201.pt`
  (897M, `iter=201`). Both contain model, optimizer, warmup, Gain, sampler,
  observation-normalizer, and privileged-normalizer payloads.
- The local `./frontres` environment loaded both checkpoints sequentially.
  Matching `residual_actor` tensors are finite and every actor layer changed
  between the pair; `model_state_dict.std` and both normalizer `_std` tensors
  are unchanged. This is consistent with `model_200 -> model_201` being the
  first actor-weighted update boundary.
- The local environment passes the sequence-eval contract, but it has no
  importable `isaaclab` package and no local `AMASS_G1NPZ_Final` or
  `AMASS_G1Segment` directory. Therefore no physical matched sequence eval was
  executed locally and no policy-quality claim is made from this audit.
- Evidence level: checkpoint-structure-confirmed and contract-confirmed;
  physical offline evaluation remains runtime-unconfirmed.

## E60 - Matched Actor Boundary Offline Evaluation (2026-07-16)

- Fresh logs: `eval_actor_probe_model_200.txt` and
  `eval_actor_probe_model_201.txt`. Both use seed `20260716`, the same two
  motion IDs, the same reset/preroll/eval frames, the same `local_rp`
  perturbation strengths, and the same effective K=4. Checkpoint replay state
  is explicitly ignored.
- Aggregate result over two sequences: success and survival remain identical
  at `50%` and `80.8` respectively. `gain_total` changes from `-0.088949`
  (model_200) to `-0.095243` (model_201), while Style changes
  `-0.043280 -> -0.049838`, Physics changes `-0.023619 -> -0.023569`, and
  repair cost changes `0.147002 -> 0.145581`.
- The first motion remains a failure and worsens in Gain
  `-0.188393 -> -0.203621`; the second motion remains successful and improves
  in Gain `0.010495 -> 0.013134`. MPJPE and real-vs-zero diagnostics are mixed.
- Both policies produce non-zero 6D segment actions, so the actor update did
  not create a no-op. The two-motion sample is too small to claim actor
  improvement or degradation as a general result.
- Evidence level: matched offline runtime-confirmed; policy-quality decision
  remains open pending a larger fixed-motion pair evaluation.

## E61 - Phase B Gain Consumer Alignment Reclassification (2026-07-16)

- The active Phase B objective is the semantic consumer path, not proof that a
  single actor update improves behavior:
  `paired capture -> diagnostic Gain -> per-step reward -> returns/advantages`.
- The required boundary is same-transaction identity of paired rows, effective
  K, done mask, and Style/Physics/Repair components. Local K/unit contracts
  pass; formal consumer comparison remains open.
- The four matched model-200/model-201 sequence logs remain raw exploratory
  quality evidence. They are not promoted to a training gate and do not
  establish long-run RL improvement.
- Model-pair and multi-seed quality evaluation is deferred until after formal
  consumer alignment. The double-layer Segment Replay design supplies repeated
  segment evidence and PPO updates; one update is not expected to improve every
  metric.
- Evidence level: workflow decision confirmed by active contracts and formal
  audit scope; formal same-transaction consumer equality remains unconfirmed.

Next:
- Run the smallest local component/consumer pseudo test, then expose the same
  values in one official formal-route update. Do not change Gain weights or PPO
  semantics and do not start a model-pair quality rerun for this boundary.

## E62 - Local Gain Consumer Pseudo-Data Gate (2026-07-16)

- Scope: local implementation/consumer path only; no simulator physics, policy
  quality, or single-update improvement claim.
- Commands and results:
  - `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_gain_components_contract.py` -> `ok`.
  - `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_gain_connectivity_contract.py` -> `ok`.
  - `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py` -> `ok`.
  - `frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_training_pseudo_contract.py` -> `result: PASS`.
- Observed facts:
  - `K=1/4/8` survival quality uses the same raw step counts with the
    expected inverse-K normalization; the K=4 per-step survival Gain sum
    matches final survival Gain.
  - canonical `gain_total` reaches Segment storage and K-step returns; an
    extreme unrelated environment reward does not replace it.
  - paired action, done mask, reset validity, returns, and advantages retain
    their declared shapes and role masks in the pseudo consumer route.
  - the training pseudo route reports `gain_total=0.750000` while
    `env_reward=-0.500000` and rejects incomplete/non-finite/zero-update/too-
    few-valid summaries.
- Limitation: some weak pseudo fixtures intentionally print
  `gain.source=UNCONFIRMED` because they do not contain physical four-role
  evidence. This is not formal live Gain evidence.
- Evidence level: implementation and offline consumer confirmed; formal
  same-transaction runtime comparison remains open.

Next:
- Run the smallest official `MODE=train` runtime update and compare
  `AUDIT-GAIN-01` with `AUDIT-RETURN-01` for the same paired transaction.
  Do not change Gain weights, PPO semantics, or start long training.

## E63 - Same-Rollout Gain Consumer Boundary (2026-07-17)

- Scope: preserve the current Phase B boundary while handing the work back to
  the main conversation. This entry records source-level ownership, not a new
  quality claim.
- Code fact: `AUDIT-GAIN-01` and `AUDIT-RETURN-01` are not independent
  samples. The official route passes one `FrontRESSegmentLiveRolloutCapture`
  through `run_frontres_segment_live_probe()` into
  `build_live_segment_storage()`. The Gain diagnostic reads the paired
  capture; storage reads its reward steps, done steps, and horizon K before
  computing returns and advantages for PPO.
- Confirmed boundary: both diagnostics refer to the same rollout and the same
  policy-row batch. The remaining check is the values carried by those rows:
  effective K, done mask, Style/Physics/Repair components, and the equality
  between per-step survival Gain sum and final survival Gain.
- Evidence class: source/code-confirmed plus local consumer pseudo-data
  confirmed by E62. A single PPO update improving behavior is not required and
  is not used as a Phase B acceptance target.
- Open: one official `MODE=train` log must expose the above values together so
  the formal runtime claim can be marked closed. Do not change Gain weights,
  PPO semantics, or start long training for this boundary.

## E64 - Rollout Transaction Identity Offline Repair (2026-07-17)

- Problem: Cards 15/16/17 had no explicit shared runtime identity, while Card
  22 consumed update-loop summaries without stating whether they represented a
  single capture or an aggregate.
- Fix: one capture now creates `audit_transaction_id` and
  `audit_batch_signature` from the ordered row tuple
  `(segment_id, role, motion_id, start_frame, effective_K)`. Gain, storage,
  returns, and formal probes preserve the identity. Storage rejects mixed
  transactions. The PPO adapter intentionally strips this diagnostic metadata
  before constructing `FrontRESSegmentPPOBatch`. The update loop and
  diagnostics classify `single` versus `aggregate` and report transaction/batch
  counts.
- Offline evidence: `py_compile`; gain connectivity; live update-loop;
  diagnostics; formal runtime audit; Gain component; live probe; live training
  pseudo; and Stage 3 pseudo contracts passed after the repair.
- Evidence level: implementation and offline identity propagation confirmed;
  equality in a real formal runtime log remains S4 open. No reward, action,
  PPO, or Gain formula was changed.
- Next: only a tiny official `MODE=train` run may close S4 identity equality;
  do not start long training for this boundary.

## E65 - Current-State Stale Text Cleanup (2026-07-17)

- Problem: the current audit header and engineering plan used old rerun3 reset
  language as if reset ownership were still the active blocker.
- Correction: rerun3 termination facts remain preserved as historical evidence;
  the current Phase B gate is the post-E64 real transaction identity and
  numeric Gain-to-return comparison.
- The locked formal command now uses the unique
  `formal_runtime_audit_gain_identity_20260717.txt` log and matching run name,
  so the next raw evidence cannot overwrite the 20260716 artifact.
- Scope: governance/document wording only. No source behavior, probe location,
  reward, action, PPO, sampler, or checkpoint behavior changed.
- Verification: formal audit contract and `git diff --check` are required;
  the next live command remains blocked until source/checkpoint identity is
  confirmed and the user approves the tiny official `MODE=train` run.

## E66 - Card 17 Policy-Row Gain-Step Diagnostic Repair (2026-07-17)

- Live finding: the official run reached Cards 15/16/17/22 with one shared
  transaction, but `AUDIT-RETURN-01` printed full `[T,32]` quartet
  `gain_steps`; the 24 non-policy rows are intentionally NaN, while the
  policy slice and actual returns/advantages were finite.
- Root cause: the audit formatter read `capture.gain_steps` before applying the
  same `n_train` policy-row boundary already used by the reward owner.
- Fix: `_policy_gain_steps_for_audit()` now reports only `[T,:n_train]` for
  Card 17. No reward, storage, returns, advantage, PPO, or quartet semantics
  changed.
- Offline evidence: py_compile and
  `frontres_formal_runtime_audit_contract.py` passed, including a `[T,32]`
  fixture with `[T,24]` non-policy NaNs and a finite `[T,8]` policy slice.
- Status: superseded by E67 below for the current-revision official rerun.

## E67 - Official Gain Identity Rerun (2026-07-17)

- Raw evidence: `/Users/chengyuxuan/ArtiIntComVis/formal_runtime_audit_gain_identity_20260717.txt`,
  809 lines, modified after the E66 source fix. The filename retained the
  original run name, but its runtime output is the post-fix rerun.
- Official route: `AUDIT-ROUTE-01` reports
  `objective=segment_replay_hrl`, `live_train=1`, `alternate_modes=0`,
  `iterations=1`; Stage 3 uses `model_warmup.pt` with the active 6D actor.
- Card 15/16/17 identity: paired evidence, canonical Gain, and returns all
  report `audit_transaction_id=iter0:capture1`,
  `audit_batch_signature=b21ee717d66475f3`, and
  `audit_identity_state=complete`. Card 22 reports
  `mode=single`, `transactions=1`, `batches=1`, `same_transaction=True`.
- Card 17 repair is live-confirmed: `AUDIT-RETURN-01` reports policy-row
  `gain_steps=shape=(8,8), finite=1`; raw survival steps and effective K are
  `[8]`, the per-step survival sum error is `0.0`, and returns/advantages are
  finite. The previous `[T,32] finite=0` output was the stale formatter path.
- Gain values are finite under `FRS-GAIN-v002`: style `0.002784`, physics
  `0.001231`, repair cost `0.073285`, total `-0.006978`. The negative total is
  a one-batch quality observation, not a Phase B consumer failure.
- K interpretation: `cache_horizon_k=4` is the Stage 1 reference-cache
  window, while the sampled/effective training horizon is `K=8`; the log and
  current documentation treat these as distinct owners.
- PPO limitation: the single update is `phase=critic_only`,
  `actor_weight=0`, so it proves finite critic/update diagnostics and frozen
  GMT ownership, not actor learning or post-training improvement. The run
  still saved `model_1.pt` with model, optimizer, normalizer, sampler, Gain
  config, and warmup payloads.
- Migration warning retained: `model_warmup.pt` has no Gain config and carries
  a legacy 12D noise-std tensor; the active 6D runtime explicitly resets std
  to `0.01`. This did not prevent the formal route from completing, but it
  remains a checkpoint-migration risk to audit separately.
- Status: Phase B single-capture Gain-consumer identity is `S4 observed`; do
  not promote long-training readiness. Mixed-K population, actor-update, and
  resume gates remain open.

## E68 - Actor-Warmup And Mixed-K Formal Sentinel (2026-07-17)

- Raw evidence: `formal_runtime_audit_actor_sentinel_20260717.txt`, completed
  at `iter=220/220` through the official `MODE=train` Segment route with
  `formal_runtime_audit=1`, 32 environments, and one update step per loop.
- Warmup: the route stayed `critic_only` through the configured 200 critic
  iterations, then entered `actor_warmup` at `phase_iter=0` with
  `actor_weight=0.002`, reaching `0.040` at the final iteration. Actor
  parameter updates were nonzero; this proves the ramp entry, not full actor
  weight or joint-RL behavior.
- PPO/trust: actor-warmup updates report finite loss/gradient/parameter delta,
  `trust_accepted=1`, `rejected=0`, and post-update KL below the configured
  `desired_kl=0.01` in the observed actor-warmup samples. Frozen GMT remains
  outside the optimizer (`gmt_trainable=0`, `gmt_in_optimizer=0`).
- Mixed K: later formal captures report effective `K` from `8` through `64`;
  `gain_steps=shape=(64,8), finite=1`, survival step-sum error remains `0.0`,
  and returns/advantages remain finite. Each update-loop diagnostic still
  reports one transaction and one batch with `same_transaction=True`.
- Persistence: `model_100.pt`, `model_200.pt`, and `model_220.pt` each report
  model, optimizer, normalizer, sampler, Gain config, and warmup payloads.
- Runtime status: no formal audit `finite=0`, `missing`, `NaN`, `Inf`,
  traceback, or uncaught exception was observed. Isaac Sim emitted a
  non-fatal `DriverShaderCacheManager` shutdown warning during teardown.
- Status: actor-warmup and mixed-K integration are `S4 observed`. Full actor
  weight/joint-RL, checkpoint resume, and post-training quality remain open;
  long training is not promoted by this sentinel alone.

## E69 - Full-Resume Formal Sentinel (2026-07-17)
- Raw evidence: `formal_runtime_audit_resume_sentinel_20260717.txt`, produced
  by the official `MODE=train` route with 64 environments, four update steps,
  `formal_runtime_audit=1`, and `is_full_resume=True`.
- Load identity: the runner loaded `model_220.pt` at checkpoint iteration 220,
  verified `FRS-GAIN-v002`, and restored the residual actor, critic, optimizer,
  prefix normalizer, Segment sampler, warmup config, noise std `0.01`, and DR
  scale `1.25`. GMT remained frozen.
- Schedule continuity: the resumed route reported `phase=actor_warmup`,
  `phase_iter=20`, and `actor_weight=0.042`; warmup did not restart at zero.
- Update continuity: four PPO updates consumed 16/16/14/15 valid policy rows.
  Every observed update had finite loss, nonzero parameter delta, accepted
  trust-region status, and post-update KL below `desired_kl=0.01`.
- Save identity: the loop completed `updates=4/4` and wrote `model_221.pt`
  with model, optimizer, observation normalizer, sampler, Gain config, and
  warmup payloads at absolute iteration 221.
- Diagnostic defect found: the old progress formatter printed `iter=221/1`
  by mixing absolute checkpoint iteration with this command's local iteration
  budget. The formatter now reports `absolute_iter=221 local=1/1`; an offline
  regression contract protects this display-only distinction.
- Fresh local verification: Python compilation passed; the live-training
  pseudo contract passed including the resume-progress fixture; the formal
  Runtime Atlas contract passed; the aggregate suite reported `44/44` with
  zero failures; Runtime Atlas rebuilt 22 owner cards; `git diff --check`
  passed.
- Quality boundary: aggregate `gain_total=-0.015758` and repaired MPJPE was
  worse than noisy MPJPE in this sentinel. These are quality observations, not
  resume-connectivity failures. E69 does not prove full actor weight, long-run
  learning quality, or policy superiority.
- Status: checkpoint/full-resume continuity is `S4 observed`. Full actor-weight
joint RL and policy-quality evaluation remain open.

## E70 - Full Actor-Weight Joint-RL Formal Sentinel (2026-07-17)
- Raw evidence: `formal_runtime_audit_joint_resume_20260717.txt`, produced by
  the official `MODE=train` route after full resume from `model_200.pt`.
- Joint boundary: at absolute iteration 700 the warmup owner emitted
  `phase=joint`, `phase_iter=0`, and `actor_weight=1.0`.
- Update evidence: the four joint-phase PPO batches contained
  `valid=13/14/16/16`; every batch reported nonzero actor parameter delta and
  `trust_accepted=1`. Post-update KL remained below the configured
  `desired_kl=0.01` for these four batches.
- Frozen GMT boundary: every joint-phase update reported `gmt_trainable=0`
  and `gmt_in_optimizer=0`.
- Persistence: `model_701.pt` was saved at iteration 701 with model,
  optimizer, observation normalizer, sampler, Gain config, and warmup payloads
  present.
- Limitation: E70 proves formal joint-phase connectivity, full actor weight,
  frozen-GMT gradient ownership, accepted PPO updates, and checkpoint
  persistence. It does not prove that the learned policy beats zero/HSL,
  improves Gain, avoids no-op, or generalizes.
- Status: Phase B formal runtime closure is `S4 observed`; policy efficacy is
  transferred to the independent Policy Quality Audit Q gates.
