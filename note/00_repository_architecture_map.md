# FEMR Current Repository Architecture

This document records current ownership only. Historical designs belong under
`note/frontres_core/contracts/history/` and must not be reconstructed here.

## 1. Active Method Route

```text
Stage 1 segment cache
  -> AMASS segment identity + Clean/Noisy dynamic state
Stage 2 HSL
  -> initialize one full-6D Delta SE(3) repair actor
Stage 3 Segment Replay
  -> sample global/replay/review segments
  -> construct Noisy/Repair local counterfactuals
  -> dynamic reset or faithful preroll
  -> one repair at t, then frozen-FEMR GMT Clean continuation
  -> paired intent gain + physics gain - repair regularizer
  -> direct full-6D PPO update
  -> rollout-evidence-only segment-priority update
  -> checkpoint / periodic eval / sequence eval
```

The active policy output is exactly:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Perturbation family does not mask policy dimensions. Confidence columns,
`active_task_dims`, alpha/rho routing, and proposal-conditioned authority are
historical designs and are not current architecture.

## 2. Entry And Configuration

- `scripts/rsl_rl/train.py`: mode dispatch and training/evaluation entry.
- `scripts/rsl_rl/cli_args.py`: CLI surface.
- `run_stage3.sh`: Stage 3 command wrapper and argument forwarding.
- `source/whole_body_tracking/.../config/g1/agents/rsl_rl_mosaic_cfg.py`:
  experiment defaults, FrontRES stage, Segment Replay, PPO, and diagnostics.

Every active feature must be traceable from config through the formal runtime
route. A helper or test-only implementation is not an integrated feature.

## 3. Environment, Observation, And Perturbation

- `whole_body_tracking/tasks/tracking/mdp/commands.py`: frozen-GMT reference
  command and motion phase/state.
- `mdp/observations.py`: tracking observations.
- `mdp/balance.py`: ZMP/support/balance context.
- `mdp/motion_perturbations.py`: environment-side reference artifacts.
- `rsl_rl/modules/frontres_observation_layout.py`: legacy 870D actor layout;
  v015 requires current Noisy root error plus future q29-intent augmentation.
- `rsl_rl/frontres/frontres_dr_curriculum.py`: perturbation strength/family
  curriculum.
- `rsl_rl/frontres/perturbation_runtime.py`: runtime perturbation payload.

The current experiment enables `local_rp`; this labels the artifact source but
the repair actor remains full 6D. Prefix statistics and frozen-GMT suffix
statistics have separate checkpoint ownership.

## 4. Stage 1 Segment Cache

- `frontres_segment_cache_builder.py`: cache build orchestration.
- `frontres_segment_cache_indexer.py`: AMASS segment indexing.
- `frontres_segment_cache_schema.py`: manifest and shard schema.
- `frontres_segment_cache_io.py`: resumable and atomic shard IO.

The cache stores replayable dynamic state, not only pose. Manifests are the
reader contract and build signatures are the resume source of truth.

## 5. Policy, Proposal-Only HSL, And Runtime Write

- `rsl_rl/modules/front_residual_actor_critic.py`: full-6D residual policy and
  distribution statistics.
- `rsl_rl/runners/frontres_warmup.py`: sealed q29 Stage-1 actor input and current anti-DR target validation.
- `rsl_rl/runners/frontres_hsl_rollout_target.py`: legacy Stage-3 rollout label reject-only boundary.
- `rsl_rl/algorithms/frontres_unified.py`: v015 rejects nonzero Stage-3 online HSL loss.
- `rsl_rl/runners/frontres_rollout_step.py`: v015 rejects direct Stage-3 HSL target writes.
- `rsl_rl/frontres/task_space_correction.py`: Delta SE(3) application.
- `rsl_rl/frontres/frontres_action_cone.py`: named physical execution bounds,
  including upward-dz safety; it does not own family masks.
- `rsl_rl/runners/frontres_rollout_step.py`: policy action production.
- `rsl_rl/runners/frontres_runtime.py`: correction write into the reference
  consumed by frozen GMT.

Stage 2 and Stage 3 share the same six-dimensional actor interface. Sampled
action, stored action, old mean/sigma, log probability, and executed correction
must use one representation.

## 6. Segment Replay Data And Sampling

- `frontres_segment_dataset.py`: lazy cache loading and semantic segment batch.
- `frontres_segment_reset.py`: reset request/result and dynamic-state boundary.
- `frontres_segment_stage1_env_hooks.py`: command-side segment reset adapter.
- `commands.py::MultiMotionCommand.refresh_frontres_reference_cache_current_frame()`:
  command-owned current-frame cache construction shared by ordinary command
  updates and Segment index reset; only the ordinary update advances the frame.
- `frontres_segment_sampler.py`: global/replay/review source selection,
  priority, segment state, trial planning, and `8/16/32/64` horizon assignment.

The sampler owns row expansion and rollout budget. Under v015, each local
scenario seals x_t, one current root artifact, future q29 intent, and a Clean
continuation. Policy rows may become PPO rows; search/counterfactual rows remain
replay evidence. K measures one first action only and is integrated only when
its frozen-FEMR lifecycle reaches reset, rollout, return, sampler update, and
live diagnostics.

## 7. Formal Stage 3 Runner Route

- `frontres_segment_live_sampler.py`: current trial/quartet assembly; v015
  requires two-role local-scenario assembly.
- `frontres_segment_live_probe.py`: reset, per-row K rollout, paired score
  capture, valid-policy gate, Segment storage write, and evidence summary.
- `frontres_segment_live_update_loop.py`: repeated probe/update orchestration.
- `frontres_segment_live_training.py`: formal training loop, checkpoint cadence,
  periodic evaluation, and offline sequence-evaluation entry.
- `on_policy_runner.py`: thin dispatch surface to these helpers.

Step 4B adds a deliberately separate CPU fake path:

```text
OnPolicyRunner.run_frontres_v015_formal_transaction()
-> injected request provider
-> run_frontres_v015_formal_transaction_update_loop()
-> sealed v015 plan/accumulator
-> grouped v003 loss
-> exactly one explicit optimizer step
```

`frontres_segment_live_sampler.py` owns that plan/accumulator and
`frontres_segment_live_probe.py` owns the one update. The generic `learn`,
live-training loop, checkpoint route, and simulator do not dispatch it.

Step 4C adds a fake-S3 persistence boundary without changing that dispatch:
`frontres_segment_live_update_loop.py` opens a `collecting` barrier before the
injected provider; `frontres_segment_live_probe.py` publishes only a committed
exact-one-update receipt; and `frontres_checkpointing.py` owns the versioned
q29 H/prefix-normalizer/grouped-loss identity. In-flight work cannot be saved
or resumed, and a committed resume restarts idle without raw scenario or batch
state. This is CPU-fake S3 evidence, not generic checkpoint/live routing.

Runner modules orchestrate. They do not own the priority formula or PPO loss.
Trial metadata reaches reset/evidence/diagnostics but does not enter the PPO
tuple.

## 8. Repair Quality And Gain

The active v015 design is:

```text
intent_gain  = internal_fidelity(Repaired | I_noisy)
             - internal_fidelity(Noisy | I_noisy)
physics_gain = physics_quality(Repaired)       - physics_quality(Noisy)
gain_total   = w_intent * intent_gain + w_physics * physics_gain
             - w_repair * repair_cost
```

- `frontres_gain.py`: required v003 paired Intent/Physics/Repair component
  owner and named scales/weights. Intent uses root-invariant 29DoF articulated
  fidelity rather than full Clean global motion. Physics includes paired success/survival,
  ZMP/support margin, and a documented foot-height contact proxy. Repair Cost
  uses executed full-6D actions with per-row K/done masks and optional Clean
  no-op diagnostics; missing runtime inputs remain `UNCONFIRMED`.
- `frontres_segment_live_probe.py`: captures the post-override executed
  full-6D transition action steps and paired K-step Gain; formal policy rows
  use `gain_total`/`gain_steps` for storage returns, and missing formal Gain
  evidence rejects the legacy score fallback.
- `frontres_executability.py`: legacy family-specific executability evidence;
  it is excluded from the active PPO, sampler, diagnostics, and evaluation
  route and is no longer a formal Gain owner.
- `frontres_segment_reward.py`: legacy/general score-window API retained only
  for compatibility; it is excluded from the active Gain route.
- `contracts/active/reward/FRS-GAIN-v003-intent-physics-local-repair.md`:
  accepted intent/physics/repair semantics.

Generic environment reward, teleoperation reward, velocity-command reward, and
unrelated task reward are excluded from the active Gain route by design. The
target architecture excludes the legacy score from PPO, sampler, diagnostics,
and evaluation, but the 2026-07-13 Step 6C audit found that diagnostics and
  periodic and sequence evaluation now consume the shared Gain owner in the
  offline formal route. `reward_accum` remains only as explicitly labeled raw
  debug input; it is not an accepted evaluation metric.

Training and evaluation must share component functions, units, signs, scales,
and K-step aggregation. Current code still implements v002 Clean-global Style,
the 65D tape route, and quartet roles; those paths are v015 contract-mismatch
until later gates migrate or isolate them.

## 9. Storage, PPO, And Priority

- `frontres_segment_storage.py`: independent Stage 3 PPO tuple, per-row K
  returns/done/valid masks, plus the v015 immutable local metadata adapter. It
  seals transaction/snapshot/motion/Segment/trial with scenario/hash/`x_t`/q29/K/
  evidence identity; the legacy `to_ppo_batch()` explicitly rejects that v015
  carrier.
- `frontres_segment_live_sampler.py`: fake-S2 expected-row plan validates the
  full multi-Segment x M identity before any update and aggregates only candidate
  adapter shards.
- `frontres_segment_live_probe.py`: fake-S2 validates q29/HSL/normalization
  isolation, calls unchanged grouped v003 PPO once, and requires one explicit
  optimizer counter increment. This is not the generic formal training route.
- `frontres_segment_ppo.py`: direct full-6D clipped surrogate, exact old/new
  distribution KL, post-update trust-region diagnostics, optimizer step, and
  sign-preserving scale-only advantage scaling.
- `frontres_segment_sampler.py`: rollout-evidence priority and persistent replay
  state. `frontres_segment_live_sampler.py` must migrate to the shared
  `FRS-GAIN-v003` result at the evidence boundary. Existing v002 consumer
  evidence is historical only and cannot prove v015 semantics.

PPO and sampler consume different roles of the same rollout-time Gain evidence.
PPO consumes source-consistent policy tuples; sampler consumes the shared Gain
result for priority. Post-update KL, parameter deltas, and logger state must
not influence segment priority. The former score route is legacy and is
excluded from the active route; migration isolation is covered by E14/E15/E16.
`E-FI-14` and `E-FI-15` establish CPU-fake exact-one-update and persistence
atomicity respectively. Generic formal dispatch, actual checkpoint cadence/
resume, simulator, and live runtime remain separate gates.

## 10. Checkpoint, Evaluation, And Diagnostics

- `frontres_checkpointing.py`: formal `OnPolicyRunner` Stage 2/GMT/FrontRES
  policy, normalizer, optimizer, sampler, and Gain-identity save/load owner;
  under v015 fake-S3 it additionally owns exact q29 H/prefix-normalizer/
  grouped-loss identity and rejects partial transaction persistence.
- Detached Segment checkpoint compatibility helpers were removed in Step 10C-C1;
  the only active checkpoint owner is `frontres_checkpointing.py`.
- `frontres_segment_sequence_eval.py`: sequence plan and preroll/eval boundary.
- `frontres_runner_logging.py` and Segment runner helpers: live diagnostics.

Evaluation samples independently from training state and reports motion id,
start frame, perturbation family/strength, reset/preroll status, survival,
paired gain, motion quality, and full-6D action evidence. Missing evidence is
`UNCONFIRMED`, never zero.

The live train diagnostic route now consumes the canonical Gain decomposition
through `frontres_segment_live_probe.py` ->
`frontres_segment_live_update_loop.py` ->
`frontres_segment_diagnostics.py`; legacy score scalars are isolated from this
route. Periodic evaluation now uses
`frontres_segment_live_training.py` -> `_capture_paired_gain` and is
`implemented-not-integrated` for S4; sequence evaluation now shares the same
owner for item/per-motion/aggregate summaries, with S4 population unconfirmed.

## 11. Test Ownership And Acceptance

Tests live under `source/rsl_rl/rsl_rl/tests/`.

Every feature has two distinct gates:

1. Implementation: the owner module represents and processes the intended
   semantics under contract/pseudo tests.
2. Integration: the formal training route calls the mechanism and propagates
   its outputs to the intended consumer.

Runtime-dependent claims additionally require a live sentinel. The aggregate
entry is `frontres_segment_all_contract_suite.py`; full-6D/no-mask semantics are
guarded by `frontres_full6_no_active_mask_contract.py`.

## 12. Note Ownership

- `frontres_core/contracts/README.md`: active contract registry.
- `frontres_core/contracts/active/`: current method/training/optimization/eval
  truth.
- `frontres_core/contracts/history/`: versioned historical designs only.
- `frontres_core/plans/` and `frontres_core/checklists/`: disposable current
  task state, refreshed rather than accumulated.
- `architecture/`: current code/module/runtime maps; no historical narrative.
- `artifacts_mining/`: dataset mining evidence.
- `software_engineering/`: user-owned reading material.

When active contracts, this map, and code disagree, stop and report the exact
conflict. Do not use a historical design to fill the gap.
