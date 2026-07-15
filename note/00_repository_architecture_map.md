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
  -> construct trial/quartet rows
  -> dynamic reset or faithful preroll
  -> paired Noisy/Repaired K-step frozen-GMT rollout
  -> paired style gain + physics gain - repair regularizer
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
- `rsl_rl/modules/frontres_observation_layout.py`: 100D FrontRES prefix, 770D
  GMT suffix, and 870D actor layout.
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

## 5. Policy, Stage 2 HSL, And Runtime Write

- `rsl_rl/modules/front_residual_actor_critic.py`: full-6D residual policy and
  distribution statistics.
- `rsl_rl/runners/frontres_hsl_rollout_target.py`: full-6D supervised target.
- `rsl_rl/algorithms/frontres_unified.py`: Stage 2 HSL update.
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
- `frontres_segment_sampler.py`: global/replay/review source selection,
  priority, segment state, trial planning, and `8/16/32/64` horizon assignment.

The sampler owns row expansion and rollout budget. Policy rows may become PPO
rows; search/counterfactual rows remain replay evidence. K-step curriculum is
integrated only when assigned K reaches reset, rollout, return, sampler update,
and live diagnostics.

## 7. Formal Stage 3 Runner Route

- `frontres_segment_live_sampler.py`: sample and trial/quartet batch assembly.
- `frontres_segment_live_probe.py`: reset, per-row K rollout, paired score
  capture, valid-policy gate, Segment storage write, and evidence summary.
- `frontres_segment_live_update_loop.py`: repeated probe/update orchestration.
- `frontres_segment_live_training.py`: formal training loop, checkpoint cadence,
  periodic evaluation, and offline sequence-evaluation entry.
- `on_policy_runner.py`: thin dispatch surface to these helpers.

Runner modules orchestrate. They do not own the priority formula or PPO loss.
Trial metadata reaches reset/evidence/diagnostics but does not enter the PPO
tuple.

## 8. Repair Quality And Gain

The accepted design is:

```text
style_gain   = style_quality(Repaired | Clean) - style_quality(Noisy | Clean)
physics_gain = physics_quality(Repaired)       - physics_quality(Noisy)
gain_total   = w_style * style_gain + w_physics * physics_gain
             - w_repair * repair_cost
```

- `frontres_gain.py`: shared paired Style/Physics/Repair component owner and
  named scales/weights. Style includes body MPJPE, velocity, acceleration, and
  root-orientation geodesic error. Physics includes paired success/survival,
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
- `contracts/active/reward/FRS-GAIN-v001-style-physics-repair.md`: accepted
  style/physics/repair semantics.

Generic environment reward, teleoperation reward, velocity-command reward, and
unrelated task reward are excluded from the active Gain route by design. The
target architecture excludes the legacy score from PPO, sampler, diagnostics,
and evaluation, but the 2026-07-13 Step 6C audit found that diagnostics and
  periodic and sequence evaluation now consume the shared Gain owner in the
  offline formal route. `reward_accum` remains only as explicitly labeled raw
  debug input; it is not an accepted evaluation metric.

Training and evaluation must share component functions, units, signs, scales,
and K-step aggregation. Formal policy-row training, sampler priority/state, and
periodic evaluation are connected to the shared decomposition and tested
offline. Real component population and persistence still require later gates
before this route is training-ready.

## 9. Storage, PPO, And Priority

- `frontres_segment_storage.py`: independent Stage 3 PPO tuple, per-row K
  returns, done masks, and valid mask.
- `frontres_segment_ppo.py`: direct full-6D clipped surrogate, exact old/new
  distribution KL, post-update trust-region diagnostics, optimizer step, and
  sign-preserving scale-only advantage scaling.
- `frontres_segment_sampler.py`: rollout-evidence priority and persistent replay
  state. `frontres_segment_live_sampler.py` now requires the shared
  `FRS-GAIN-v001` result at the evidence boundary, and useful/state/priority
  consumers now use canonical Gain plus validity/fall/contact/horizon facts;
  cross-consumer S2 acceptance is closed by E14/E15/E16; the remaining boundary
  is real S4 population.

PPO and sampler consume different roles of the same rollout-time Gain evidence.
PPO consumes source-consistent policy tuples; sampler consumes the shared Gain
result for priority. Post-update KL, parameter deltas, and logger state must
not influence segment priority. The former score route is legacy and is
excluded from the active route; migration isolation is covered by E14/E15/E16.

## 10. Checkpoint, Evaluation, And Diagnostics

- `frontres_checkpointing.py`: formal `OnPolicyRunner` Stage 2/GMT/FrontRES
  policy, normalizer, optimizer, sampler, and Gain-identity save/load owner.
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
