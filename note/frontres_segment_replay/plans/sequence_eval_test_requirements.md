# Stage 3 Sequence Eval Test Requirements

Date: 2026-07-04

Revision: v2 module-by-module test audit plan.  This replaces the earlier
single-bug-first audit and treats sequence eval itself as the system under
test.

## Problem

The previous sequence-eval result shows suspiciously similar motion metrics
across different motion ids.  The evaluation code must be audited as a test
system before its result is used to judge training quality.

This plan checks the test code in small module-owned steps.  Each step has one
main boundary, one smallest useful contract/probe, and one stop condition.

Execution priority:

1. Verify local contracts for Modules A-F with fake runners and hand-checkable
   tensors.
2. Fix only the module whose contract fails.
3. Run a minimal live sentinel only after the local contracts pass.

## Functional Requirements

For every evaluated motion sequence, sequence eval must prove these facts:

1. `sequence_index -> motion_id -> segment/start frame` is unique and logged.
2. The env reset phase starts the selected motion at frame 0.
3. The preroll phase advances from frame 0 to the selected segment start.
4. The scoring rollout begins after preroll and does not score preroll frames.
5. The current experiment uses explicit rp-only perturbation:
   `stage3_index_perturbation_family=local_rp`.
6. Clean/reference, repaired, and noisy motion-quality tensors use the same
   coordinate semantics before MPJPE, velocity, and acceleration metrics.
7. Repaired and noisy metrics compare against the correct reference for the
   same motion/frame role, not a reused tensor from another sequence.
8. Per-item, per-motion, and final summaries aggregate the same semantic fields.
9. Action diagnostics, especially `delta_se_norm`, remain visible when all envs
   eventually fall.
10. Logs contain enough compact facts to distinguish a bad policy from a bad
    evaluator without printing full tensors.

## Local Patterns To Reuse

Use existing local test style; do not introduce a new pytest/fixture framework.

- FEMR `frontres_segment_live_probe_contract.py`: fake env event trace,
  reset request assertions, `_probe_tensor` logs.
- FEMR `frontres_segment_sequence_eval_contract.py`: sequence eval fake runner
  and item/per-motion/final log checks.
- MOSAIC `frontres_authority_event_mask_semantics.py`: tiny tensor semantics
  test with exact `valid_mask` assertions.
- MOSAIC `frontres_storage_algorithm_loss.py`: identifiable tensors routed
  through formal storage/update code.
- MOSAIC `frontres_reward_compute.py`: minimal `FakeRunner` surface for reward
  and diagnostics helpers.

Internet examples are only useful as live-smoke-test style references:
small env count, few steps, reset/step/finite checks.  They should not replace
the local contract tests above.

## Module Map

### Module A: Sequence Selection

Owner:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_sequence_eval.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`

Boundary:

`sampled segment -> motion_id/start_frame -> reset spec -> eval spec`.

Risk:

Different sequence rows can accidentally reuse one motion id, one frame, or one
previous reset spec, making all sequence results look similar.

### Module B: Reset And Preroll

Owner:

- `source/rsl_rl/rsl_rl/frontres/frontres_segment_reset.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`

Boundary:

`reset frame 0 -> preroll steps -> scoring start frame`.

Risk:

The evaluator can accidentally score the segment-start reset directly, or score
preroll frames, instead of evaluating after sequence-start preroll.

### Module C: Perturbation Family

Owner:

- `run_stage3.sh`
- `run/run_frontres_stage3_segment_hrl.sh`
- `scripts/rsl_rl/train.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py`

Boundary:

`FRONTRES_SPECIALIST_MODE=rp -> --frontres_specialist_mode rp -> local_rp reset
request -> env perturbation state`.

Risk:

The evaluator can silently use mixed/full perturbations or no perturbation,
making test results unrelated to the intended rp-only experiment.

### Module D: Motion Quality Capture

Owner:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py`

Boundary:

`command body positions / robot body positions -> comparable clean/repaired/noisy
tensors -> MPJPE/velocity/acceleration`.

Risk:

Cross-role env origin, mismatched coordinate frames, or wrong reference rows can
dominate MPJPE and make different motions look nearly identical.

### Module E: Rollout State Isolation

Owner:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`

Boundary:

`sequence loop item -> reset/preroll/read obs/eval -> capture summary`.

Risk:

Per-sequence tensors can be reused across items, or observations can be stale
after preroll, producing repeated metrics.

### Module F: Metric Aggregation

Owner:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
- `source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py`

Boundary:

`item summary -> per-motion rows -> final mean`.

Risk:

Different valid-mask rules can make item/per-motion/final disagree.  In the
current logs, `delta_se_norm` can become zero when all samples fall because the
valid mask removes every action sample.

## Step Plan

### Step 1: Motion-Quality Coordinate Frame

Scope:

- Verify that clean/repaired/noisy body tensors are comparable after capture.

Non-scope:

- Do not judge policy quality.
- Do not change reset/preroll or perturbation selection.

Files:

- Modify: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_capture_contract.py`

Owner module:

- Module D: Motion Quality Capture.

Core parameter path:

`body_pos_w/body_pos_relative_w/robot_body_pos_w -> captured tensors -> MPJPE`.

Test class:

- Core param path.

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_capture_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_motion_quality_capture_contract.py
```

Expected result:

- Identical body poses placed at different env origins produce near-zero MPJPE.

Status:

- Done.

Evidence:

```text
[probe motion_quality_capture] clean_max=0.4000000059604645 repaired_mpjpe=6.35782896551973e-07 noisy_mpjpe=0.0
result: PASS
```

Re-verified on 2026-07-05 with the same commands:

```text
[probe motion_quality_capture] clean_max=0.4000000059604645 repaired_mpjpe=6.35782896551973e-07 noisy_mpjpe=0.0
result: PASS
```

### Step 2: Sequence Selection And Reset Spec Identity

Scope:

- Prove each sequence item carries the intended `motion_id`, `segment_id`, reset
  frame, and eval start frame.

Non-scope:

- Do not run IsaacLab.
- Do not change motion-quality math.

Files:

- Modify/test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
- Modify only if the test exposes a bug:
  `source/rsl_rl/rsl_rl/runners/frontres_segment_sequence_eval.py`
  or `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`

Owner module:

- Module A: Sequence Selection.

Core parameter path:

`sequence sample -> reset spec(frame 0) -> eval spec(segment start)`.

Test class:

- Core param path.

Command:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- Fake runner event log proves each sequence has a unique `motion_id`.
- Reset spec uses frame 0.
- Eval spec uses the same `motion_id` and target segment start.

Stop condition:

- Printed probe shows `motion_id`, `reset_frame`, `eval_start_frame`, and event
  order for at least two distinct sequences.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step2] sequence_plan_identity motion_ids=('walk_a', 'walk_b', 'walk_c') segment_ids=[1, 3, 4] reset_frames=[0, 0, 0] eval_start_frames=[12, 0, 25]
[probe step2] reset_spec_identity segment_id=7 motion_id=walk_reset reset_frame=0 original_start_frame=31 horizon_k=4
[probe step2] sequence_owner_identity motion_ids=('motion_0', 'motion_1') reset_starts=[(0, 0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 0, 0)] eval_starts=[(3, 3, 3, 3, 3, 3, 3, 3), (4, 4, 4, 4, 4, 4, 4, 4)]
[probe step23] sequence_eval_contract unique_motion_ids=True reset_frame=0 eval_start_frame=segment_start segment_id_preserved=True requested_sequences_not_env_count=True
frontres_segment_sequence_eval_contract: ok
```

### Step 3: Reset-Preroll-Scoring Boundary

Scope:

- Prove preroll is executed before scoring and preroll frames are not included
  in captured eval metrics.

Non-scope:

- Do not test perturbation correctness.
- Do not test aggregate statistics.

Files:

- Modify/test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
- Modify only if needed:
  `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`

Owner module:

- Module B: Reset And Preroll.

Core parameter path:

`reset obs -> preroll rollout(capture_motion_quality=False) -> read obs ->
scoring rollout(capture_motion_quality=True)`.

Test class:

- Core param path.

Command:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- Event log exactly shows reset, preroll without capture, observation refresh,
  then scoring with capture.

Stop condition:

- Contract fails if capture is enabled during preroll or if observation refresh
  is skipped before scoring.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step3] reset_preroll_scoring_order events=['reset', 'read_obs', 'eval_mode', 'rollout', 'read_obs', 'rollout', 'reset', 'read_obs', 'eval_mode', 'rollout', 'read_obs', 'rollout'] preroll_capture=[False, False] scoring_capture=[True, True] scoring_obs=['obs_2', 'obs_4']
[probe step24] sequence_eval_live_owner reset_before_preroll=True preroll_no_capture=True obs_refresh_before_eval=True preroll_before_eval=True eval_capture=True reset_trace_silenced=True role_envs_repeated=True max_preroll_routed=True motion_metrics_printed=True perturbation_rp_preserved=True
frontres_segment_sequence_eval_contract: ok
```

### Step 4: Perturbation Family Contract

Scope:

- Prove rp-only configuration reaches sequence eval reset requests and env hook
  perturbation state.

Non-scope:

- Do not inspect policy performance.
- Do not add new perturbation modes.

Files:

- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py`
- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_stage1_env_hooks_contract.py`
- Modify only if needed:
  `run_stage3.sh`, `run/run_frontres_stage3_segment_hrl.sh`,
  `scripts/rsl_rl/train.py`,
  `source/rsl_rl/rsl_rl/frontres/frontres_segment_stage1_env_hooks.py`

Owner module:

- Module C: Perturbation Family.

Core parameter path:

`FRONTRES_SPECIALIST_MODE=rp -> CLI flag -> reset request family -> env applied
family`.

Test class:

- Core param path plus secondary launch contract.

Command:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_launch_command_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage1_env_hooks_contract.py
```

Expected result:

- Launch command contains `--frontres_specialist_mode rp`.
- Env hook receives only `local_rp`.

Stop condition:

- Contract prints rp-only route and rejects non-rp contamination for the current
  specialist eval path.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step7] stage3_sequence_eval_launch: stage3=True resume_stage1=True is_full_resume_false=True update_steps_3=True specialist_rp=True update_loop=False sequence_eval=True legacy_stage2=False mosaic_path=False
[probe step4] index_reset_applies_local_rp_only_perturbation family=('local_rp',) strength=[1.25] planar_mask=[False] yaw_mask=[False] global_z_mask=[False] local_rp_mask=[True]
frontres_segment_stage3_launch_command_contract: ok
PASS: FrontRES Stage 1 env adapter hooks trace motion, clean reset, perturbation, and baseline rollout.
```

### Step 5: Rollout State Isolation

Scope:

- Prove each sequence item creates fresh reset/preroll/eval state and does not
  reuse previous capture tensors.

Non-scope:

- Do not run long live eval.
- Do not change metric formulas.

Files:

- Modify/test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
- Modify only if needed:
  `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
  or `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`

Owner module:

- Module E: Rollout State Isolation.

Core parameter path:

`sequence index -> fake capture object id / reward tensor / survival tensor ->
item summary`.

Test class:

- Core param path.

Command:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- Two fake sequences with deliberately different rewards, falls, and motion
  tensors produce different item summaries.

Stop condition:

- Contract fails if sequence 2 reuses sequence 1 capture tensors or summary.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step5] rollout_state_isolation motion_0_repaired=0.220000 motion_1_repaired=0.240000 motion_0_survival=43.0 motion_1_survival=31.5 motion_0_delta=0.244949 motion_1_delta=0.489898
[probe step24] sequence_eval_live_owner reset_before_preroll=True preroll_no_capture=True obs_refresh_before_eval=True preroll_before_eval=True eval_capture=True reset_trace_silenced=True role_envs_repeated=True fresh_sequence_capture=True max_preroll_routed=True motion_metrics_printed=True perturbation_rp_preserved=True
frontres_segment_sequence_eval_contract: ok
```

### Step 6: Metric Aggregation And Valid-Mask Semantics

Scope:

- Prove item/per-motion/final aggregation uses one semantic rule and action
  diagnostics are not hidden by all-fall valid masks.

Non-scope:

- Do not change motion capture.
- Do not add new metrics beyond fields already logged.

Files:

- Modify/test: `source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py`
- Modify/test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`
- Modify only if needed:
  `source/rsl_rl/rsl_rl/frontres/frontres_segment_diagnostics.py`
  or `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`

Owner module:

- Module F: Metric Aggregation.

Core parameter path:

`delta_se/actions -> valid_mask -> item scalar -> per-motion row -> final scalar`.

Test class:

- Core param path.

Command:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_diagnostics_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- All-fall samples with nonzero actions still report nonzero action diagnostic.
- Per-item, per-motion, and final rows agree on aggregation semantics.

Stop condition:

- Contract fails on the current suspicious behavior where item-level
  `delta_se_norm` can become `0.0` solely because all envs fell.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step6] all_fall_motion_quality_action_visible mpjpe=0.000000 delta=0.625831 dz_up=0.500
[probe step6] sequence_aggregation_all_fall_action_visible item_delta=0.734847 per_motion_min=0.734847 final_delta=0.734847
result: PASS
frontres_segment_sequence_eval_contract: ok
```

### Step 7: Minimal Live Sentinel

Scope:

- Run one tiny real sequence eval after contract tests pass and inspect only
  boundary facts.

Non-scope:

- Do not use this as final performance evaluation.
- Do not run 10 long sequences.

Files:

- No code change expected.
- Log target under `/hdd1/cyx/FEMR/...` when run on server.

Owner module:

- End-to-end live sentinel across Modules A-F.

Core parameter path:

`launch command -> sequence log -> per-item facts`.

Test class:

- Live sentinel path.

Expected log facts:

- distinct `motion_id`
- `reset_frame=0`
- nonzero / varied `eval_start_frame`
- `family_counts={'local_rp': ...}`
- motion-quality fields not suspiciously constant by construction
- action diagnostics visible even on falls

Stop condition:

- If live sentinel violates a module contract, return to that module step
  instead of changing multiple areas at once.

Status:

- Preflight verified on 2026-07-05.
- Real live sentinel log reviewed on 2026-07-05.

Preflight evidence:

```text
[probe step7] stage3_sequence_eval_launch: stage3=True resume_stage1=True is_full_resume_false=True update_steps_3=True specialist_rp=True update_loop=False sequence_eval=True legacy_stage2=False mosaic_path=False
[FrontRES Stage3 startup preflight] PASS mode=sequence_eval
Command: python scripts/rsl_rl/train.py ... --frontres_specialist_mode rp ... --frontres_segment_sequence_offline_eval_only --frontres_segment_sequence_eval_sequences 2 --frontres_segment_sequence_eval_max_preroll_steps 120 --frontres_segment_offline_eval_steps 120
```

Server command:

```text
CUDA_VISIBLE_DEVICES=3 \
CACHE_DIR=/hdd1/cyx/AMASS_G1Segment \
LOG_PATH=/hdd1/cyx/FEMR/stage3_sequence_eval_sentinel_seq2_step120_preroll120.txt \
FRONTRES_SPECIALIST_MODE=rp \
OFFLINE_EVAL_SEQUENCES=2 \
OFFLINE_EVAL_STEPS=120 \
OFFLINE_EVAL_MAX_PREROLL_STEPS=120 \
RUN_FOREGROUND=1 \
bash /hdd1/cyx/FEMR/run_stage3.sh \
  /hdd1/cyx/FEMR/model/model_27000.pt \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  4 \
  1 \
  1 \
  sequence_eval
```

Server log checklist:

```text
rg -n "FrontRES Segment Sequence Eval|Sequence Eval Item|motion_id=|reset_frame=|eval_start_frame=|family_counts=|delta_se_norm=|non_rp_frac|Traceback|ERROR|CUDA out of memory|nan|inf" /hdd1/cyx/FEMR/stage3_sequence_eval_sentinel_seq2_step120_preroll120.txt
```

Live log evidence:

```text
log: /Users/chengyuxuan/ArtiIntComVis/stage3_sequence_eval_sentinel_seq2_step120_preroll120.txt
[FrontRES Stage] ... segment_sequence_eval=True ... segment_train=False ... is_full_resume=False
[INFO] FrontRES perturbation alignment: rp (root_tilt=0.5/0.08, iid_rp=0.4/0.08; xy/yaw/z/joint disabled)
[FrontRES Segment Sequence Eval Plan] max_preroll_steps=120 ...
[FrontRES Segment Sequence Eval Progress] sequence=1/2 motion_id=KIT/883/amass_g1_wipe_back_horizontal02_poses_reflect.npz reset_frame=0 preroll_steps=106 eval_steps=120
[FrontRES Segment Sequence Eval Progress] sequence=2/2 motion_id=CMU/29/amass_g1_29_04_poses_reflect.npz reset_frame=0 preroll_steps=8 eval_steps=120
perturbation: family_counts={'local_rp': 4} strength_min=1.250000 strength_mean=1.250000 strength_max=1.250000 local_rp_frac=100.0% non_rp_frac=0.0%
final: success=37.5% fall=62.5% survival=92.6 gain=-0.031203 delta_se_norm=0.095452
```

Live verdict:

- Verified: launch entered sequence eval, two distinct motion ids were sampled,
  reset frame stayed 0, eval start frames differed, perturbation was rp-only,
  action diagnostics stayed visible, and no Python traceback/OOM was found.
- Weak: per-motion rows are not directly comparable with item/final rows yet.
  Item/final summarize all 4 role envs, while per-motion rows expose a
  single-row motion view, so success/fall and `delta_se_norm` can differ by
  aggregation scope.

### Step 8: Per-Motion Aggregation Scope

Scope:

- Make per-motion rows use the same aggregate scope as the sequence item when
  all role envs in that item belong to one motion.

Non-scope:

- Do not change rollout, perturbation, score construction, motion capture, or
  training.

Files:

- Modify: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
- Modify: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`
- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`

Owner module:

- Module F: Metric Aggregation.

Core parameter path:

`capture done/actions -> item summary -> per-motion row -> final row`.

Test class:

- Core param path.

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
```

Expected result:

- For a same-motion 4-role sequence item, item and per-motion rows report the
  same success, fall, survival, and `delta_se_norm` values.

Stop condition:

- Contract fails if per-motion rows fall back to single-row role stats.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step8] per_motion_scope_matches_item success=0.250 per_motion_success=0.250 delta=0.489898 per_motion_delta=0.489898
frontres_segment_sequence_eval_contract: ok
```

### Step 9: Sequence Eval Runtime Debug Block

Scope:

- Print enough runtime parameters after each sequence eval item to diagnose
  policy output, rp perturbation routing, reset/preroll frame identity, reward
  pairing, action magnitude, role masks, and motion-quality metrics from one
  live log.

Non-scope:

- Do not change rollout, policy, perturbation sampling, score construction,
  motion capture, aggregation math, training, or checkpoint behavior.

Files:

- Modify: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`

Owner module:

- Module E/F: Rollout State Isolation and Metric Aggregation.

Core parameter path:

`plan/reset batch -> reset request/result -> scoring obs -> capture action/reward/motion tensors -> item summary`.

Test class:

- Live sentinel path with contract coverage for printed parameter categories.

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- Each sequence item prints `[FrontRES Segment Sequence Eval Debug]` with
  plan, eval/reset batch, reset request/result, observation shape/value summary,
  capture roles/shapes, reward pairs, done/survival, action rows, transition
  distribution tensors, motion tensors, per-role motion errors, and summary.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step9] sequence_eval_debug_log_covers_runtime_parameters=True
frontres_segment_sequence_eval_contract: ok
```

## Step 10 - Add Sequence Eval Oracles and Differential Proxy

Scope:

- Extend the existing `[FrontRES Segment Sequence Eval Debug]` block with
  searchable oracle booleans and differential proxy values for sequence
  evaluation review.

Non-scope:

- Do not add extra rollout passes, change policy behavior, change perturbation
  sampling, change metrics, or alter training.

Files:

- Modify: `source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py`
- Test: `source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py`

Owner module:

- Module E/F: Rollout State Isolation and Metric Aggregation.

Core parameter path:

`motion_id/reset_frame/preroll/eval_frame/perturb_family/action/metric roles`.

Test class:

- Core param path plus live sentinel debug snapshot.

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/runners/frontres_segment_live_training.py source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_sequence_eval_contract.py
```

Expected result:

- The debug block prints `oracles:` with reset/eval-frame/motion/rp-only/role
  alignment booleans.
- The debug block prints `differential_proxy:` with action nonzero flags,
  repaired-vs-noisy score delta, MPJPE delta, and reward-pair gain mean.

Status:

- Done on 2026-07-05.

Evidence:

```text
[probe step10] sequence_eval_debug_log_covers_oracles_and_differential_proxy=True
frontres_segment_sequence_eval_contract: ok
frontres_segment_live_probe_contract: ok
```

## Execution Rule

Execute only one step at a time.  After each step:

1. Run its listed command.
2. Record the observed probe line in this note.
3. State whether the boundary is verified, failed, or still weak.
4. Only then move to the next step.
