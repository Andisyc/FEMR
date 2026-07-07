# Stage 3 Runtime Distribution Checklist

Use this checklist for the current Segment Replay HRL path.  Older rho,
authority, and acceptance checklists are historical unless a run explicitly
enters those ablation branches.

## Active Scope

- Current method: one policy directly outputs 6D `Delta SE(3)` repair actions.
- Current training route: Stage 3 Segment Replay HRL with index-only dynamic
  perturbation and PPO update.
- Method boundary: direct 6D `Delta SE(3)` is the current first-order executable
  closure of the older HSL proposal / HRL acceptance idea.  Direction,
  authority, and no-op are compressed into one action surface:
  direction = repair direction, magnitude = implicit authority, zero action =
  do-not-repair.
- The current theoretical boundary is partial observability / state aliasing:
  the actor sees only deployable 870D observations, while Clean reference and
  exact artifact cause are training/evaluation facts.  Therefore Stage 3 should
  optimize no-regret residual authority, not exact Clean-reference
  reconstruction.
- Current suspicion: policy outputs can become much stronger than the
  supervised anti-perturbation target, especially under the Stage 3
  perturbation curriculum.

## Non-Scope

- Do not reintroduce rho/authority/acceptance as the active correctness
  standard.
- Do not treat full 6D output under `local_rp` perturbation as a bug by itself.
- Do not add a multi-signal online adaptive sampler as the next default fix.
  Keep sampling as one simple frontier-envelope strategy unless offline/log
  evidence proves fixed config-level bucket weights are insufficient.
- Do not launch another expensive live run before checking pure distribution
  math offline when the question is sampler/curriculum-only.
- Do not judge Stage 3 success from survival or scalar environment reward alone.
  High-frequency residual oscillation, persistent leaning, lateral drift, or
  strong clean-frame edits are method failures even if the rollout survives.
- Do not feed Clean-derived correction into actor observations.  Clean may be
  used for targets, counterfactual evaluation, and offline diagnostics only.

## Current Open Questions

### Method / Training Mechanism

- Does short K-step reward make strong immediate correction look good, while
  longer K-step reward would expose accumulated-state recovery and favor weaker
  or delayed repair?
- Does PPO encourage strong intervention because the action distribution is
  rewarded on marginal rollout gain and has no explicit no-regret or magnitude
  boundary for low-need states?
- Does the current reward allow reward hacking: strong residuals, high-frequency
  left-right oscillation, persistent root lean, or lateral drift that improve a
  scalar score while looking unnatural in Demo?
- Are clean / near-clean samples explicitly protected so that the learned
  default action remains no-op instead of "always repair"?
- Does the training distribution overrepresent high-strength repair and teach
  a default over-strong action?

### Pure Curriculum Distribution

- Does Stage 3 `dr_min_scale=1.25` remove true low-perturbation samples from
  training?
- Do easy/frontier/hard factors collapse near high frontier scale because of
  min/max clamps?
- Does the current single frontier-envelope sampler provide enough low/no-op,
  mid, and capped hard exposure without adding a second adaptive controller?
- If conflicts appear between easy safety and hard mining, treat no-op safety
  and survival as boundary constraints, not as equal-weight sampling objectives.

### Engineering Consistency

- Does the 770-dim GMT proprioceptive suffix get carried inside an 870-dim FEMR
  observation only because of the 100-dim FrontRES-only prefix, and can the code
  keep this split explicit?
- Are Stage 3 extra-prefix mean/std statistics loaded? If not, the 100-dim
  FrontRES-only prefix is passed through raw while the 770-dim GMT suffix is
  normalized by the frozen GMT normalizer.
- Does the policy still expose a 12D action surface while Segment Replay storage
  only consumes the first 6D Delta SE action?
- Is supervised target construction in bounded Delta space while policy log-prob
  and mean diagnostics are interpreted in raw action space?

### Privileged Information Boundary

- Should perturbation strength be an actor input? This is deliberately undecided:
  true strength is privileged information. If used by the actor during training,
  the result becomes a teacher policy and would need a distillation story before
  deployment. Prefer first checking whether deployment-visible anchor-error
  history already contains enough strength information.

## Offline Numerical Simulation First

Before adding live snapshots for curriculum distribution questions, create a
small deterministic test module or script that simulates the same config values
and helper functions:

- input: Stage 3 config values, frontier scale samples, env count, and seed;
- owner: `frontres_dr_curriculum.py` and the Stage 3 sampler helpers;
- output: easy/frontier/hard fractions, strength min/p10/p50/p90/max/mean, and
  clamp-hit fractions;
- invariant: reported distribution explains whether true low-strength samples
  are present before any IsaacLab rollout is launched.

First offline simulation target:

- owner: `source/rsl_rl/rsl_rl/frontres/frontres_dr_curriculum.py`;
- inputs: current Stage 3 `dr_min_scale`, `dr_max_scale`, easy/frontier/hard
  weights and factors, env count, frontier scales from low to high;
- outputs: strength min/p10/p50/p90/max/mean, easy/frontier/hard fractions, and
  min/max clamp-hit fractions;
- decision: if true low perturbations are absent, fix curriculum before adding
  live action/reward probes.

## Live Probe Only After Offline Simulation

Use a live probe only for values that require the real runner, environment, or
policy:

- reset request family and strength actually reaching env hooks;
- actor observation prefix normalization status;
- target/action/reward/advantage by strength bucket;
- parameter delta after optimizer update;
- checkpoint/resume behavior.

## Sampling Design Decision

Do not add a separate multi-signal adaptive distribution controller as the next
default mechanism.  Similar mature designs use simple sampling strategies:
prioritized replay / prioritized level replay, automatic domain randomization,
or curriculum difficulty expansion.  The current Segment Replay path already
has the needed simple adaptive component:

```text
global sampling   -> coverage / unseen segments
replay sampling   -> prioritized useful learning signal
review sampling   -> solved segment retention
```

Use the existing Segment Replay priority replay for hard-sample reuse.  Keep DR
strength distribution simple:

```text
bucket ranges follow the probed frontier envelope
bucket weights stay config-level unless evidence proves otherwise
segment-level replay priority handles adaptive difficult-sample reuse
```

This avoids turning sampling into a second method while still getting adaptive
hard-example replay.

## Reward-Hacking Boundary And Demo Gate

The active Stage 3 reward should be treated as a no-regret residual-authority
objective, not as an unconstrained survival objective:

```text
main evidence:
  repaired_vs_noisy_gain

authority bounds:
  residual magnitude
  residual action-rate / temporal smoothness
  residual jerk or second difference if action-rate alone misses oscillation

clean protection:
  action near zero on clean / near-clean references
  repaired clean rollout not worse than Clean/GMT or no-op GMT

demo-quality guards:
  action RMS
  action-rate RMS
  root roll/pitch long-window bias
  lateral drift / persistent side lean
  fall rate and survival
  repaired-vs-noisy and repaired-vs-clean visual inspection buckets
```

The next reward design should prefer a small set of boundary terms before
adding new architecture:

```text
R_stage3 =
  gain(repaired, noisy)
  - lambda_mag * ||Delta SE||^2
  - lambda_rate * ||Delta SE_t - Delta SE_{t-1}||^2
  - lambda_clean * clean_or_low_need_weight * ||Delta SE||^2
  - lambda_bias * long_window_root_bias_or_drift
```

Implementation caution:

- `lambda_bias` should start as a diagnostic or weak guard.  It can easily
  suppress legitimate recovery if it is too strong.
- Action-rate / residual-smoothness is the lowest-risk first reward addition
  because it directly targets Demo-visible high-frequency shaking.
- Clean no-op protection is the most important conceptual guard because it
  prevents the policy from treating every reference as corrupted.
- Do not reuse GMT robot-action penalties blindly as FrontRES penalties.  A GMT
  torque or robot action-rate spike may be caused by tracker dynamics rather
  than the FrontRES residual itself; prefer penalties on the 6D residual first,
  then keep robot-side torque/action-rate as diagnostics.

External design references to keep in mind:

- action smoothness regularization in robot RL, e.g. CAPS-style temporal and
  spatial smoothness penalties;
- physics-based character imitation such as DeepMimic separates task success
  from pose/velocity/end-effector/root imitation quality;
- adversarial motion prior work treats natural motion quality as a separate
  style/evidence channel rather than relying on survival alone;
- existing IsaacLab/GMT reward practice already includes action-rate, joint
  acceleration, and torque penalties, but Stage 3 should first regularize the
  residual action that FEMR actually controls.

## Demo Protocol

Do not choose Demo clips by survival only.  Each selected sequence should have
the same fixed comparison layout:

```text
Clean/GMT | Noisy/GMT | Repaired/GMT
```

Required Demo buckets:

- clean or near-clean: Repaired should stay close to no-op;
- repairable noisy: Repaired should improve over Noisy without obvious shaking;
- frontier hard: Repaired should improve or at least not clearly worsen Noisy;
- failure case: show the boundary if both Noisy and Repaired fail.

Minimum searchable metrics to print before recording:

```text
sequence_id
perturbation_family / strength
gain(repaired, noisy)
fall / survival
action_rms
action_rate_rms
root_roll_pitch_bias
lateral_drift
clean_noop_action_rms when applicable
```

## Priority Order

1. Offline simulate Stage 3 strength distribution.
2. Audit and fix observation normalization/loading for the 30-dim extra prefix.
3. Remove or hard-gate unused 12D action-surface behavior for the active Segment
   Replay HRL path, while preserving explicit ablations.
4. Add reward-hacking diagnostics for residual magnitude, residual action-rate,
   clean no-op, long-window root bias, and lateral drift.
5. Add the smallest reward boundary terms only after diagnostics show which
   Demo-visible failure is present.
6. Re-evaluate K-step horizon and PPO magnitude pressure after the engineering
   route is clean.
7. Decide perturbation-strength input only after proving deployment-visible obs
   is insufficient.

## Stop Condition For The Next Step

Before adding new reward terms, a short eval/train diagnostic should print the
reward-hacking boundary metrics for the same sequence set:

- `gain(repaired, noisy)`;
- `fall` and `survival`;
- `action_rms` and `action_rate_rms`;
- clean / near-clean `action_rms`;
- long-window root roll/pitch bias;
- lateral drift or persistent side lean;
- perturbation family and strength bucket.

Decision rule:

- if shaking is visible or `action_rate_rms` is high, add residual action-rate
  smoothness first;
- if clean / near-clean samples receive nonzero repair, add clean no-op
  protection first;
- if persistent lean or drift appears without high action-rate, keep it as a
  diagnostic or weak guard before making it a strong reward term;
- if none of these fail but rollout gain remains poor, re-open the observation
  sufficiency and K-step horizon questions instead of adding presentation
  penalties.

## Test Evidence

### 2026-07-05 - Point 6 Offline Curriculum Distribution

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py
python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py
```

Result:

- current config: `dr_min=1.25`, `dr_max=4.50`, weights
  `easy/frontier/hard=0.45/0.40/0.15`, factors `0.75/1.00/1.08`;
- `frontier=1.25`: `min=1.250`, `p50=1.250`, `at_min=86.6%`;
- `frontier=2.00`: `min=1.500`, so the easy bucket is no longer low;
- `frontier=4.50`: `min=3.375`, `p50=4.500`, `at_max=54.2%`;
- `below_min=0.0%` for all tested frontier scales.

Conclusion:

- Point 6 is confirmed for the current helper/config: Stage 3 has no true
  low-perturbation samples below `dr_min=1.25`.
- Late curriculum has no mild repair bucket; even the easy class becomes
  high-strength when the frontier scale is high.

### 2026-07-05 - Candidate Frontier-Envelope Distribution Test

Command:

```text
python -m py_compile source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py
python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py
```

Candidate rule:

- treat the probed GMT limit `g` as the upper envelope, not the distribution
  mean;
- sample stable buckets: low `20%`, mid `30%`, frontier `40%`, hard/stress
  `10%`;
- ranges: `[0, 0.25g]`, `[0.25g, 0.70g]`, `[0.70g, 1.00g]`,
  `[1.00g, min(1.10g, dr_max)]`;
- if `g == dr_max`, the hard bucket becomes a near-frontier stress band instead
  of collapsing all samples onto `dr_max`.

Result:

- `frontier=1.25`: `min=0.002`, `p50=0.870`, `below_old_min=90.4%`;
- `frontier=2.00`: `min=0.003`, `p50=1.392`, `below_old_min=45.5%`;
- `frontier=4.50`: `min=0.007`, `p50=3.132`, `p90=4.292`,
  `below_old_min=22.5%`;
- bucket fractions stay stable at approximately
  `low=20.7%`, `mid=29.7%`, `frontier=40.1%`, `hard=9.5%`.

Conclusion:

- The candidate rule matches the intended shape: as the probed limit rises, the
  right boundary expands gradually while low and medium perturbations remain in
  the training distribution.

### 2026-07-05 - Formal Training Code Updated

Changed files:

- `source/rsl_rl/rsl_rl/frontres/frontres_dr_curriculum.py`;
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py`.

Formal rule:

- `sample_per_env_dr_strength(...)` now treats the probed frontier as an
  envelope and samples per-env strengths from low/mid/frontier/hard buckets;
- active G1 config explicitly sets weights
  `low/mid/frontier/hard=0.20/0.30/0.40/0.10`;
- active G1 config explicitly sets ranges
  `low_hi=0.25`, `mid_hi=0.70`, `hard_hi=1.10`;
- diagnostics keep old compatibility keys: `easy=low+mid`, plus explicit
  `low`, `mid`, `frontier`, `hard`, and `mean`.

Verification:

```text
python -m py_compile source/rsl_rl/rsl_rl/frontres/frontres_dr_curriculum.py source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py
python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_curriculum_distribution_contract.py
```

Result after implementation:

- `frontier=1.25`: `p50=0.870`, `below_old_min=90.4%`;
- `frontier=2.00`: `p50=1.392`, `below_old_min=45.5%`;
- `frontier=4.50`: `p50=3.132`, `p90=4.292`,
  `below_old_min=22.5%`, `at_max=0.2%`;
- bucket fractions stay near
  `low=20.7%`, `mid=29.7%`, `frontier=40.1%`, `hard=9.5%`.

Conclusion:

- The formal Stage 3 training sampler no longer collapses toward high
  perturbations as the GMT frontier rises.

### 2026-07-05 - Step 5A: Stage 3 Proposal-Only Output Contract

Problem:

- Stage 3 Segment PPO stores and updates 6D `Delta SE(3)` actions, but the
  legacy task-space FrontRES policy could still expose 12D
  `[Delta SE(3), rho(6)]` outputs.
- That made the live path depend on first-6 slicing in storage/evaluation
  helpers.

Formal rule:

- Stage 3 `segment_replay_hrl` uses proposal-only task-space output:
  `num_task_corrections=6`, `task_conf_dim=0`.
- Legacy HSL/acceptance branches may still use `task_conf_dim=1/2/6`.
- If no coefficient head exists, env-side task-space correction uses implicit
  unit coefficients.

Changed files:

- `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`;
- `source/rsl_rl/rsl_rl/frontres/task_space_correction.py`;
- `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`;
- `scripts/rsl_rl/train.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_task_space_proposal_only_contract.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_entrypoint_pseudo_contract.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_task_space_proposal_only_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_entrypoint_pseudo_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_ppo_contract.py
python -m py_compile source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py source/rsl_rl/rsl_rl/frontres/task_space_correction.py source/rsl_rl/rsl_rl/runners/on_policy_runner.py scripts/rsl_rl/train.py source/rsl_rl/rsl_rl/tests/frontres_task_space_proposal_only_contract.py source/rsl_rl/rsl_rl/tests/frontres_segment_stage3_entrypoint_pseudo_contract.py source/rsl_rl/rsl_rl/tests/frontres_stage_entrypoint_contract.py
```

Observed facts:

- proposal-only policy prints
  `mean_shape=(2, 6)`, `std_shape=(2, 6)`, `action_shape=(2, 6)`,
  `log_prob_shape=(2,)`, `correction_shape=(2, 6)`;
- Stage 3 entrypoint prints `task_conf_dim=0` and
  `split_acceptance_head=False`;
- legacy Segment probe contracts still pass with 12D input sliced to 6D, so
  old HSL/acceptance compatibility is preserved.

Conclusion:

- Point 5 is fixed for the Stage 3 active training path: FEMR no longer needs
  to output 12 dimensions when Segment PPO trains only 6D `Delta SE(3)`.

### 2026-07-05 - Step 5B: Live Probe 6D Surface Sentinel

Problem:

- After Step 5A, active Stage 3 uses native 6D proposal-only policy output, but
  the live-probe trace only printed a sentinel on the legacy 12D fallback path.
- A formal run could therefore still show the old
  `storage_uses_first_6_delta_se_dims` diagnostic when using an old branch, but
  had no equally clear proof that the active branch was already native 6D.

Formal rule:

- If rollout policy actions are exactly 6D, the live probe must print
  `semantic=storage_uses_native_6d_delta_se_policy`.
- If a legacy 12D action reaches the Segment Replay storage adapter, the old
  first-6 fallback remains explicit and searchable as a compatibility path.

Changed files:

- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
```

Expected probe fact:

```text
[probe step5b] native_6d_live_probe_trace: trace_count=1 native_6d=True legacy_slice=False
```

Conclusion:

- Step 5B completes the diagnostic side of point 5: a live Stage 3 log can now
  distinguish native 6D policy output from legacy 12D first-6 slicing.

### 2026-07-05 - Step 5C: Checkpoint/Resume 6D Surface Contract

Problem:

- Step 5A changed active Stage 3 to a native 6D proposal-only actor.
- The remaining safety question is checkpoint boundary behavior: HSL init may
  copy a Stage 1 proposal head, but formal Stage 3 full-resume must not accept
  a legacy 12D actor state.

Formal rule:

- `is_full_resume=True` requires an exact 6D Stage 3 repair actor state.
- HSL init still maps the Stage 1 two-head checkpoint into the 6D proposal actor
  and ignores acceptance-head weights.

Changed files:

- `source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py
```

Expected probe facts:

```text
[probe step5c] full_resume_exact_6d: copied=('0.bias', '0.weight', '2.bias', '2.weight') last_weight_shape=(6, 5)
[probe step5c] full_resume_rejects_legacy_12d: error=full Stage 3 resume requires an exact 6D repair actor state
```

Conclusion:

- Step 5C completes the checkpoint side of point 5: old HSL checkpoints remain
  usable for initialization, but full Stage 3 resume cannot silently load a
  legacy 12D actor.

### 2026-07-05 - Step 3A: Observation Layout Contract

Problem:

- The active FrontRES actor now receives an 870D policy observation, which can
  look like a padded 770D GMT proprioceptive observation.
- The current contract is `100D` FrontRES-only prefix plus `770D`
  GMT-compatible suffix; the historical Step 3A contract was `800 = 30 + 770`.

Formal rule:

- `obs[0:100]` is the FrontRES-only prefix.
- `obs[100:870]` is the GMT-compatible suffix and is the only part sent through
  the frozen GMT observation normalizer.
- The actor-facing normalized observation keeps shape `870D`; the GMT-facing
  suffix keeps shape `770D`.

Changed files:

- `source/rsl_rl/rsl_rl/runners/frontres_runtime.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py
```

Observed fact:

```text
[FrontRES Observation Layout Contract] obs_shape=(2, 870) extra_shape=(2, 100) gmt_shape=(2, 770) normalized_shape=(2, 870) gmt_normalizer_calls=[(2, 770)]
```

Conclusion:

- Step 3A originally made the `800 = 30 + 770` contract explicit; the current
  tested contract is `870 = 100 + 770`.
- This step does not yet prove that Stage 3 extra-prefix mean/std are loaded;
  that remains the Step 3C / point 4 boundary.

### 2026-07-05 - Step 3B: Shared Layout Helper And GMT Direct Connector

Problem:

- Step 3A tested the observation split but only the runner normalizer
  used the helper.
- The GMT direct path still had its own inline slicing logic.

Formal rule:

- `source/rsl_rl/rsl_rl/modules/frontres_observation_layout.py` owns the split.
- Runner normalization and `FrontRESActorCritic._run_gmt_direct(...)` must use
  the same split helper.
- Actor-facing observations keep `870D`; GMT direct receives only the `770D`
  suffix.

Changed files:

- `source/rsl_rl/rsl_rl/modules/frontres_observation_layout.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_runtime.py`;
- `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py`;
- `note/00_repository_architecture_map.md`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py
```

Observed facts:

```text
[FrontRES Observation Layout Contract] obs_shape=(2, 870) extra_shape=(2, 100) gmt_shape=(2, 770) normalized_shape=(2, 870) gmt_normalizer_calls=[(2, 770)]
[FrontRES Observation Layout GMT Direct] cached_shape=(2, 870) gmt_policy_calls=[(2, 770)] action_shape=(2, 770)
```

Conclusion:

- Step 3B removes the duplicated actor-to-GMT slicing logic from the active
  normalizer/GMT connector boundary.

### 2026-07-05 - Step 3C: Extra-Prefix Mean/Std Checkpoint Contract

Problem:

- Stage 3 uses a frozen GMT normalizer for the 770D suffix, but the 100D
  FrontRES-only prefix needs Stage-1 empirical mean/std.
- Saving only the frozen GMT normalizer would drop the 100D prefix stats from a
  Stage 3 checkpoint.

Formal rule:

- When checkpoint payloads are saved and `_frontres_extra_mean/std` exist,
  `obs_norm_state_dict` stores the combined `870D` state:
  `[100D extra prefix | 770D GMT suffix]`.
- On restore, Stage 3 extracts only the 100D prefix into
  `_frontres_extra_mean/std`.
- The frozen GMT `obs_normalizer` is not overwritten by the combined 870D
  checkpoint state.

Changed files:

- `source/rsl_rl/rsl_rl/modules/frontres_observation_layout.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_checkpointing.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_checkpointing.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py`;
- `source/rsl_rl/rsl_rl/tests/frontres_segment_live_sampler_contract.py`;
- `note/00_repository_architecture_map.md`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_checkpoint_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py
```

Observed facts:

```text
[probe step3c] load_extra_prefix_stats: keys=('obs_norm_state_dict', 'privileged_obs_norm_state_dict') extra_mean_shape=(1, 100) extra_std_shape=(1, 100) gmt_loaded=False
[probe step3c] save_combined_obs_norm: mean_shape=(1, 870) std_shape=(1, 870) var_shape=(1, 870)
[probe step9] suite_summary: contract_count=37 failed_count=0 total_marker_count=37
```

Conclusion:

- Step 3C fixes the point 4 boundary for Segment checkpoint payloads: Stage 3
  can persist and restore the 100D extra-prefix stats without corrupting the
  frozen GMT suffix normalizer.

### 2026-07-07 - Step 3D: Active K-Step PPO Return And GMT Normalizer Freeze

Problem:

- `frontres_authority_return_horizon` belonged to the retired authority
  actor-critic branch; the active Stage 3 Segment PPO path still used the
  averaged immediate segment reward as `returns`.
- The runner could put the frozen GMT suffix normalizer into train mode during
  FrontRES training, making the checkpoint mean/std contract depend on a
  mode-side effect.

Formal rule:

- Live Segment PPO stores per-step `reward_steps` and `done_steps` during the
  K-step rollout.
- `FrontRESSegmentRolloutStorage.compute_returns_and_advantages(...)` computes
  discounted K-step returns for the active storage batch and keeps the old
  immediate-return behavior only when no per-step trace is supplied.
- Stage 3 default `frontres_segment_k` is 8.
- When `_frontres_gmt_obs_dim` is set, `OnPolicyRunner.train_mode()` keeps the
  GMT normalizer in eval mode and sets `until = 0`; the 100D prefix stats remain
  separate from the frozen 770D suffix stats.

Changed files:

- `source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py`;
- `source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py`;
- `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`;
- `scripts/rsl_rl/train.py`;
- `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`;
- `source/rsl_rl/rsl_rl/modules/rsl_rl_cfg.py`;
- `source/whole_body_tracking/whole_body_tracking/utils/rsl_rl_cfg.py`.

Verification:

```text
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_storage_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_live_probe_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_observation_layout_contract.py
/Users/chengyuxuan/ArtiIntComVis/FEMR/frontres/bin/python source/rsl_rl/rsl_rl/tests/frontres_segment_all_contract_suite.py
```

Observed facts:

```text
[FrontRES Observation Layout Contract] obs_shape=(2, 870) extra_shape=(2, 100) gmt_shape=(2, 770) normalized_shape=(2, 870) gmt_normalizer_calls=[(2, 770)]
[FrontRES Observation Layout Freeze Contract] expects_train_mode_to_preserve_frozen_gmt_normalizer=True
[probe step9] suite_summary: contract_count=41 failed_count=0 total_marker_count=41
```
