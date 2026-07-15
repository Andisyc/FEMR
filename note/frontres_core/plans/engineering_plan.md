# FrontRES Current Engineering Plan

Status: Phase B Step E integrated-live; sampled-frame command-cache defect closed by E37
Updated: 2026-07-15
Scope: restore `FRS-DP-09` Actor/Critic warmup on the formal Stage 3 Segment PPO route and close the minimal `FRS-DP-05` Frozen GMT evidence gap.

Current bounded recovery step: `AUDIT-RESET-LIFECYCLE-01` observes reset-before/randomized/after episode buffers, quartet root/joint state pairing, and per-step role-aware done/timeout/termination/survival. It does not modify reset behavior, PPO eligibility, Gain, or fail-fast guards.

### Reset Recovery Step B / 2: Quartet Dynamic-State Reset

Objective: expand each sampled policy reset row to its corresponding
Policy/Candidate/Noisy/Clean environment rows before rollout.

Scope: establish pair layout before reset; attach explicit role env IDs to the
index-reset request; write one motion/frame-derived root and joint dynamic state
to all four roles; reset their episode-length lifecycle.

Non-scope: perturbation-family semantics, FrontRES full-6D action, Gain, PPO,
valid masks, termination thresholds, or zero-update guards.

Owner files/modules:
- `runners/frontres_segment_live_probe.py`: layout-to-reset connector.
- `frontres/frontres_segment_stage1_env_hooks.py`: simulator reset owner.
- focused lifecycle and role-pairing contracts.

Expected evidence: S1/S2 `T-role/T-state/T-forward/T-timeout`; all role rows
share motion/frame, origin-relative root, joint pose/velocity, and zero episode
length while perturbation scale/mask remains policy-owned.

Stop condition: any baseline role remains stale, perturbation leaks into Clean,
sample-level reset result changes from eight rows to 32 rows, or existing
index-reset and aggregate contracts regress.

### Reset Recovery Step C / 3: Termination Term Localization

Objective: identify the exact IsaacLab termination term that kills all aligned
quartet rows at rollout step 0.

Scope: read current-step term masks from `TerminationManager.get_term()` after
`env.step`; summarize every active term by Policy/Candidate/Noisy/Clean role.

Non-scope: termination thresholds/functions, reset, action, Gain, PPO, valid
masks, and fail-fast behavior.

Owner files/modules:
- `runners/frontres_formal_runtime_audit.py`: compact per-term role summary.
- `runners/frontres_segment_live_probe.py`: post-step observation connector.
- formal runtime audit contract.

Expected evidence: S2 `T-role/T-source/T-value` and S4 one-run term identity.

Stop condition: the probe changes done behavior, active term names cannot be
read, or a live term mask cannot be reconciled with the returned done mask.

Step result: completed by `E33`. At rollout step 0, `anchor_pos` is true for
all 8 Policy, 8 Candidate, 8 Noisy, and 8 Clean rows. Every other active term
is false. The returned done mask therefore reconciles exactly with the active
term masks.

### Reset Recovery Step D / 4: Anchor Position Value Localization

Objective: identify why `bad_anchor_pos_z_only()` sees an error above `0.5 m`
immediately after an otherwise aligned quartet reset.

Scope: observe role-aware reference anchor z, robot torso z, signed/absolute z
error, threshold, command time step, and cached-reference identity immediately
before termination computation at rollout step 0.

Non-scope: threshold changes, termination suppression, command update order,
reset writes, PPO, Gain, valid masks, and fail-fast behavior.

Owner files/modules:
- `tracking/mdp/terminations.py`: `bad_anchor_pos_z_only()` value owner.
- `tracking/mdp/commands.py`: cached reference anchor and robot anchor owners.
- `runners/frontres_segment_live_probe.py`: role-aware formal-route connector.

Expected evidence: S2 `T-source/T-value/T-frame/T-role` plus one S4 snapshot.

Stop condition: the first mismatched object is identified among cached
reference frame/time, reference anchor z, robot torso z, or threshold routing.
Insertion status: user-reviewed and inserted as `AUDIT-ANCHOR-Z-01`. The probe
is default-off and does not change the returned termination mask. S4 value
provenance is live-confirmed by `E35`.

Step result: completed by `E35`. On the first termination call, clean reference
and robot anchor z both average about `0.79 m`, while raw/final reference z is
still near the environment-origin height (`0..0.03 m`). The resulting absolute
error averages about `0.776 m` and alone exceeds the `0.5 m` threshold for all
roles. On the second call, raw and clean reference z agree and error falls below
`0.014 m`, but the first done mask is already sticky for the rollout.

### Reset Recovery Step E / 5: Sampled-Frame Command Cache Initialization

Objective: initialize the command's cached perturbed reference from the sampled
motion/frame before the first termination evaluation, without advancing the
frame or changing perturbation semantics.

Scope: add one command-owned current-frame cache refresh boundary; call it from
the index-reset adapter after motion/frame and perturbation role state are set,
then write the robot from that same frame/cache identity.

Non-scope: calling the full `_update_command()` from reset, changing time-step
order, suppressing done, raising the threshold, PPO, Gain, or valid masks.

Expected evidence: S1/S2 `T-frame/T-role/T-state/T-forward` proving
raw-reference z equals current-frame perturbed reference for all roles before
the first step; S4 proving step-0 `anchor_pos=0` without bypassing termination.

Stop condition: cache refresh advances `time_steps`, draws perturbation twice,
changes Clean semantics, or first-step raw/clean/robot identity remains broken.

Implementation result: `MultiMotionCommand` now owns
`refresh_frontres_reference_cache_current_frame()`. Both ordinary command
updates and index reset call the same cache construction; only ordinary updates
advance `time_steps`. The index-reset adapter invokes it after motion/frame and
role perturbation setup and before robot write/first termination. Offline
offline evidence is `E36`; live evidence is `E37`.

Live result: completed by `E37`. First-call raw/clean/robot anchor z align,
maximum absolute error is `0.020011 m`, all quartet roles survive all eight
steps, `valid=8`, one critic-only PPO update is accepted, and `model_1.pt` is
saved. No termination threshold or done handling was changed.

## Objective

Align the formal Stage 3 code with `FRS-TRAIN-v003`:

```text
HSL actor checkpoint
-> critic_only: value update, actor update weight = 0
-> actor_warmup: Segment PPO actor weight rises from 0 to 1
-> joint: ordinary direct full-6D Segment PPO
```

This restores optimization protection for the same direct full-6D repair
actor. It does not restore confidence, rho, authority, acceptance, active-dim
masks, generic runner PPO, or supervised loss inside Stage 3.

## Source Comparison

The pre-modification `HEAD` contains the reusable schedule idea in
`frontres/training_schedule.py::frontres_ppo_actor_weight_for_iter` and applies
it in the generic `OnPolicyRunner.learn()` loop. The current formal Stage 3
route instead dispatches to `learn_frontres_segment_live()` and
`run_frontres_segment_single_update()`. Therefore the schedule semantics must
be ported to the Segment PPO owner rather than copying the old generic loop.

## Step Map

### Step 1 / 4: Segment Warmup Phase Owner

Objective: implement a pure, deterministic Stage 3 phase schedule and weighted
Segment PPO objective.

Scope: phase config, iteration-to-phase mapping, critic-only actor weight 0,
actor warmup monotonic ramp, joint weight 1.

Non-scope: runner wiring, checkpoint IO, live environment, perturbation/K/Gain.

Owner files/modules:
- `frontres/frontres_segment_warmup.py`: phase calculation.
- `algorithms/frontres_segment_ppo.py`: actor-weighted PPO objective.
- focused S1 contract test.

Expected evidence: S1 `T-value`, `T-grad`, and boundary tests for all phases.

Stop condition: critic-only produces actor/std gradients, value loss is
disabled, actor weight is non-monotonic, or full-6D action semantics change.

### Step 2 / 4: Formal Stage 3 Integration

Objective: propagate the phase through the official Stage 3 train branch.

Scope: Stage 3 preset, live training iteration, single-update config,
production diagnostics.

Non-scope: changing sampler, rollout roles, K, Gain, trust-region, or eval.

Owner files/modules:
- `scripts/rsl_rl/train.py`: production defaults.
- `runners/frontres_segment_live_training.py`: phase selection per iteration.
- `runners/frontres_segment_live_probe.py`: pass actor weight to Segment PPO.
- `runners/frontres_segment_live_update_loop.py`: phase diagnostics.
- focused S2 connectivity test.

Expected evidence: official Stage 3 preset reaches every phase; actor parameters
stay unchanged in critic-only while critic parameters change; actor delta
appears during actor warmup/joint.

Stop condition: formal train bypasses the schedule, alternate probe-only
branches are used as proof, or diagnostics report a phase different from the
loss weight.

### Step 3 / 4: Persistence And Frozen GMT

Objective: prove resume phase identity and frozen-GMT optimizer isolation.

Scope: checkpointed iteration/config identity, recomputed phase after resume,
GMT `requires_grad=False`, GMT exclusion from optimizer, zero GMT parameter
delta after one Segment update.

Non-scope: saving frozen GMT weights or changing GMT execution behavior.

Owner files/modules:
- `runners/frontres_checkpointing.py` and Stage 3 checkpoint contracts.
- `algorithms/frontres_unified.py` optimizer construction.
- S2/S3 `T-connect`, `T-grad`, `T-persist`, `T-state` tests.

Expected evidence: resume at the same iteration selects the same phase and
actor weight; one Segment update cannot change GMT parameters.

Stop condition: phase state depends on an unsaved mutable counter, optimizer
contains GMT parameters, or GMT parameter delta is nonzero.

### Step 4 / 4: Cross-File Acceptance

Objective: close offline alignment and prepare, but do not run, Phase B live
audit.

Scope: impacted tests, aggregate suite, Architecture current-state refresh,
test inventory/control board/evidence/checklist consistency.

Non-scope: live IsaacLab execution or long training.

Expected evidence: S0-S3 tests pass with fresh counts; all documentation uses
`FRS-TRAIN-v003`; remaining S4 facts are explicit. The current aggregate is
`44/44` with `failed_count=0` after adding the task-space correction contract.

The second Phase B attempt reached canonical Gain and exposed a separate
integration mismatch: final `done_any` erased the full Style row instead of
truncating its trajectory at the fall. Gain capture now reuses the per-step
horizon/alive mask for body and root-orientation Style, while storage keeps
terminal PPO eligibility unchanged. The focused regression and aggregate suite
pass; the current source still requires rerun3 before any S4 promotion.

The third Phase B attempt confirmed finite Gain and returns but sampled zero
PPO-eligible policy rows. Production PPO correctly selected a no-update result;
the formal audit helper incorrectly asserted that every batch must update. The
helper now reports `update_observed=0` without changing control flow. Rerun3
uses 32 environments to provide eight policy rows and improve the chance of
observing a real optimizer step. The entire worktree, including
`scripts/rsl_rl/train.py`, must be synchronized before that run.

The fourth attempt did not satisfy this plan: the simulator reported 8 envs,
quartet policy count 2, and stale startup audit ordering. Both policy rows fell,
so the required live-training guard rejected `update_count=0`. No production
guard or PPO rule should change; the next action remains the exact synchronized
32-env rerun3.

The later 32-env rerun3 satisfied the cost configuration but still produced
zero valid rows: all 32 quartet rows terminated within K=8. The next bounded
step is no longer another PPO rerun. It is a reset-lifecycle audit covering
per-step timeout/termination, survival step, quartet role, episode-length state,
and whether index reset writes all four paired dynamic states. PPO, Gain, and
the zero-update guard are non-scope until this owner closes.

Stop condition: any DP-09 owner is only locally implemented, stale test counts
remain, or Architecture still describes the missing route as active.

### Step F: Formal Diagnostic Tuple Closure

Objective: close the four compact `missing` fields observed in E37 without
changing Stage 3 training semantics.

Scope: correct live-summary aliases for reset success, applied correction norm,
and quartet roles; retain canonical reward in the read-only storage batch;
refresh contracts and Runtime Atlas.

Non-scope: reset/cache behavior, Gain calculation, PPO batch/loss, optimizer,
or warmup schedule.

Status: integrated-offline in E38. The next official formal run must verify the
four values and enter actor warmup before this step receives S4 closure.

### Step G: Actor-Warmup Formal Sentinel

Objective: observe one critic-only iteration followed by one actor-warmup
iteration on the official Stage 3 route while rechecking the E38 fields.

Scope: 32 environments, two learning iterations, one update per iteration,
formal probes enabled, periodic evaluation disabled. Audit-only warmup
boundaries are critic=1 and actor=2, so iteration 0 has actor weight 0 and
iteration 1 has actor weight 0.5.

Non-scope: long training, production warmup defaults (200/500), reward tuning,
or checkpoint promotion.

Expected evidence: `AUDIT-WARMUP-01` transitions from `critic_only` to
`actor_warmup`; the actor-warmup update has finite loss/gradient/parameter
delta and valid trust diagnostics; K rollout, apply, pair, and return rows no
longer contain `missing`.

Status: S4 runtime-observed in E39. The two-iteration sentinel reached
critic-only and actor-warmup, and all four E38 fields were populated.

### Step H: Perturbation Audit Alias Closure

Objective: remove the final duplicate `missing` values without changing the
perturbation or K-step training paths.

Scope: add max-horizon and advantage-normalization fields to the formal task
config owner; copy consumed reset-request family counts and strength
distribution into the live summary; update the formal probe contract.

Non-scope: perturbation sampling/application, K curriculum, Gain, PPO, or
training defaults.

Status: integrated-offline in E40. One short S4 rerun must show
`max_horizon_k=64`, `family_counts={'local_rp': 8}`, and finite strength
min/mean/max without `missing`.
