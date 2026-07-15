# FrontRES Current Engineering Plan

Status: Phase B rerun3 still pending; fourth attempt used stale script and 8 env instead of locked 32
Updated: 2026-07-15
Scope: restore `FRS-DP-09` Actor/Critic warmup on the formal Stage 3 Segment PPO route and close the minimal `FRS-DP-05` Frozen GMT evidence gap.

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

Stop condition: any DP-09 owner is only locally implemented, stale test counts
remain, or Architecture still describes the missing route as active.
