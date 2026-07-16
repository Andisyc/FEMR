# Formal Runtime Audit

Status: `phase-b-gain-v002-actor-warmup-runtime-observed-quality-gate-open`

Current C boundary 2026-07-16: the active reward contract is now
`FRS-GAIN-v002`. The formal audit was updated to expose raw survival steps,
per-row effective K, repaired/noisy survival quality, normalized survival Gain,
and the equality between per-step survival Gain sum and final survival Gain.
The v002 formal route is now runtime-observed through the first actor-warmup
update. Representative actor learning and long-run quality are still open; all
prior v001 live Gain/return claims remain stale for this boundary.

Correction 2026-07-15: the first 20-card Atlas revision pointed many cards at
owner functions while emitting their labels only from five runner summaries.
That projection was invalid. Each current card is now accepted only when its
declared owner file contains the same `AUDIT-*` ID, a real default-off emit or
print boundary, adjacent `B1/B2/B3` reading comments, and `Result:
PENDING_LIVE.`. The contract test enforces this source-to-Atlas identity.

Human selection rationale 2026-07-15: every B1/B2/B3 boundary now records
`whyHere` and `failureOwner` beside its source location and captured object.
B1 isolates upstream source failures, B2 identifies the first point where the
owner's semantic object is complete, and B3 detects overwrite, stale-field, or
bypass failures before the formal consumer. All 60 `whyHere` statements are
boundary-specific; the generator and viewer contracts reject duplicated
template rationale.

Source navigation 2026-07-15: every B-step stores a generated `sourceLine` and
a validated local `/open-source` hyperlink. The visible insertion location is
blue with an external-link marker. Clicking it keeps the Atlas page open while
the local server invokes VS Code with `--goto file:line:column`. Generation
fails when an owner lacks the matching block comment, and the contract verifies
that each linked line contains the expected block ID. Direct `vscode://` links
were retired because some browser/OS combinations interpreted the line suffix
as a new empty filename.
Created: 2026-07-14
Audit type: official Stage 3 Segment Replay formal-route live sentinel

## Round Identity

- Active method contract: `FRS-METHOD-v011-segment-replay`
- Active training contract: `FRS-TRAIN-v003-segment-replay-warmup`
- Active reward contract: `FRS-GAIN-v002-style-physics-repair`
- Code revision inspected: `2ff791e` plus the current dirty worktree; the
  deployed source snapshot must match this worktree before live evidence is accepted.
- Checkpoint identity: must be supplied and printed by the upcoming formal run.
- Official command: locked below under `Tiny Formal-Route Command`.
- Current gate: the 32-environment rerun3 executed, but all 32 quartet rows
  terminated within K=8, including all eight policy rows. PPO correctly had no
  eligible row. Reset/episode/done ownership must be audited before another
  formal run.
- Runtime Audit Atlas: `note/architecture/04_stage3_formal_runtime_audit.html` backed by `runtime/04_stage3_formal_runtime_audit.data.json`, using the same `repository_reading_atlas` card layout as 01.
- Prior offline evidence: retained in
  `note/testing/evidence_ledger_frontres_gain_2026-07-13.md` and the current
  repository testing documents; it is not promoted to live evidence.

## Scope

The audit will prove the official formal Stage 3 route only:

`config -> scripts/rsl_rl/train.py -> OnPolicyRunner -> Segment sampler ->`
`reset/preroll -> full-6D Delta SE(3) rollout -> Segment storage -> paired`
`Gain v002 -> direct Segment PPO -> sampler/diagnostics -> checkpoint boundary`

C-specific evidence path:
`raw survival_steps + effective_horizon_K -> survival_quality_repaired/noisy
-> physics_survival_gain -> per-step survival_gain_steps -> returns/advantages`.

Cost reductions may use a tiny environment count, short iteration count, and
small sampled workload. The audit must not change branch selection, action
representation, perturbation semantics, K-step behavior, Gain ownership, PPO
update semantics, or checkpoint ownership.

Out of scope for this round: legacy MOSAIC-only paths, toy contracts as live
evidence, sentinel-only/storage-only/update-loop-only/offline-eval-only
branches, and long training.

## First Live Attempt

Raw evidence: `formal_runtime_audit_20260715.txt`.

- Runtime-observed before failure: official Stage 3 route, HSL actor/normalizer
  load, global Segment sampling, policy-row K=8 through reset, local-rp batch
  strength, quartet `2/2/2/2`, finite `(8,870)` observation, and finite
  full-6D mean/sigma/action.
- Failure owner: `task_space_correction.py::_frontres_contact_consistent_position_correction`.
  PyTorch rejected a scalar lower bound combined with a per-row Tensor upper
  bound in `Tensor.clamp` before `AUDIT-APPLY-01` could emit.
- Correction: preserve the accepted per-row interval semantics with
  `torch.minimum(torch.maximum(z, z_lower), z_upper)` and explicit dtype/device
  alignment. This changes no action dimension, reward, sampler, or PPO rule.
- Audit clarification: perturbation config now reads the agent/runner owner;
  Stage 1 reference length is printed as `cache_horizon_k`, distinct from the
  curriculum `budget/effective horizon_k`.
- Because code changed after the failed attempt, no source result comment is
  promoted to PASS. All live rows require the same formal-route rerun.

## Second Live Attempt

Raw evidence: `formal_runtime_audit_20260715.txt`. The rerun was pulled to the
original local path, so this file now contains the second attempt rather than
a separately named `rerun1` artifact.

- Newly runtime-observed before failure: finite per-row task-space correction,
  frozen GMT execution (`gmt_training=False`), finite paired body/ZMP/contact
  evidence, and finite Physics Gain and Repair Cost.
- Failure owner: `_capture_paired_gain()` passed final `~done_any` as a Style
  row mask. A row that fell anywhere in K therefore lost its entire finite
  pre-fall trajectory, although Physics Gain and PPO eligibility already own
  terminal semantics.
- Correction: Style body and root-orientation components now use the shared
  per-step `horizon & not_done_before_step` mask. A fall truncates later frames
  without deleting earlier evidence. Storage still excludes the terminal
  policy row from PPO through its independent `valid_mask`.
- Audit-field correction: `AUDIT-PERTURB-01` now emits after the Stage 3 preset
  assigns `max_horizon_k=64`; the second attempt's `missing` value was a
  print-order defect, not the sampler's effective horizon.
- Regression evidence: a K=3 fixture falls on step 2 and poisons step 3 with
  an extreme value; pre-fall MPJPE Gain remains finite while the PPO row remains
  invalid. Focused Gain/live/formal tests and aggregate `44/44` pass.
- Because the Gain owner changed after this run, all live rows remain
  `stale-rerun-required`; no source `Result` comment is promoted to PASS.

## Third Live Attempt

Raw evidence: `formal_runtime_audit_20260715.txt`, 650 lines, local timestamp
2026-07-15 15:46:01. The same local filename was overwritten again, so it now
contains this third attempt.

- Runtime-observed: official Stage 3 `train` route, Stage 2 checkpoint and
  prefix-normalizer load, global Segment sampling, K=8 quartet `2/2/2/2`,
  finite 870D observation, full-6D action/application, frozen GMT execution,
  paired Style/Physics/Repair evidence, finite `gain_total`, and finite
  returns/advantages.
- Gain correction confirmed live: `style_gain` and `gain_total` are finite;
  the previous all-NaN Style failure no longer occurs.
- Update boundary: both policy rows were terminal/ineligible, so Segment PPO
  correctly returned `valid_count=0`, zero loss, and no optimizer step. The
  audit helper then raised its own `assert valid_count > 0` before
  `AUDIT-PPO-01`, final diagnostics, or checkpoint persistence could emit.
- Correction: `print_ppo_audit()` now accepts the production no-update result
  and prints `valid=0 update_observed=0`. This does not change storage masks,
  PPO loss, warmup, sampler, or optimizer behavior. A valid-count-zero contract
  reproduces the old assertion and passes after the correction.
- Deployment mismatch: the startup `train.py` probe printed
  `max_horizon_k=missing`, while the later runner probe printed 64. Local
  `train.py` already prints only after assigning 64. Therefore this run used a
  mixed or stale server source snapshot and cannot establish current-revision
  S4 evidence.
- Current status: all reached rows remain `stale-rerun-required`; PPO update,
  accepted diagnostics, and checkpoint payload remain unconfirmed.

## Fourth Live Attempt - Command Mismatch

Raw evidence: `formal_runtime_audit_20260715.txt`, 767 lines, local timestamp
2026-07-15 16:09:19. The same local filename was overwritten again.

- The zero-valid audit correction is runtime-confirmed: `AUDIT-PPO-01` emitted
  `valid=0 update_observed=0` and did not terminate the run.
- Production then correctly rejected the iteration with
  `FrontRES Segment live update produced update_count=0`. This is the accepted
  fail-fast guard for formal training and must not be disabled or removed.
- The simulator reported `PhysX GPU capacities prepared for 8 envs` and quartet
  `2/2/2/2`. The locked rerun3 requires 32 envs and eight policy rows. Rerun3
  was therefore not actually executed.
- The same two global segments were sampled as in the previous attempt. Both
  policy rows fell, `valid=0`, `gain_total=-0.152922`, positive-gain fraction
  0%, and no optimizer update was possible. This is a two-row failed sample,
  not evidence that the PPO implementation is broken.
- Startup still printed `max_horizon_k=missing`, while the later runner owner
  printed 64. Current local `train.py` prints only after assigning 64, proving
  the server script was stale even though the updated source audit helper was
  present.
- PPO post-update trust evidence and checkpoint payload remain unconfirmed.
  Source comments stay `PENDING_LIVE`; Atlas/checklist remain stale or
  unconfirmed until the exact 32-env command and full worktree are deployed.

## Rerun3 - Full Quartet Termination

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 767 lines, local
timestamp 2026-07-15 16:24:31.

- Command cost identity is confirmed: PhysX prepared 32 envs; quartet layout
  was `8/8/8/8`; sampler and batch each contained eight policy rows.
- All 32 rollout rows terminated within K=8. All eight policy rows were
  invalid, `evidence.fall_count=8`, `updates=0/1`, and the required production
  guard rejected `update_count=0`.
- This is not explained by policy-row sample count. Candidate/Noisy/Clean rows
  also terminated, so the failure owner is earlier than PPO and cannot be
  attributed only to actor correction.
- Code audit found that index reset writes motion/frame and robot state only to
  request rows `env_ids=0..7`. The quartet command owner synchronizes reference
  motion indices, but the index-reset owner does not visibly rewrite the other
  24 robot dynamic states or reset quartet episode lifecycle state.
- `run_frontres_segment_live_probe()` randomizes `episode_length_buf` before
  index reset; the index reset hook does not reset it. The current log does not
  expose per-step timeout versus physical termination, so this remains a
  code-supported root-cause hypothesis rather than a completed fix.

## Step A - Reset Lifecycle Probe Inserted

- Added `AUDIT-RESET-LIFECYCLE-01` at the formal rollout owner. It records
  role-aware `episode_length_buf` before randomization, after randomization, and
  after reset; post-reset quartet root/joint pair errors; and per-step
  done/timeout/physical-termination/alive/survival plus first-done step.
- The probe is default-off and does not change reset, rollout, valid masks,
  Gain, PPO, or the production zero-update guard.
- S2 pseudo evidence separates timeout from physical termination on multi-role
  rows and detects deliberately mismatched Noisy root and Clean joint state.
- The Runtime Atlas now contains 21 owner cards. The source-link/B1-B3/whyHere
  contract and viewer import pass; the aggregate suite passes 44/44.
- At insertion time the status remained `PENDING_LIVE`; the following formal
  tiny run supplied the required owner-localizing evidence.

## Step A Live Result - Quartet Reset Owner Confirmed

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 776 lines, local
timestamp 2026-07-15 17:19.

- The reset probe reached the official 32-env quartet route with role counts
  `8/8/8/8`.
- `episode_length_buf` was randomized and not reset, but all step-0 terminations
  were physical: every role reported `done=8`, `time_out=0`, `terminated=8`.
  Therefore stale episode length is not the immediate step-0 failure owner.
- Robot joint state was not paired after reset. Candidate/Noisy/Clean joint max
  error versus policy was `8.43462/8.34135/6.73203`.
- The first root metric used world coordinates and is contaminated by different
  per-env origins, so its `64.0339/47.9014/64.1966` values are invalid evidence.
  The probe now subtracts `scene.env_origins`; origin-relative root evidence
  requires the next rerun.
- The index-reset trace confirms only env ids 0-7 were written. The remaining
  24 baseline rows received synchronized reference identity later, but not the
  matching robot dynamic state.
- Earliest contradicted invariant: quartet dynamic-state reset ownership. PPO,
  Gain, valid masks, action bounds, and the zero-update guard remain downstream
  consequences and must not be changed for this failure.

## Step B - Quartet Dynamic-State Reset Repair

- The formal live probe now configures the split-env layout before index reset
  and attaches explicit Policy/Candidate/Noisy/Clean env IDs to the request.
- The index-reset adapter expands each sampled motion/frame to all active roles,
  writes motion group, root pose/velocity, joint pose/velocity, clears
  `motion_end_buf`, and resets `episode_length_buf` for every role row.
- Perturbation strength/family remains attached only to Policy source rows.
  Existing command synchronization subsequently shares Noisy corruption with
  Candidate/Noisy while retaining the Clean baseline behavior.
- Sample-level reset result remains one row per sampled segment, so PPO/storage
  evidence ownership is unchanged.
- The eight-row golden fixture passes with frames and local roots
  `[3,4,3,4,3,4,3,4]`, matched joint state, zero episode age, and nonzero DR
  scale only on Policy rows. Live S4 survival remains pending.

## Step B Live Result - Reset Alignment Closed

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 777 lines, local
timestamp 2026-07-15 17:49.

- The deployed code emitted explicit role IDs `0..7`, `8..15`, `16..23`, and
  `24..31`; the updated adapter therefore ran on all 32 rows.
- After reset every role had episode age zero. Candidate/Noisy/Clean
  origin-relative root max error was at most `1.90735e-6`; joint error was
  exactly zero. The quartet reset repair is live-confirmed.
- Nevertheless all four roles still reported eight physical terminations at
  step 0 and zero timeouts. This moves the first unresolved owner from reset
  state to the environment termination terms.
- Added a default-off per-term snapshot using IsaacLab
  `TerminationManager.get_term()`. The next tiny formal rerun will report role
  masks for `motion_end`, `time_out`, `anchor_pos`, `anchor_ori`, and
  `ee_body_pos`; no threshold or termination function was changed.
- Required next boundary: add role-aware per-step done/timeout/survival audit,
  then repair full-quartet reset only if the live probe confirms this owner.
- Startup still prints `max_horizon_k=missing` before the runner prints 64, so
  the server `scripts/rsl_rl/train.py` remains stale relative to local source.

## Step C Live Result - Termination Owner Identified

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 777 lines, local
timestamp 2026-07-15 18:05.

- At rollout step 0, `anchor_pos` is true for all Policy/Candidate/Noisy/Clean
  rows: `{policy:8,candidate:8,noisy:8,clean:8}`.
- `motion_end`, `time_out`, `anchor_ori`, and `ee_body_pos` are zero for every
  role. Their union exactly explains the returned physical termination mask.
- The active G1 path defines this term through `bad_anchor_pos_z_only()`, which
  compares reference anchor z with robot anchor-body z against the configured
  `0.5 m` threshold.
- This closes term identity but not value provenance. The next live-dependent
  boundary is the signed z error and the two values that produce it. Threshold
  changes or termination suppression are not justified by this log.
- `AUDIT-ANCHOR-Z-01` is inserted at `bad_anchor_pos_z_only()`. It captures
  per-role world-frame reference/robot z, signed/absolute error, original/raw/
  correction reference decomposition, command time step, motion index, and the
  unchanged termination mask. Runtime status is `runtime-observed` by `E35`.

## Step D Live Result - Stale Command Cache Identified

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 779 lines, local
timestamp 2026-07-15 22:02.

- First termination call: clean reference z and robot torso z are aligned near
  `0.79 m`, but raw/final reference z remains near `0..0.03 m` for every role.
- The signed error is therefore about `-0.776 m`; absolute error is
  `0.543..0.869 m`, so all 32 rows legitimately exceed threshold `0.5 m`.
- Motion indices and sampled time steps are already quartet-aligned. The bad
  object is specifically the command's cached perturbed position, not the
  sampled frame identity, robot reset state, FrontRES z correction, or threshold.
- On the second call, raw/clean z agree and absolute error drops below
  `0.014 m`. This occurs only after the first termination caused automatic env
  reset; the rollout's first-done state is already irreversible.
- Classified defect: index-reset integration defect. It assigns motion/frame
  and robot state but reaches first termination before initializing the command
  cache for that sampled frame.

## Step E Offline Result - Current-Frame Cache Fix Integrated

- `MultiMotionCommand.refresh_frontres_reference_cache_current_frame()` now
  owns position/quaternion perturbation cache, supervised target construction,
  vertical projection, and quartet synchronization for the current frame.
- Ordinary `_update_command()` advances the frame and then calls this owner
  exactly once. Segment index reset calls the same owner after assigning its
  explicit sampled frame, without advancing `time_steps`.
- The eight-row golden fixture preserves frames `[3,4,3,4,3,4,3,4]`, performs
  one cache refresh, and obtains current-frame cache z `[1,1,1,1,1,1,1,1]`.
- A production-source contract proves the refresh owner has no frame increment,
  performs one position draw, one quaternion draw, and one quartet sync.
- Evidence class: integrated-offline. The original E35 runtime result is stale
  for current code; one tiny formal rerun must show first-call raw/clean/robot z
  aligned and `anchor_pos=0` before this defect is closed.

## Step E Live Result - Cache And First-Step Termination Closed

Raw evidence: `formal_runtime_audit_20260715_rerun3.txt`, 803 lines, local
timestamp 2026-07-15 22:32.

- First owner call has raw/clean/final reference z around `0.78..0.80 m`, robot
  z in the same range, and maximum absolute error `0.020011 m`.
- `anchor_pos`, every other active termination term, and returned done are zero
  for Policy/Candidate/Noisy/Clean at every one of eight rollout steps.
- Survival reaches eight for every row; first-done indices remain `-1`.
- Storage receives 32 rows and eight valid Policy rows. One formal critic-only
  PPO update executes with actor weight zero, trust accepted, KL
  `6.008e-05`, and no rollback/retry.
- `model_1.pt` saves model, optimizer, observation normalizer, sampler, Gain
  config, and warmup state. The cache/reset defect is integrated-live closed.
- Tiny-run Gain is `-0.006978` because repair cost `0.073285` exceeds the small
  positive Style/Physics gains. This is a one-batch method-quality observation,
  not a cache regression; actor weight is intentionally zero in critic-only.
- E39 confirms the four E38 fields and actor-warmup. E41 confirms the E40
  perturbation fields on the official route. Compact formal diagnostics are
  now runtime-complete; method-quality acceptance remains separate.

## Current Checklist

| ID | Owner/function | Core parameter | Probe location | Expected shape/value | Status | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| GOV-01 | workflow governance | contract/code/checkpoint/command identity | this document | identity is explicit before run | user-reviewed | this file |
| METHOD-07 | critic-ready actor curriculum | HSL -> protected RL warmup -> actor ramp -> joint RL | `AUDIT-HSL-LOAD-01`, `AUDIT-WARMUP-01`, `AUDIT-PPO-01` | phase/weight and frozen actor boundary | runtime-observed | E39 critic_only -> actor_warmup(0.5), valid=7, trust accepted |
| ROUTE-01 | formal entrypoint and runner | Stage 3 branch selection | `AUDIT-ROUTE-01` | live_train=1 and alternate_modes=0 | runtime-observed | E37 official train route |
| OBS-01 | observation/normalizer | 870D balance/ZMP observation | `AUDIT-OBS-01` | expected layout, finite, same normalizer identity | runtime-observed | E37 finite 870D, 100D+770D split |
| K-01 | sampler/reset/rollout | per-row K and valid rows | `AUDIT-KPLAN-01`, `AUDIT-KROLLOUT-01` | K tensor, role rows, reset/valid counts | runtime-observed | E39 reset_success_frac=1.0, K=8..64, valid=7 |
| ACT-01 | actor/rollout/storage | full-6D Delta SE(3) | `AUDIT-ACTION-01`, `AUDIT-APPLY-01` | observation and action/storage tuple shapes, finite | runtime-observed | E39 full-6D finite, delta_norm=0.016680 |
| GAIN-01 | paired evidence/Gain | Style/Physics/Repair/Total | `AUDIT-PAIR-EVIDENCE-01`, `AUDIT-GAIN-01`, `AUDIT-RETURN-01` | all canonical Gain components populated | runtime-observed | E39 reward/return/advantage and canonical Gain populated |
| PPO-01 | Segment PPO update | old stats -> loss -> optimizer step | `AUDIT-PPO-01` | phase, loss, grad, delta, pre/post KL, trust, frozen GMT | runtime-observed | E39 actor-warmup valid=7, post KL=0.005442, trust accepted |
| PERSIST-01 | checkpoint boundary | model/normalizer/optimizer/sampler/Gain/warmup | `AUDIT-PERSIST-01` | payload identity at actual save | runtime-observed | E37 model_1.pt payload complete |
| DIAG-01 | diagnostics | live populated metrics | all `AUDIT-*` snapshots | compact searchable fields; missing remains explicit | runtime-observed | E41 contains no audit missing/NaN/exception and closes perturb fields live |

## Phase B Probe Ownership

Human control surface: open `note/architecture/04_stage3_formal_runtime_audit.html`. Every row below has a matching Atlas reading block and a `PENDING_LIVE` source comment at the real owner location.

| Audit ID | Parent design points | Formal owner | Runtime question |
| --- | --- | --- | --- |
| `AUDIT-ROUTE-01` | all active points | formal training loop | Is this the official Stage 3 train route? |
| `AUDIT-PERTURB-01/02` | M-02 | perturb config/application | Are configured and applied perturbations aligned? |
| `AUDIT-SEGDATA-01/SAMPLER-01` | SR-01 | dataset/sampler | Are segment identity and priority evidence from one transaction? |
| `AUDIT-KPLAN-01/KROLLOUT-01` | M-06 | sampler/rollout | Does per-row K survive planning and execution? |
| `AUDIT-RESET-LIFECYCLE-01` | M-06, Q-PAIR, SR-01 | reset/rollout | Do all quartet roles start from aligned episode and robot state, and are done rows timeout or physical termination? |
| `AUDIT-OBS-01/ACTION-01/APPLY-01` | M-04 | runtime/policy/task correction | Does 870D obs produce and execute full-6D repair? |
| `AUDIT-GMT-01` | M-10 | frozen GMT | Is GMT frozen while executing repaired reference? |
| `AUDIT-PAIR-01/PAIR-EVIDENCE-01` | Q-PAIR | pair layout/capture | Are roles and paired execution evidence aligned? |
| `AUDIT-GAIN-01/RETURN-01` | Q-01 | Gain/storage | Does canonical Gain reach PPO returns? |
| `AUDIT-HSL-LOAD-01` | M-03 | checkpoint load | Did Stage 2 actor/normalizer initialize Stage 3? |
| `AUDIT-WARMUP-01/PPO-01/DIAG-01` | M-05 | warmup/PPO/diagnostics | Did the permitted update become the reported accepted state? |
| `AUDIT-PERSIST-01` | M-03, SR-01, M-05, Q-01 | checkpoint save | Does model_N.pt retain all formal identities? |

### Expanded 20-owner probe inventory

| Audit IDs | Parent design | Formal owner boundary |
| --- | --- | --- |
| `AUDIT-ROUTE-01` | M-03, M-05, SR-01 | formal Stage 3 route |
| `AUDIT-PERTURB-01`, `AUDIT-PERTURB-02` | M-02 | perturbation config and applied rows |
| `AUDIT-SEGDATA-01`, `AUDIT-SAMPLER-01` | SR-01 | segment dataset and replay transaction |
| `AUDIT-KPLAN-01`, `AUDIT-KROLLOUT-01` | M-06 | K plan and executed horizon |
| `AUDIT-RESET-LIFECYCLE-01` | M-06, Q-PAIR, SR-01 | episode buffer, quartet root/joint pairing, per-step done/timeout/termination/survival |
| `AUDIT-OBS-01`, `AUDIT-ACTION-01`, `AUDIT-APPLY-01` | M-04 | 870D obs, full-6D actor, task correction |
| `AUDIT-GMT-01` | M-10 | frozen GMT execution |
| `AUDIT-PAIR-01`, `AUDIT-PAIR-EVIDENCE-01` | Q-PAIR | quartet roles and paired evidence |
| `AUDIT-GAIN-01`, `AUDIT-RETURN-01` | Q-01 | canonical Gain and PPO returns |
| `AUDIT-HSL-LOAD-01` | M-03 | Stage 2 actor/normalizer load |
| `AUDIT-WARMUP-01`, `AUDIT-PPO-01`, `AUDIT-DIAG-01` | M-05 | warmup, accepted PPO state, diagnostics |
| `AUDIT-PERSIST-01` | M-03, SR-01, M-05, Q-01 | formal checkpoint payload |

All 21 Atlas cards remain `PENDING_LIVE`. Offline contracts prove insertion and synchronization only.

## Method Semantics Extracted (Dr.Cheng)

- Indispensable variable: executable improvement of a corrupted reference frame
  under frozen GMT, measured by paired Noisy/Repaired evidence over a dynamic
  segment.
- Policy action: full six-dimensional Delta SE(3),
  `[dx, dy, dz, droll, dpitch, dyaw]`; `local_rp` labels the corruption and
  does not mask the repair output.
- Training signal: `gain_total = style_gain + physics_gain - repair_cost`
  with the active named weights; generic environment reward is not a fallback.
- Ownership split: the sampler owns segment selection/replay priority from
  rollout evidence; Segment PPO owns policy-action returns/advantages and the
  direct full-6D update.
- Horizon: the per-row K assignment must survive sampling, reset, rollout,
  storage/returns, sampler evidence, and diagnostics; the preset exposes
  `8/16/32/64` support with `max_horizon_k=64`.

## Method Design Map

The primary review unit is the research design, not an entrypoint, tensor, or
diagnostic. Runtime routes, storage, checkpoints, and logs are implementation
boundaries attached below the method design that they serve.

| ID | Research design point | Current meaning | Current status |
| --- | --- | --- | --- |
| M1 | Front End Residual architecture | A deployable residual actor reads the FrontRES/GMT observation, writes a bounded full-6D Delta SE(3) correction before frozen GMT, and does not replace the tracker | active; contract/code-confirmed; live pending |
| M2 | Perturbation Family | Perturbation family defines the corruption distribution; the current experiment selects `local_rp`, while the actor still outputs full 6D | active; contract/code-confirmed; live distribution pending |
| M3 | HSL warmup | Stage 2 supervises the same full-6D actor with task-space repair targets and an easy-to-hard perturbation/DR schedule, then saves `model_warmup.pt` | active; code-confirmed; current checkpoint identity pending |
| M4 | HRL 4-split rollout | Each selected case is evaluated through Projected/Repaired policy, Candidate/Search, Noisy GMT, and Clean GMT rows with explicit role ownership | active; contract/code-confirmed; live role counts pending |
| M5 | Segment Replay mechanism | Global sampling discovers coverage; replay revisits useful segments; review rechecks stale/solved segments; priority uses rollout-time evidence | active; contract/code-confirmed; live source/priority distribution pending |
| M6 | K-step curriculum | Per-row horizons such as `8/16/32/64` expose immediate repair and delayed regret and must survive every downstream boundary | active; offline-connected; live effective-K distribution pending |
| M7 | Critic/Actor warmup curriculum | Any transition from HSL initialization into RL must first make the critic usable, then protect and gradually release the actor so early RL gradients cannot destroy the pretrained policy | active under FRS-TRAIN-v003; offline-connected; live phase/gradient evidence pending |
| M8 | ZMP reward | Two distinct active roles must stay separate: deployable ZMP/balance observation context, and paired ZMP/support-margin evidence inside Physics Gain. The review-only reward adapter is not the formal PPO reward | active after role separation; live ZMP population pending |
| M9 | Gain design | One canonical owner combines paired Style gain, Physics gain, and full-6D Repair cost; PPO, sampler, diagnostics, and eval consume the same decomposition | active; offline-connected; live component population pending |

### M1 - Front End Residual architecture

- Observation and layout: `mdp/observations.py` ->
  `modules/frontres_observation_layout.py:split_frontres_policy_obs` line 8 ->
  `runners/frontres_runtime.py:apply_obs_normalizer` line 54.
- Actor: `modules/front_residual_actor_critic.py:update_distribution` line 796,
  `act` line 922, and `get_actions_log_prob` line 977.
- Runtime write: `runners/frontres_rollout_step.py:prepare_frontres_rollout_step`
  line 228 -> `frontres_action_cone.py:project_task_target` line 20 ->
  `task_space_correction.py:apply_frontres_task_corrections` line 60 -> frozen
  GMT command.
- Human reading question: what FrontRES sees, what exactly it outputs, where
  safety is applied, and where GMT consumes the repaired reference.

### M2 - Perturbation Family

- CLI/config selection: `scripts/rsl_rl/train.py:_configure_frontres_motion_perturbations`
  line 547 and the specialist preset handling around lines 607-683/1215-1221.
- Curriculum definition: `frontres/frontres_dr_curriculum.py` and
  `frontres/training_schedule.py:frontres_warmup_perturbation_mode_groups` line
  143.
- Runtime application: `frontres/perturbation_runtime.py` and
  `mdp/motion_perturbations.py`.
- Segment persistence/forwarding: `frontres_segment_dataset.py` -> live sampler
  -> reset request -> rollout diagnostics.
- Human reading question: which corruption is sampled, at what strength, on
  which rows, and whether family ever leaks into policy action masking.

### M3 - HSL warmup

- Stage preset: `scripts/rsl_rl/train.py:_apply_frontres_stage_preset` lines
  743-750 selects `supervised_restore`.
- Warmup decision/call: `on_policy_runner.py` lines 890-911.
- Warmup owner: `runners/frontres_warmup.py:resolve_frontres_warmup_iterations`
  line 21 and `run_frontres_joint_warmup` line 70.
- Full-6D target: `mdp/observations.py:get_supervision_target_task_space` line
  374 and `runners/frontres_hsl_rollout_target.py`.
- Perturbation/DR curriculum inside HSL: `frontres_warmup.py` lines 121-177.
- Checkpoint output: `frontres_warmup.py` lines 620-624 saves
  `model_warmup.pt`.
- Human reading question: target source, actor input/output, valid dimensions,
  easy-to-hard perturbation schedule, and the exact artifact handed to Stage 3.

### M4 - HRL 4-split rollout

- Layout owner: `runners/frontres_training_setup.py:configure_frontres_pair_layout`
  line 142.
- Environment branch identity:
  `mdp/commands.py:set_frontres_quartet_baseline` line 1245.
- Branch semantics: Projected/Repaired policy, Candidate/Search, Noisy GMT, and
  Clean GMT; current counts are `n_train/n_candidate/n_base/n_clean`.
- Rollout construction: `frontres_rollout_step.py:prepare_frontres_rollout_step`
  line 228 and `frontres_segment_live_probe.py:_run_live_rollout_capture` line
  2144.
- Policy-credit boundary: `_trial_metadata_ppo_update_mask` line 1418 and
  `build_live_segment_storage` line 1427.
- Human reading question: what each split means, which action each split runs,
  which comparisons form evidence, and which rows can update the actor.

### M5 - Segment Replay mechanism

- Semantic owner: `frontres_segment_sampler.py:FrontRESSegmentSampler` line
  140; `sample` line 207 and `update_with_probe` line 289.
- Global/replay/review budgeting: `plan_rollout_budget` line 394.
- Formal connector: `frontres_segment_live_sampler.py:run_frontres_segment_sampler_step`
  line 189 and `_build_current_segment_batch` line 373.
- Dynamic state boundary: `frontres_segment_dataset.py`,
  `frontres_segment_reset.py`, and `_apply_current_segment_reset` line 960.
- Evidence/priority connector: `build_live_sampler_evidence` line 446 ->
  `FrontRESSegmentSampler.update_with_probe` line 289.
- Human reading question: why a segment is sampled again, what state is reset,
  and which rollout fact changes its priority.

### M6 - K-step curriculum

- Stage 3 config: `train.py` lines 836-837 sets base K 8 and maximum K 64.
- Assignment: `frontres_segment_sampler.py:plan_rollout_budget` line 394 and
  `expand_rollout_trials` line 452.
- Forwarding chain: trial attachment line 350 -> trial metadata line 1239 ->
  reset line 960 -> rollout line 2144 -> valid-step masks/returns in
  `frontres_segment_storage.py:compute_returns_and_advantages` line 175 ->
  sampler evidence and diagnostics.
- Human reading question: where K is chosen, how row termination works, and
  whether delayed regret reaches both PPO and replay priority.

### M7 - Critic/Actor warmup curriculum

- User-confirmed method principle: entering RL requires a staged transition so
  a noisy/untrained value surface and early high-variance policy gradients do
  not destroy the HSL-initialized actor.
- Intended causal order:
  `HSL actor initialization -> protected RL warmup -> actor ramp -> joint RL`.
  During the protected warmup, the critic learns while actor PPO weight remains
  zero. The historical formal values aligned critic warmup and actor warmup at
  200 iterations, followed by a 500-iteration actor ramp; the debug values were
  50/50/100.
- Historical design evidence:
  `FRS-METHOD-v000-design-history-compendium.md` lines 408-434 records the
  general `Critic-Ready Actor Update` principle: train the evaluator first and
  let the actor follow only after its local ordering is usable.
- Historical schedule evidence: the same compendium lines 2015-2029 records
  critic warmup, actor warmup/ramp, absolute-iteration resume semantics, and
  the expected transition from actor weight 0 to 1.
- Historical runtime evidence: `train_stage3_segment_hrl.txt` line 392 records
  `PPO actor warmup=200 ramp=500` in a Segment Replay HRL run.
- Recoverable implementation evidence from `HEAD` before the current cleanup:
  `frontres_ppo_actor_weight_for_iter` returned actor weight 0 during warmup,
  linearly ramped it over `ppo_actor_ramp_iterations`, then returned 1.
- Important separation: the historical authority/rho actor used this staging,
  but does not own the principle. The active direct full-6D actor needs its own
  critic-ready transition without restoring rho, acceptance, authority heads,
  or active-dimension masks.
- Current implementation: `train.py` configures 200 critic-only iterations and
  a 500-iteration actor ramp. `frontres_segment_warmup.py` owns phase mapping,
  and `run_frontres_segment_single_update` applies the actor loss weight while
  clearing non-critic gradients in critic-only. `AUDIT-PPO-01` now exposes the
  live phase, loss weight, gradient/update facts, KL, trust decision, and GMT
  optimizer isolation without restoring any retired authority/rho mechanism.

### M8 - ZMP reward

- Shared geometry/formula: `mdp/balance.py:frontres_balance_context_from_feet`
  line 26.
- Deployable observation role: `mdp/observations.py:frontres_balance_context_proxy`
  line 133.
- Paired rollout evidence role:
  `frontres/frontres_balance.py:_frontres_branch_balance_margin` line 15 and
  `frontres_segment_live_probe.py:_capture_physics_frame` line 2542.
- Physics Gain consumer: `frontres_gain.py:compute_paired_physics_gain` line
  146.
- Non-formal compatibility path:
  `mdp/rewards.py:frontres_no_regret_balance_reward_candidate` line 407 is a
  review-only adapter, not the PPO reward owner.
- Human reading question: which balance facts are actor inputs, which are
  privileged rollout evidence, and which scalar actually reaches Gain.

### M9 - Gain design

- Pure owner: `frontres_gain.py:compute_paired_style_gain` line 94,
  `compute_paired_physics_gain` line 146, `compute_repair_cost` line 184, and
  `compute_segment_gain` line 354.
- Live paired capture: `frontres_segment_live_probe.py:_capture_paired_gain`
  line 1562.
- PPO return: `_segment_storage_rewards` line 1666 and
  `_segment_storage_reward_steps` line 1708 -> Segment storage/PPO.
- Replay priority: `frontres_segment_live_sampler.py:build_live_sampler_evidence`
  line 446 -> sampler update.
- Diagnostics/evaluation: `frontres_segment_live_training.py` and
  `frontres_segment_diagnostics.py`.
- Human reading question: raw component source, pairing, sign/scale/K mask,
  consumer identity, and missing-data behavior.

## Implementation Audit Objects (secondary, not method design)

The D01-D12 items below are retained only as lower-level runtime verification
objects. They are not research contributions or top-level method design points.
Line numbers describe the current source before review comments are inserted;
function names are the stable navigation anchors.

| ID | Active design point | Non-negotiable invariant | Fastest live falsifier |
| --- | --- | --- | --- |
| D01 | Formal Stage 3 route | `MODE=train` reaches `learn_frontres_segment_live`; no alternate probe/eval branch substitutes for training | route snapshot reports stage, objective, train mode, and `runner_learn=True` |
| D02 | 870D deployable observation and normalizer identity | actor receives 100D FrontRES prefix + 770D GMT suffix; Stage 2/3 stats are loaded and applied in the same layout | prefix/suffix/normalized shapes and finite ranges disagree or checkpoint stats are absent |
| D03 | Direct full-6D Delta SE(3) action | corruption family never narrows `[dx,dy,dz,droll,dpitch,dyaw]`; safety projection is not an active-dim mask | actor/executed/stored action is not `(N,6)` or an RP family zeros other dimensions by policy mask |
| D04 | Global/replay/review Segment sampling and explicit trial roles | segment identity and source survive row expansion; only policy rows receive PPO credit | source/role counts disappear, row domains disagree, or search rows become PPO-valid |
| D05 | Dynamic reset or faithful preroll | reset restores dynamic state and metadata in the same row domain; pose-only teleport is insufficient | reset-success count differs from trial rows or velocity/state fidelity fails |
| D06 | Per-row K curriculum | assigned `8/16/32/64` horizon survives trial plan, reset, rollout masks, returns, evidence, and diagnostics | sampled K differs from effective/storage K or every row is silently clamped to 8 |
| D07 | Single paired Style/Physics/Repair Gain owner | Clean/Noisy/Repaired use matching segment/K; generic env reward and legacy RP score cannot replace Gain | required components are absent/stale, pairing count differs, or PPO return follows env reward |
| D08 | Policy-credit and storage tuple identity | action, old log-prob, old mean/sigma, return, advantage, and valid mask describe the same policy rows and representation | tuple shapes/source differ or invalid/search rows contribute gradient |
| D09 | Sign-preserving direct Segment PPO | default advantage mode is `scale_only`; clipped surrogate, old/new KL, optimizer step, rollback, and LR order remain explicit | positive gain flips sign, loss/grad is nonfinite, update count is zero, or post-KL escapes trust control |
| D10 | Sampler evidence isolation | sampler priority consumes rollout-time canonical Gain, not post-update KL, parameter delta, or logger state | poisoning PPO diagnostics changes sampler priority/update evidence |
| D11 | Checkpoint ownership and Stage 2 -> Stage 3 identity | one owner saves/loads policy, normalizer, sigma/optimizer, sampler, and Gain identity as applicable | loaded path/state keys differ, normalizer layout changes, or final checkpoint is outside the formal log route |
| D12 | Non-stale diagnostics and evaluation isolation | required live facts are populated or `UNCONFIRMED`, never silent zero; eval samples fresh data without mutating training state | repeated metadata is identical without resampling, required raw components vanish, or eval changes sampler/RNG state |

## Design-To-Code Matrix

### D01 - Formal Stage 3 route

- Config owner: `scripts/rsl_rl/train.py:_apply_frontres_stage_preset` at line
  706; the active Stage 3 block begins at line 751 and sets objective, K,
  advantage mode, sampler fractions, and live-train routing.
- Dispatch owner: `scripts/rsl_rl/train.py:main` at lines 1379-1385 selects
  `runner.learn_frontres_segment_live` only when formal live training is active.
- Thin runner API: `on_policy_runner.py:learn_frontres_segment_live` at line 596.
- Formal loop: `frontres_segment_live_training.py:run_frontres_segment_live_training_loop`
  at line 1818.
- Read first: Stage 3 preset -> `main` dispatch -> formal loop guard.

### D02 - Observation and normalizer

- Layout owner: `modules/frontres_observation_layout.py:split_frontres_policy_obs`
  at line 8; checkpoint prefix extraction/composition at lines 17 and 62.
- Runtime normalization: `runners/frontres_runtime.py:apply_obs_normalizer` at
  line 54.
- Live observation boundary:
  `frontres_segment_live_probe.py:_read_live_observations` at line 2060.
- Actor consumer: `modules/front_residual_actor_critic.py:update_distribution`
  at line 796.
- Persistence owner: `frontres_checkpointing.py:save_runner` line 226 and
  `load_runner` line 333.
- Read first: layout split -> runtime normalizer -> actor distribution ->
  checkpoint save/load.

### D03 - Full-6D action and safety boundary

- Config declaration: `rsl_rl_mosaic_cfg.py` line 832 sets
  `num_task_corrections=6`.
- Distribution/action/log-prob owner:
  `front_residual_actor_critic.py:update_distribution` line 796, `act` line 922,
  and `get_actions_log_prob` line 977.
- Rollout construction: `frontres_rollout_step.py:prepare_frontres_rollout_step`
  at line 228.
- Physical safety projection: `frontres_action_cone.py:project_task_target` at
  line 20.
- Reference write: `task_space_correction.py:apply_frontres_task_corrections`
  at line 60.
- Read first: actor raw distribution -> bounded action/log-prob -> rollout plan
  -> safety projection -> command write.

### D04 - Segment sampling and trial roles

- Semantic sampler: `frontres_segment_sampler.py:FrontRESSegmentSampler` line
  140; `sample` line 207; `sample_rollout_rows` line 242.
- Budget/role expansion: `plan_rollout_budget` line 394 and
  `expand_rollout_trials` line 452.
- Formal connector: `frontres_segment_live_sampler.py:run_frontres_segment_sampler_step`
  line 189.
- Trial metadata attachment and batch construction:
  `_attach_frontres_segment_trial_plan` line 350 and
  `_build_current_segment_batch` line 373.
- Read first: sampler source choice -> budget/trial expansion -> attached batch
  role metadata.

### D05 - Dynamic reset/preroll

- Reset contract objects: `frontres_segment_reset.py`.
- Formal reset entry:
  `frontres_segment_live_probe.py:_apply_current_segment_reset` at line 960.
- Trial metadata row owner:
  `frontres_segment_live_probe.py:_current_trial_metadata` at line 1239.
- Environment hook owner: `frontres_segment_stage1_env_hooks.py` and the
  tracking command/reset adapter.
- Rollout consumer: `_run_live_rollout_capture` at line 2144.
- Read first: current batch metadata -> reset request/result -> rollout start.

### D06 - Per-row K curriculum

- Config: `train.py` lines 836-837 sets base K 8 and max K 64.
- Assignment owner: `frontres_segment_sampler.py:plan_rollout_budget` line 394
  and `expand_rollout_trials` line 452.
- Forwarding owners: live sampler trial attachment line 350; trial metadata line
  1239; reset line 960; rollout capture line 2144.
- Return owner:
  `frontres_segment_storage.py:compute_returns_and_advantages` at line 175.
- Diagnostic consumers: live update loop and train summary.
- Read first: assigned K -> attached K -> reset K -> valid-step mask -> return K
  -> printed effective K.

### D07 - Canonical paired Gain

- Pure owner: `frontres_gain.py:compute_paired_style_gain` line 94,
  `compute_paired_physics_gain` line 146, `compute_repair_cost` line 184,
  `compute_segment_gain_step` line 265, and `compute_segment_gain` line 354.
- Live capture owner:
  `frontres_segment_live_probe.py:_capture_paired_gain` line 1562.
- PPO reward/step extraction: `_segment_storage_rewards` line 1666 and
  `_segment_storage_reward_steps` line 1708.
- Sampler connector:
  `frontres_segment_live_sampler.py:build_live_sampler_evidence` line 446.
- Diagnostic/eval consumers: `frontres_segment_live_training.py` and
  `frontres_segment_diagnostics.py`.
- Read first: pure component formulas -> paired live capture -> PPO returns and
  sampler evidence -> diagnostics.

### D08 - Policy credit and storage tuple

- PPO eligibility owner:
  `frontres_segment_live_probe.py:_trial_metadata_ppo_update_mask` line 1418.
- Formal storage build: `build_live_segment_storage` line 1427.
- Storage owner: `frontres_segment_storage.py:FrontRESSegmentRolloutStorage`
  line 67; `add_transition` line 113; `compute_returns_and_advantages` line 175;
  `FrontRESSegmentStorageBatch.to_ppo_batch` line 42.
- PPO consumer: `run_frontres_segment_single_update` line 1928.
- Read first: role/reset masks -> transition tuple -> return/advantage -> PPO
  batch.

### D09 - Segment PPO and trust region

- Loss owner: `frontres_segment_ppo.py:compute_frontres_segment_ppo_loss` line
  176.
- Advantage owner: `_prepare_advantages` line 401.
- Exact distribution KL owner: `_distribution_kl_mean` line 456.
- Optimizer/update owner:
  `frontres_segment_live_probe.py:run_frontres_segment_single_update` line 1928.
- Post-update evidence and rollback helper:
  `_post_update_segment_ppo_diagnostics` line 337.
- Read first: PPO batch validation -> advantage scaling -> policy evaluation ->
  surrogate/KL -> backward/step -> post-update diagnostics/rollback/LR.

### D10 - Sampler evidence isolation

- Evidence construction:
  `frontres_segment_live_sampler.py:build_live_sampler_evidence` line 446.
- Priority/state update:
  `frontres_segment_sampler.py:update_with_probe` line 289.
- Canonical Gain selection: `_active_gain` line 637.
- Formal call order: `run_frontres_segment_sampler_step` line 189.
- Read first: immutable rollout summary -> evidence object -> priority/state
  update; PPO diagnostics must not enter these inputs.

### D11 - Checkpoint identity

- Evidence record: `frontres_checkpointing.py:record_frontres_checkpoint_probe`
  line 132.
- Save owner: `save_runner` line 226.
- Load/migration owner: `load_runner` line 333.
- Formal training save boundary:
  `frontres_segment_live_training.py:_save_live_checkpoint` line 1785 and final
  save in `run_frontres_segment_live_training_loop` lines 1866-1869.
- Read first: load path and migration -> live state -> save payload -> checkpoint
  probe.

### D12 - Diagnostics and evaluation

- Live train formatter:
  `frontres_segment_live_training.py:_print_live_train_summary` line 1575.
- Periodic evaluation: `run_frontres_segment_periodic_eval` line 145.
- Sequence evaluation: `run_frontres_segment_sequence_offline_eval` line 269.
- Shared formatters: `frontres_segment_diagnostics.py:format_segment_train_effect_log`
  line 119, `format_segment_motion_quality_log` line 167, and
  `format_segment_periodic_eval_log` line 286.
- Read first: captured raw facts -> summary keys -> formatter -> eval state
  isolation and metadata freshness.

## Source Comment Plan (runtime-probing-debug)

No source comment has been inserted yet. After user confirmation, comments will
be added in three bounded passes so each pass can be reviewed independently.
Every status/docstring and major block comment will carry its owning method ID
such as `M1`, `M4`, or `M9`, so a reader can move in both directions:
method design -> code owner and code owner -> method design.

### Comment Pass A - Formal routing, sampling, reset, and K

- `scripts/rsl_rl/train.py:_apply_frontres_stage_preset`: short caller-facing
  docstring plus `B1` mode exclusion, `B2` formal Stage 3 semantic config, and
  `B3` route summary comments for D01/D06.
- `frontres_segment_live_training.py:run_frontres_segment_live_training_loop`:
  `Status/Upstream/Downstream/Evidence/Gap` docstring and `B1` route guard,
  `B2` formal update, `B3` validation/diagnostics/checkpoint comments.
- `frontres_segment_sampler.py:sample`, `plan_rollout_budget`,
  `expand_rollout_trials`: block comments for source choice, K assignment, and
  semantic trial roles.
- `frontres_segment_live_sampler.py:run_frontres_segment_sampler_step`,
  `_attach_frontres_segment_trial_plan`, `_build_current_segment_batch`:
  comments that distinguish segment rows, expanded env rows, policy rows, and
  evidence-only rows.
- `frontres_segment_live_probe.py:_apply_current_segment_reset` and
  `_current_trial_metadata`: comments for row-domain and dynamic-state fidelity.

### Comment Pass B - Observation, action, Gain, storage, and PPO

- `frontres_observation_layout.py` and `frontres_runtime.py:apply_obs_normalizer`:
  comments for 100D prefix, 770D suffix, privilege status, and checkpoint stats.
- `front_residual_actor_critic.py:update_distribution`, `act`, and
  `get_actions_log_prob`: block comments for raw Gaussian -> bounded full-6D
  action -> inverse-transform log-prob identity.
- `frontres_rollout_step.py:prepare_frontres_rollout_step`,
  `frontres_action_cone.py:project_task_target`, and
  `task_space_correction.py:apply_frontres_task_corrections`: comments separating
  policy semantics, physical safety, and command write.
- `frontres_gain.py`: module audit status and `B1/B2/B3` comments for
  Style/Physics/Repair, paired total, missing-component semantics, and K masks.
- `frontres_segment_live_probe.py:build_live_segment_storage` and
  `frontres_segment_storage.py`: comments for policy-credit mask, same-source
  tuple, raw advantages, and PPO conversion.
- `frontres_segment_ppo.py:compute_frontres_segment_ppo_loss` and
  `run_frontres_segment_single_update`: comments for scale-only advantages,
  clip/KL, update order, post-update trust evidence, rollback, and LR.

### Comment Pass C - Sampler isolation, checkpoint, and diagnostics

- `frontres_segment_live_sampler.py:build_live_sampler_evidence` and
  `frontres_segment_sampler.py:update_with_probe`: comments stating rollout-time
  evidence ownership and forbidden PPO-diagnostic inputs.
- `frontres_checkpointing.py:save_runner/load_runner` and
  `frontres_segment_live_training.py:_save_live_checkpoint`: status comments for
  state ownership, Stage 2 -> Stage 3 identity, and remaining live gap.
- `frontres_segment_live_training.py:_print_live_train_summary` and
  `frontres_segment_diagnostics.py`: comments for raw source, aggregation,
  `UNCONFIRMED`, and non-stale requirements.

Comment language will be Chinese with ASCII punctuation. Comments will explain
contracts, blocks, coordinate/shape/role/gradient boundaries, and current
evidence only; they will not restate individual assignments or claim runtime
facts before the live run.

## Secondary Execution Order (repo-architecture-atlas)

| Boundary | Owner and entry | Upstream | Downstream | Evidence class |
| --- | --- | --- | --- | --- |
| Config / entry | `scripts/rsl_rl/train.py:_apply_frontres_stage_preset`, `main` | CLI, task config, checkpoint path | `OnPolicyRunner` | code-confirmed; live pending |
| Runner dispatch | `source/rsl_rl/rsl_rl/runners/on_policy_runner.py:learn_frontres_segment_live` | Stage 3 config | `run_frontres_segment_live_training_loop` | code-confirmed; live pending |
| Training loop | `frontres_segment_live_training.py:run_frontres_segment_live_training_loop` | runner/boundary | repeated `run_frontres_segment_live_update_loop`, checkpoint save | code-confirmed; live pending |
| Sampling | `frontres_segment_live_sampler.py:run_frontres_segment_sampler_step` | global/replay/review sampler state | trial/quartet batch and reset metadata | code-confirmed; live pending |
| Reset / rollout | `frontres_segment_live_probe.py:run_frontres_segment_live_probe` | current segment batch | observations, executed action, K-step capture | code-confirmed; live pending |
| Storage | `frontres_segment_live_probe.py:build_live_segment_storage`, `frontres_segment_storage.py` | captured policy rows and paired evidence | valid PPO batch, returns/advantages | code-confirmed; live pending |
| Gain | `frontres_gain.py`, called from live probe | Clean/Noisy/Repaired paired rollout | canonical `gain_total` and component evidence | contract-confirmed; live pending |
| PPO | `frontres_segment_live_probe.py:run_frontres_segment_single_update`, `frontres_segment_ppo.py:compute_frontres_segment_ppo_loss` | same-source action/stats/returns | optimizer update and trust-region diagnostics | contract-confirmed; live pending |
| Diagnostics / checkpoint | `frontres_segment_live_training.py:_print_live_train_summary`, `_save_live_checkpoint` | update summary | log and saved artifact | code-confirmed; live pending |

## Minimal S/T Selection (all-module-test)

| ID | Required tier | T kinds | Why selected | Intentionally skipped |
| --- | --- | --- | --- | --- |
| ROUTE-01 | S2, S4 | `T-connect`, `T-live` | prove `train` reaches the formal Segment loop | alternate branch tests are not formal evidence |
| OBS-01 | S1, S3, S4 | `T-shape`, `T-order`, `T-finite`, `T-persist`, `T-live` | 870D layout and Stage 2 normalizer/checkpoint identity are high-risk | export/play are out of this training sentinel |
| K-01 | S2, S4 | `T-connect`, `T-order`, `T-mask`, `T-live` | implementation already supports K; live must show multiple effective K | long-horizon quality and 64-step statistics are not claimed from one run |
| ACT-01 | S1, S2, S4 | `T-shape`, `T-order`, `T-cone`, `T-scale`, `T-connect`, `T-live` | full-6D actor -> executed correction -> stored action is a core identity | no action-mask ablation; it is forbidden by the active contract |
| GAIN-01 | S1, S2, S4 | `T-value`, `T-unit`, `T-K`, `T-step-sum`, `T-sign`, `T-pair`, `T-connect`, `T-single-owner`, `T-no-legacy-score`, `T-live` | v002 must show raw steps separately, normalize by each K, and preserve paired Gain semantics in IsaacLab | no generic reward comparison; it is outside the active method |
| PPO-01 | S1, S2, S4 | `T-clip`, `T-KL-exact`, `T-detach`, `T-permute`, `T-advantage-sign`, `T-update-order`, `T-state`, `T-connect`, `T-live` | update semantics and direct full-6D policy update are high-risk | no new algorithm change is authorized by this audit |
| PERSIST-01 | S3, S4 | `T-persist`, `T-order`, `T-diff`, `T-live` | one-iteration formal loop saves a checkpoint; identity must be logged | full resume migration is a separate S3/S4 run |
| DIAG-01 | S1, S2, S4 | `T-unconfirmed`, `T-nonstale`, `T-decompose`, `T-unit`, `T-live` | raw survival, quality, normalized Gain, and missing-state behavior must be separately visible | long-run quality and periodic/sequence eval are separate gates |

Existing S0-S3 contract evidence is reused as implementation evidence, not
promoted to live evidence: the aggregate suite, full-6D/no-mask, actual policy
distribution, Gain connectivity, Stage 3 entrypoint, and checkpoint contracts
already cover the local boundaries.

## Inserted AUDIT Probe Set

The authoritative probe inventory is the 20-owner table above and the Runtime
Audit Atlas. Every owner contains its matching default-off `AUDIT-*` emission,
adjacent B1/B2/B3 reading comments, and `Result: PENDING_LIVE`. The prior
seven-probe planning projection is retired and must not be used for live
acceptance. `AUDIT-GAIN-01` now captures raw survival steps, effective K,
repaired/noisy quality, normalized survival Gain, and step-sum error.
`AUDIT-RETURN-01` captures the same K and per-step survival Gain trace beside
storage reward, returns, and advantages.

## Tiny Formal-Route Command

This is the locked command for the C v002 live audit. It uses the official
wrapper and `MODE=train`; `--frontres_formal_runtime_audit` only enables
default-off observations and does not select a sentinel, probe-only,
storage-only, update-loop-only, or offline-eval branch.

```bash
CUDA_VISIBLE_DEVICES=2 \
CACHE_DIR=/hdd1/cyx/AMASS_G1Segment \
LOG_PATH=/hdd1/cyx/FEMR/formal_runtime_audit_gain_v002_20260716.txt \
PERIODIC_EVAL_ENABLED=0 \
RUN_NAME=FEMR_FORMAL_RUNTIME_AUDIT_GAIN_V002_20260716 \
HYDRA_FULL_ERROR=1 \
bash /hdd1/cyx/FEMR/run_stage3.sh \
  /hdd1/cyx/FEMR/model/model_warmup.pt \
  /hdd1/cyx/AMASS_G1NPZ_Final \
  32 \
  1 \
  1 \
  train \
  --frontres_formal_runtime_audit
```

Expected cost: one official Stage 3 training iteration, one Segment PPO update
step per iteration, 32 environments (eight policy rows), one final checkpoint,
and no periodic
evaluation. The wrapper backgrounds the process; the audit log is the named
runtime evidence file. The checkpoint path, cache path, motion path, and
deployed source snapshot must be confirmed on the target machine before this
command is accepted as live evidence.

## Stop Conditions

- Stop before live execution if the locked command enters an alternate
  sentinel, probe, storage, update-loop, offline-eval, or sequence-eval branch.
- Stop if checkpoint identity, method contract, or active branch is unknown.
- Stop if the active `FRS-TRAIN-v003` warmup route or its printed phase/weight
  disagrees with the deployed code and checkpoint identity.
- Stop if an owner does not receive the expected shape/count/value relation.
- Stop if `AUDIT-GAIN-01` reports missing raw survival, K, quality, or step-sum
  fields, or if `survival_gain_sum_abs_error` is not within the offline
  contract tolerance.
- Stop if `gain_source` is not `FRS-GAIN-v002`.
- Do not convert offline contract PASS into formal-route or live PASS.

## Required Next Step

Phase A, probe insertion, Atlas synchronization, and the first actor-warmup
formal run are complete. The next bounded step is a matched offline evaluation
of `model_200.pt` and `model_201.pt` from the same actor probe run. `model_200.pt`
is the pre-actor-update boundary; `model_201.pt` is the first actor-weighted
update boundary. Use identical motions, start frames, perturbation family,
effective K, and seed. Do not begin longer training until this pair is read.
