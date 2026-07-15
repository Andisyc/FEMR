---
contract_id: FRS-METHOD-v010
status: superseded
effective_date: 2026-07-05
updated_date: 2026-07-13
supersedes: FRS-METHOD-v009
superseded_by: FRS-METHOD-v011
scope: FrontRES Stage 3 Segment Replay HRL method semantics
---

# Segment Replay HRL Method Contract

## Method Boundary

The active Stage 3 method treats each motion segment as a replayable local RL
task while retaining global coverage over the motion dataset:

```text
global motion dataset
  -> slice replayable dynamic segments
  -> reset to cached/prerolled dynamic state
  -> apply a sampled perturbation
  -> FrontRES direct full-6D Delta SE(3) repair
  -> K-step frozen-GMT rollout
  -> executable gain over Noisy/GMT
  -> PPO policy update + segment-priority update
```

FrontRES is a reference repair policy. It does not replace GMT and does not own
robot joint-space control or recovery-reference generation.

The method-version delta is Segment Replay, not the six-dimensional output.
Full-6D task-space repair was already an established FrontRES invariant. This
version changes how repair cases are organized, revisited, evaluated across
time, and supplied to HRL.

## Segment Replay Design

Global-only sampling gives broad motion coverage but too few repeated trials on
each difficult local state. Local-only repetition improves one segment but
overfits and loses dataset coverage. Segment Replay closes both needs:

```text
global source -> discover diverse segments
replay source -> revisit high-learning-value segments
review source -> recheck solved or stale segments when configured
```

One stable `segment_id` owns motion identity, start frame/phase, dynamic reset
state, perturbation family/strength, rollout horizon, and replay evidence.

Segment priority must be computed from rollout-time evidence and remain
independent of post-update PPO diagnostics. A segment is worth replaying because
its execution evidence indicates learning value, not because a logger, stale
policy statistic, or post-update KL changed.

## Dynamic Reset Boundary

Segment reset must preserve the local dynamics needed by frozen GMT:

- root pose and velocity;
- joint pose and velocity;
- phase/contact-relevant state;
- controller/reference history when required;
- direct cached state or a clean preroll that reconstructs it.

A static pose-only reset is not equivalent to a replayable dynamic segment.

## K-Step Curriculum

The horizon is part of the method, not only an efficiency parameter. Short
horizons provide cheap local repair evidence. Longer horizons reveal delayed
regret, accumulated discontinuity, and failures that appear after an initially
positive gain.

```text
unknown/easy evidence -> short K
promising/frontier    -> repeated medium K
delayed regret        -> long K
```

The curriculum is accepted only when multiple intended K values reach the
formal reset, rollout, return, sampler-evidence, and diagnostic paths. A planner
that can return `8/16/32/64` while formal training executes only `K=8` is
implemented-only, not integrated.

## Action Semantics

```text
Delta SE direction -> repair direction candidate
Delta SE magnitude -> implicit execution authority
Delta SE = 0       -> no-op / do-not-repair decision
rollout gain       -> improvement over the Noisy/GMT no-op baseline
```

The executable action remains full 6D across all segments:

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

Perturbation family describes the corruption source. It must not narrow the
policy output dimensions. A `local_rp` perturbation still permits the policy to
use all six coupled correction dimensions. Legacy `active_task_dims` or
specialist masks must not silently redefine the active method as RP-only.

Safety constraints such as the upward-`dz` boundary may restrict physically
unsafe execution, but they must be named as safety constraints rather than
misrepresented as perturbation-family action masks.

## Observation Boundary

The deployable actor input is the current FrontRES/GMT observation. Clean
reference, exact artifact source, and future rollout outcome are training or
evaluation evidence, not deployable actor inputs.

The actor therefore does not promise exact Clean reconstruction under partial
observability. It learns an execution-safe residual that improves frozen-GMT
execution relative to Noisy/GMT.

## Training Evidence And Update Ownership

The method uses two paired improvements and one repair regularizer:

```text
gain_total = w_style * style_gain
           + w_physics * physics_gain
           - w_repair * repair_cost
```

Stage 2 HSL and Stage 3 PPO use the same proposal-only full-6D actor interface.
HSL may initialize the repair policy, but it is not the final proposal that
Stage 3 merely accepts or scales. PPO directly updates the full-6D repair policy
from valid policy-action rows. Search/counterfactual rows may update segment
evidence but must not receive policy-gradient credit for actions the policy did
not execute.

Style is evaluated against immutable Clean motion. Physics is evaluated from
paired frozen-GMT execution. Generic environment, teleoperation,
velocity-command, task, or unrelated tracking reward must not enter PPO return
or sampler priority. The exact decomposition belongs to
`../reward/FRS-GAIN-v001-style-physics-repair.md`.

Segment priority and PPO have separate owners:

```text
rollout evidence -> segment state/priority/replay budget
policy action + return/advantage -> PPO update
```

## Required Behavior Boundaries

- No-regret: prefer corrections that improve over Noisy/GMT.
- Replay balance: preserve global motion coverage while revisiting useful local
  segments.
- Horizon coverage: expose both immediate gain and delayed regret.
- Dynamic reset fidelity: do not replace segment state with pose-only reset.
- Full-dimensional repair: preserve all six correction freedoms regardless of
  perturbation family.
- Bounded authority: avoid correction magnitude unrelated to repair need.
- Temporal smoothness: avoid residual oscillation and jerk.
- Clean no-op protection: clean or near-clean references should prefer a zero
  or small residual.
- Frozen tracker: FrontRES must not update or replace GMT.

## Forbidden Active-Path Assumptions

- Perturbation family implies active action dimensions.
- `local_rp` means the executable repair is RP-only.
- A historical HSL/acceptance, rho, alpha, or authority-critic contract defines
  the active Stage 3 method.
- A confidence, acceptance, or rho column is appended to the active 6D policy.
- Generic environment reward is used as Segment repair gain.
- Segment Replay is reduced to a cache or an ordinary random sampler.
- A K-step planner is called integrated while formal training clamps every row
  to one fixed horizon.
- Search or counterfactual actions receive PPO credit as policy actions.
- Absolute survival alone is sufficient evidence of repair quality.
- A locally implemented training mechanism is assumed to be integrated into
  formal training without connectivity and runtime evidence.

## Required Evidence

The active path must expose enough evidence to verify:

- policy and executed action are full 6D;
- per-dimension policy mean, sigma, and executed correction;
- perturbation family and strength distribution;
- Noisy/Repaired gain and harmful-repair rate;
- effective rollout horizon distribution;
- policy/search trial-role counts and source-conditioned segment distribution;
- dynamic reset/preroll success and state fidelity;
- sampler priority flow from rollout evidence;
- training mechanisms are both implemented and integrated into the formal path.

Architecture must be updated whenever this active contract changes method
ownership, interfaces, runtime routing, or diagnostics.

## Owned Subcontracts

- Formal Stage 3 training route: `../training/FRS-TRAIN-v001-segment-replay.md`.
- Paired reward semantics: `../reward/FRS-GAIN-v001-style-physics-repair.md`.
- PPO advantage semantics: `../optimization/FRS-PPO-v001-sign-preserving-advantage-scaling.md`.
- Periodic and sequence evaluation: `../evaluation/FRS-EVAL-v001-segment-evaluation.md`.
