# FrontRES Design Contract

This document records the current experiment-level design contract for FrontRES.
It is more specific than the Dr.Cheng skill.  Read this before implementing
nontrivial changes to FrontRES training, rollout labels, PPO/HRL behavior, or
diagnostics.

## Version Goal

FrontRES is a front-end residual refiner before a frozen GMT tracker.  The
current version studies root-level reference artifacts, especially local
roll/pitch perturbations that become damaging at high strength.  The goal is
not only to keep the robot executable, but to make the repaired reference
approach the clean rollout closely enough for demo-quality behavior.

## Component Ownership

- **FEMR / FrontRES proposal** owns the task-space repair proposal
  \(\Delta g^{\mathrm{HSL}}_t \in SE(3)\), represented as
  \((\Delta x,\Delta y,\Delta z,\Delta r,\Delta p,\Delta y)\).
- **HSL** owns the main geometric restoration direction.  It uses Clean,
  Noisy, and Repaired rollout information to construct continuous sample
  weights, harmful-repair penalties, and rollout-aware supervised labels.
- **HRL / PPO** does not own the repair direction.  In the current hybrid
  design, PPO owns only the six-dimensional dynamics-aware acceptance vector
  \(\rho_t\in[0,1]^6\).
- **Dynamics-aware acceptance** decides how much of each HSL proposal dimension
  can be safely written into GMT:
  \[
  \Delta g^{\mathrm{write}}_t =
  \rho_t \odot \Delta g^{\mathrm{HSL}}_t.
  \]
  Here \(\rho_t\) is not an uncertainty confidence score.  It is the
  rollout-trained acceptance field for the clean-oriented repair direction.
- **Action Cone** owns output feasibility constraints, including active
  dimensions, per-axis bounds, upward-\(z\) constraints, and jump/contact
  restrictions.
- **Diagnostics** must describe the value actually written into GMT.  They must
  not report old conceptual variables as if they were the deployed correction.

## Current Output Interface

The active `hsl_hybrid` branch uses a twelve-dimensional task-space output:

\[
a^{\mathrm{FEMR}}_t =
(\Delta x,\Delta y,\Delta z,\Delta r,\Delta p,\Delta yaw,
\rho_x,\rho_y,\rho_z,\rho_r,\rho_p,\rho_{yaw}).
\]

The first six dimensions are the HSL repair proposal.  The last six dimensions
are the PPO-owned per-axis acceptance coefficients.  HSL answers "where should
the corrupted root frame move?"  HRL/PPO answers "how much of this clean-oriented
repair can the current dynamics accept?"

The old `conf_pos` and `conf_rpy` interface is kept only for legacy objectives
and ablations.  In the active branch the last six outputs should be described
as dynamics-aware acceptance, not as confidence.

## Forbidden Freedoms

- Do not let PPO update the \(\Delta SE(3)\) proposal direction in
  `hsl_hybrid`.  PPO must be restricted to the six acceptance outputs
  \(\rho_t\).
- Do not reinterpret \(\rho_t\) as ordinary confidence or as a generator of new
  repair directions.  It can only accept or suppress the HSL proposal.
- Do not pre-shrink the HSL label and then shrink it again through
  \(\rho_t\).
- Do not let Safe/Broken/Harmful samples dominate the proposal direction loss.
- Do not let temporal continuity cache survive an episode reset.
- Do not remove old HRL/HSL branches unless explicitly requested.  They are
  research assets for later papers.

## Sample Difficulty

Sample difficulty should remain continuous rather than hard categorical.
Use smooth gates for Safe, Repairable, Broken, and Harmful regions.  The
repairable weight should train the proposal on samples where an in-cone repair
has meaningful positive value.  Safe, Broken, and Harmful weights should either
encourage no-op behavior or suppress damaging proposals.

## Current Hybrid Training Contract

The current `hsl_hybrid` contract is:

- Supervised/HSL loss trains \(\Delta g^{\mathrm{HSL}}_t\).
- Harmful loss suppresses unsafe proposals.
- PPO uses rollout advantage but its actor gradient is restricted to
  \(\rho_t\).
- Runtime writes \(\Delta g^{\mathrm{write}}_t =
  \rho_t \odot \Delta g^{\mathrm{HSL}}_t\).
- HSL labels are not pre-shrunk in `hsl_hybrid`; acceptance is learned by PPO.

The core conceptual split is:

\[
\text{HSL}:\quad \text{where should the corrupted root frame move?}
\]

\[
\text{HRL/PPO}:\quad \text{how much of that repair can be safely accepted under
the current dynamics?}
\]

This split prevents reward hacking by making the learnable PPO degree of
freedom isomorphic to the intended concept.  PPO can filter an HSL proposal, but
it cannot invent a new repair direction.

## Code Mapping

- Config:
  `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_mosaic_cfg.py`
- Runner rollout, label construction, temporal cache, action-cone writing:
  `source/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- PPO/supervised loss and acceptance-only PPO restriction:
  `source/rsl_rl/rsl_rl/algorithms/frontres_unified.py`
- FrontRES actor output bounding:
  `source/rsl_rl/rsl_rl/modules/front_residual_actor_critic.py`

## Pre-Implementation Checklist

- State the Design Delta before coding.
- Identify the owner of every changed variable: proposal, label, reward,
  \(\rho\) acceptance, action cone, rollout cache, or diagnostic.
- Check whether the change touches old objectives or only the active branch.
- Check whether the behavior is controlled by config or requires a command
  change.
- Check whether diagnostics still match the value actually written to GMT.

## Post-Implementation Audit

- Verify the path:
  config -> runner rollout construction -> storage fields -> algorithm update
  -> runtime write -> diagnostics.
- Run `python -m py_compile` on touched Python files when practical.
- Inspect whether old concepts remain in comments, log labels, or variable
  names in a way that can mislead future debugging.
- Explicitly report whether the training command changes.
