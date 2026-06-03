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
- **Dynamics-aware acceptance** decides a current-state-conditioned dynamic
  projection of the HSL clean-oriented candidate:
  \[
  g^{\mathrm{write}}_t =
  \Pi_{\mathrm{dyn}}(\tilde{g}^{\mathrm{HSL}}_t \mid o_t).
  \]
  Here \(\rho_t\) is not an uncertainty confidence score.  It is the
  rollout-trained acceptance field for the clean-oriented repair direction.
  The admissibility question is conditioned on the current robot state
  \(o_t\), not on the previous written reference as a substitute for state.
  The relevant discontinuity is between the current body state and the
  clean-oriented candidate induced by \(\Delta g^{\mathrm{HSL}}_t\).
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

The clean-oriented candidate is:

\[
\tilde{g}^{\mathrm{HSL}}_t =
g^{\mathrm{noisy}}_t + \Delta g^{\mathrm{HSL}}_t.
\]

The acceptance vector should be understood as the current parameterization of
that projection:

\[
\rho_t =
\pi_{\mathrm{accept}}(o_t, g^{\mathrm{noisy}}_t,
\tilde{g}^{\mathrm{HSL}}_t),
\]

\[
g^{\mathrm{write}}_t =
g^{\mathrm{noisy}}_t +
\rho_t \odot
(\tilde{g}^{\mathrm{HSL}}_t - g^{\mathrm{noisy}}_t).
\]

where \(o_t\) contains the current robot state observed by GMT/FrontRES.  The
previous written reference \(g^{\mathrm{write}}_{t-1}\) may be used as a weak
temporal prior or diagnostic if a later experiment needs it, but it is not the
main state variable in this contract.

Equivalently, because the current implementation represents
\(\tilde{g}^{\mathrm{HSL}}_t - g^{\mathrm{noisy}}_t\) as
\(\Delta g^{\mathrm{HSL}}_t\), the deployed write can be implemented as
\(\Delta g^{\mathrm{write}}_t =
\rho_t \odot \Delta g^{\mathrm{HSL}}_t\).  This is an implementation
parameterization of dynamic projection, not the primary method definition.

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
- Do not make \(g^{\mathrm{write}}_{t-1}\) the conceptual owner of dynamics
  admissibility.  Current-state observations \(o_t\) own that role.
- Do not let Safe/Broken/Harmful samples dominate the proposal direction loss.
- Do not let temporal continuity cache survive an episode reset.
- Do not remove old HRL/HSL branches unless explicitly requested.  They are
  research assets for later papers.

## Repair Regime Boundary

The current branch should distinguish three regimes:

- **In-cone repair**: the robot state remains close to the clean tracking
  manifold.  The Noisy-to-Clean HSL direction is valid, and \(\rho_t\) should
  mostly accept the proposal.
- **Continuity-limited repair**: the HSL direction is still useful, but the
  current body state cannot safely accept the full clean-oriented candidate.
  \(\rho_t\) should partially accept the proposal by dimension.
- **Out-of-cone recovery**: the robot state has left the recoverable
  neighborhood.  A root-level \(\Delta SE(3)\) correction may preserve
  executability, but it cannot guarantee demo-quality restoration to Clean.

The active FEMR/FrontRES contract targets the first two regimes.  The third
regime belongs to a future state-bridging or \(\Delta q\)-level method, not to
the current root-reference residual path.

## Sample Difficulty

Sample difficulty should remain continuous rather than hard categorical.
Use smooth gates for Safe, Repairable, Broken, and Harmful regions.  The
repairable weight should train the proposal on samples where an in-cone repair
has meaningful positive value.  Safe, Broken, and Harmful weights should either
encourage no-op behavior or suppress damaging proposals.

## Perturbation Curriculum Contract

FrontRES should not train for a long time at one manually selected perturbation
strength.  A fixed high value, such as the observed frontier near
`dr_scale ~= 2.3`, can teach the model to survive or reject difficult repairs
while washing out the low-perturbation clean-restoration behavior learned during
warmup.

The curriculum should therefore sample a distribution of perturbation
difficulties instead of a single increasing scalar.  The intended distribution
contains three regions:

\[
\mathcal{D}_{easy}: \text{low perturbation for clean restoration},
\]

\[
\mathcal{D}_{frontier}: \text{near the repairable boundary for acceptance
learning},
\]

\[
\mathcal{D}_{hard}: \text{slightly beyond the boundary for robustness exposure}.
\]

The easy region must remain present after warmup.  It preserves the ability to
repair low and medium perturbations toward the clean rollout.  The frontier
region should provide the main dynamics-aware acceptance signal.  The hard
region should be sampled lightly; it exposes the model to the failure boundary
without letting broken states dominate training.

The curriculum objective is not to maximize perturbation strength as quickly as
possible.  It is to keep enough samples near the current repairable frontier:

\[
\mathrm{epLen}_{\mathrm{frontres}}
\approx
\mathrm{epLen}_{\mathrm{gmt}},
\]

while still preserving low-perturbation clean restoration.  A useful adaptive
frontier is the perturbation range where FrontRES is neither trivially better
than GMT nor already deeply broken.

The recommended sampling shape for the active `hsl_hybrid` experiment is:

\[
p(d)
=
w_e p_{\mathrm{easy}}(d)
+
w_f p_{\mathrm{frontier}}(d)
+
w_h p_{\mathrm{hard}}(d),
\]

with easy and frontier samples dominating the batch and hard samples kept as a
small tail.  A practical starting point is:

\[
w_e \approx 0.5,\qquad
w_f \approx 0.4,\qquad
w_h \approx 0.1.
\]

These weights are conceptual defaults, not paper claims.  If the frontier
samples cause `episode_frontres` to drop sharply below `episode_gmt`, lower the
frontier center or increase the easy fraction before changing the model.

The frontier center should be adaptive.  It should move up when FrontRES remains
comfortably better than GMT and move down when FrontRES becomes harmful.  This
replaces hard-coding a single `dr_scale` value.

## Checkpoint Selection Contract

Do not select the final checkpoint by default.  Strong perturbation training can
improve frontier survival while degrading clean or low-noise repair quality.
Checkpoint selection should use a fixed probe set that separately measures:

- low-perturbation clean restoration;
- medium-perturbation demo quality;
- frontier robustness / episode length;
- harmful repair rate.

For paper-oriented training, the checkpoint score should emphasize medium
perturbations while still protecting low perturbations:

\[
S_{\mathrm{paper}}
=
0.4 S_{\mathrm{mid}}
+
0.3 S_{\mathrm{low}}
+
0.2 S_{\mathrm{frontier}}
-
0.1 H.
\]

For demo-oriented training, the checkpoint score should emphasize visual
cleanliness and stability in the low-to-medium range:

\[
S_{\mathrm{demo}}
=
0.6 S_{\mathrm{low/mid-clean}}
+
0.3 S_{\mathrm{mid-survival}}
-
0.1 H.
\]

Here \(H\) is the harmful-repair score: cases where FrontRES makes the tracker
worse than the GMT baseline.  The exact probe implementation may change, but
the principle should remain: the best checkpoint is probe-selected, not simply
the latest checkpoint or the checkpoint trained at the largest perturbation.

## Current Hybrid Training Contract

The current `hsl_hybrid` contract is:

- Supervised/HSL loss trains \(\Delta g^{\mathrm{HSL}}_t\).
- Harmful loss suppresses unsafe proposals.
- PPO uses rollout advantage but its actor gradient is restricted to
  \(\rho_t\).
- Runtime parameterizes the dynamic projection as
  \(g^{\mathrm{write}}_t =
  g^{\mathrm{noisy}}_t + \rho_t \odot \Delta g^{\mathrm{HSL}}_t\).
  The short form \(\Delta g^{\mathrm{write}}_t =
  \rho_t \odot \Delta g^{\mathrm{HSL}}_t\) is only valid when it is understood
  as the residual written relative to \(g^{\mathrm{noisy}}_t\).
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

## Candidate Rollout Ranking Reward

The active method figure contains four rollout branches: Clean, Noisy,
Candidate, and Projected.  The code must preserve this distinction.  Candidate
is not only a geometric counterfactual; it is the full HSL write executed by the
frozen GMT from the same current physical state \(o_t\):

\[
g^0_t = g^{\mathrm{noisy}}_t,
\qquad
g^1_t = g^{\mathrm{noisy}}_t + \Delta g^{\mathrm{HSL}}_t,
\qquad
g^\rho_t = g^{\mathrm{noisy}}_t + \rho_t \odot \Delta g^{\mathrm{HSL}}_t.
\]

Here \(g^0_t\) is the no-write baseline, \(g^1_t\) is the Candidate/full-write
branch, and \(g^\rho_t\) is the Projected/write branch.  The current robot state
\(o_t\) is not a reference endpoint.  It is the dynamics anchor that determines
whether a reference causes a state-reference discontinuity.  Noisy itself may be
dynamically fractured because \(g^{\mathrm{noisy}}_t\) can jump away from the
unchanged physical state \(o_t\).

The HRL/PPO reward should therefore compare three current-state-conditioned
scores:

\[
J_0 = J(g^0_t \mid o_t),
\qquad
J_1 = J(g^1_t \mid o_t),
\qquad
J_\rho = J(g^\rho_t \mid o_t).
\]

The core ranking quantities are:

\[
A_{\mathrm{full}} = J_1 - J_0,
\qquad
A_\rho = J_\rho - J_0,
\qquad
A_{\mathrm{proj}} = J_\rho - J_1.
\]

The acceptance reward should make \(\rho_t\) a line-search variable over the HSL
residual line:

- if \(A_{\mathrm{full}} > 0\) and \(A_\rho < A_{\mathrm{full}}\), the policy is
  under-writing a useful HSL candidate and should increase acceptance;
- if \(A_{\mathrm{full}} < 0\), the full HSL candidate is harmful from the
  current state and acceptance should decrease;
- if \(A_{\mathrm{proj}} > 0\), the projected write is better than full-write and
  HRL has performed a real dynamics-aware projection.

This reward should replace projected-only reward in the active `hsl_hybrid`
branch.  Sample reweighting may still be built from Clean vs. Noisy damage, and
HSL supervised loss should still train the clean-oriented direction.  The new
rollout contract is a quartet:

- Projected: \(g^\rho_t\), receives PPO actor gradient through \(\rho_t\);
- Candidate: \(g^1_t\), no policy gradient, provides full-write boundary;
- Noisy: \(g^0_t\), no policy gradient, provides no-write baseline;
- Clean: \(g^{\mathrm{clean}}_t\), no policy gradient, provides ideal behavior
  and sample-difficulty calibration.

The reward implementation should log \(J_0\), \(J_1\), \(J_\rho\),
\(A_{\mathrm{full}}\), \(A_\rho\), and \(A_{\mathrm{proj}}\).  If \(\rho_t\)
remains near 0.5 while \(A_{\mathrm{full}} \gg A_\rho\), the ranking reward is
still failing to push acceptance toward the useful full-write branch.

## Implementation Design Delta

This delta defines the next code change.  It should be checked before editing
FrontRES training or runtime code.

### What Changes

The active `hsl_hybrid` implementation should make \(\rho_t\) genuinely
current-state-conditioned.  The proposal path may continue to use the
FrontRES/reference subset, but the acceptance path must see the current robot
state \(o_t\) together with the HSL clean-oriented candidate:

\[
\rho_t =
\pi_{\mathrm{accept}}(o_t, g^{\mathrm{noisy}}_t,
\tilde{g}^{\mathrm{HSL}}_t).
\]

This changes the implementation from a single residual head that happens to
output proposal and acceptance values into a two-role structure:

- proposal role: predict \(\Delta g^{\mathrm{HSL}}_t\);
- acceptance role: predict the dynamic projection coefficient \(\rho_t\).

### Invariants

- The output interface remains twelve-dimensional:
  \((\Delta x,\Delta y,\Delta z,\Delta r,\Delta p,\Delta yaw,
  \rho_x,\rho_y,\rho_z,\rho_r,\rho_p,\rho_{yaw})\).
- Runtime still writes
  \(g^{\mathrm{write}}_t =
  g^{\mathrm{noisy}}_t + \rho_t \odot \Delta g^{\mathrm{HSL}}_t\).
- PPO must not own or update the repair direction in `hsl_hybrid`.
- HSL labels must not be pre-shrunk by an older confidence or temporal gate.
- Action-cone masks, upward-\(z\) limits, and active task dimensions must apply
  to the actual written residual, not only to diagnostics.

### Component Responsibilities

- `front_residual_actor_critic.py`: owns the proposal/acceptance network
  structure, output bounding, and any observation split between reference-only
  proposal inputs and current-state acceptance inputs.
- `frontres_unified.py`: owns PPO and supervised losses.  It must preserve the
  acceptance-only PPO gradient boundary.
- `on_policy_runner.py`: owns rollout target construction, action-cone
  projection, runtime write, reward diagnostics, and checkpoint probe records.
- Config files own whether the new split is enabled and which observation
  dimensions are visible to proposal and acceptance paths.

### Observation And Network Plan

The preferred implementation is:

- proposal head input: the existing FrontRES/reference subset used for HSL
  direction learning;
- acceptance head input: current robot observation \(o_t\), noisy/reference
  context, and a detached representation of \(\Delta g^{\mathrm{HSL}}_t\) or
  \(\tilde{g}^{\mathrm{HSL}}_t\);
- proposal output: bounded six-dimensional \(\Delta SE(3)\);
- acceptance output: sigmoid-bounded six-dimensional \(\rho_t\).

If a shared backbone is used, the code must still make the gradient boundary
explicit.  PPO actor loss may update the acceptance path, but it must not use
the acceptance reward to rewrite the proposal direction.  Passing
\(\Delta g^{\mathrm{HSL}}_t\) into the acceptance head should use a detached
tensor unless a later experiment explicitly changes this contract.

### Forbidden Implementation Shortcuts

- Do not solve current-state conditioning by adding
  \(g^{\mathrm{write}}_{t-1}\) as the main acceptance input.
- Do not let PPO gradients flow from the acceptance loss into the HSL proposal
  head.
- Do not change the output to a free \(\Delta g^{\mathrm{HRL}}_t\) generator in
  this branch.
- Do not make diagnostics report \(\rho_t\odot\Delta g^{\mathrm{HSL}}_t\) if
  runtime writes a differently masked or clipped value.
- Do not change old `ppo_hrl`, `basis_restore`, or legacy confidence branches
  unless the change is guarded by an active-branch config.

### Storage And Rollout Contract

Before editing losses, check whether minibatches need additional fields.  If the
acceptance head can recompute its inputs from stored observations and stored
FrontRES actions, storage may stay unchanged.  If it needs explicit candidate,
written residual, family id, or per-mode reward components, then both
`rollout_storage.py` and `frontres_unified.py` minibatch unpacking must be
updated together.

At minimum, rollout diagnostics should expose:

- proposal \(\Delta g^{\mathrm{HSL}}_t\);
- acceptance \(\rho_t\);
- final written residual;
- active action mask;
- per-mode executable gain;
- harmful repair rate.

### Verification

After implementation, audit the full chain:

\[
\text{config}
\rightarrow
\text{actor observation split}
\rightarrow
\text{runner rollout}
\rightarrow
\text{storage/minibatch}
\rightarrow
\text{algorithm loss}
\rightarrow
\text{runtime write}
\rightarrow
\text{diagnostics}.
\]

Run `python -m py_compile` on touched Python files when practical.  A
compile-only pass is not enough; the final check must also state whether
\(\rho_t\) really sees current-state information and whether PPO remains
acceptance-only.

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
