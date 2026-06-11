# FrontRES Design Contract

This document records the current experiment-level design contract for FrontRES.
It is more specific than the Dr.Cheng skill.  Read this before implementing
nontrivial changes to FrontRES training, rollout labels, PPO/HRL behavior, or
diagnostics.

## 2026-06-11 Fixed-DR Stress Evaluation Branch

Training and evaluation have different authority.

- **Training curriculum** should keep samples near the GMT executable frontier
  so FrontRES receives learnable gradients rather than mostly broken/no-op
  cases.
- **Stress evaluation** must deliberately sweep fixed perturbation strengths,
  including strengths beyond the training frontier, to answer whether FrontRES
  extends the executable envelope over GMT.

The evaluation branch must therefore be MOSAIC-side, checkpoint-only, and
read-only with respect to learning:

```text
load checkpoint
  -> set paired FrontRES / candidate / noisy-GMT / clean-GMT rollout layout
  -> for each fixed dr_scale
  -> apply fixed perturbation family and fixed perturbation magnitude
  -> run policy rollout without PPO update, storage update, or curriculum update
  -> write per-scale FrontRES-vs-GMT episode-length and survival metrics
```

This branch is not a replacement for RobotBridge video validation.  MOSAIC owns
quantitative stress curves because it shares the training simulator, frozen GMT,
and reference perturbation implementation.  RobotBridge should later consume the
selected checkpoints and representative perturbation strengths for presentation
or real-video demos.

Required diagnostics per fixed strength:

- `dr_scale`;
- `frontres_episode_length_mean`;
- `gmt_episode_length_mean`;
- `frontres_minus_gmt`;
- `frontres_survival_rate`;
- number of completed FrontRES and GMT episodes.

The branch is invalid if it updates PPO, changes the GMT frontier curriculum,
or mixes per-env easy/frontier/hard strengths.  Stress testing should use fixed
dr_scale values such as `1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0` and compare the
curve, not a single training log point.

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
that projection.  The preferred architecture is a single FEMR/FrontRES network
with a shared representation and two semantic heads:

\[
z_t = \phi(o_t),
\quad
\Delta g^{\mathrm{HSL}}_t = h_{\mathrm{prop}}(z_t),
\quad
\rho_t = h_{\mathrm{accept}}(z_t).
\]

This preserves the original minimal-entity design: FEMR already observes the
state variables needed to judge dynamic discontinuity, so acceptance should be a
second role of the same model rather than a new policy or residual agent.
Equivalently, the acceptance role estimates:

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

### State Router Alpha

The active action interface remains twelve-dimensional.  A new scalar
\(\alpha_t\) may be added as a non-action auxiliary head inside the same
FrontRES policy:

\[
\alpha_t =
P(\text{the current state under the Noisy/GMT continuation is leaving the
executable floor} \mid o_t).
\]

\(\alpha_t\) is not another acceptance coefficient and must not be trained from
Candidate-vs-Noisy comparison.  It answers a different question before sample
selection happens: "is the current state already unstable enough that the
clean-oriented route should be bypassed and the Stable Frame route should be
used?"  Therefore the route order is:

\[
\text{State Router } \alpha_t
\rightarrow
\begin{cases}
\text{Stable Frame route}, & \alpha_t \text{ high},\\
\text{Clean-oriented HSL/HRL route}, & \alpha_t \text{ low}.
\end{cases}
\]

The first training target for \(\alpha_t\) follows the SafeFall-style
safe/ambiguous/falling split:

- **falling / unstable**: the paired Noisy/GMT baseline branch terminates before
  timeout or its executable score drops below the floor.  Train
  \(\alpha_t \rightarrow 1\).
- **safe**: the paired Noisy/GMT baseline survives and remains above the floor.
  Train \(\alpha_t \rightarrow 0\).
- **ambiguous**: near-threshold states or timeout-only resets.  Mask them out of
  the \(\alpha\) loss.

This keeps the alpha gradient source clean.  Noisy/GMT is used as evidence for
current-state recoverability, not as a desired reference route.  Candidate,
Projected, and Clean still belong to sample selection, acceptance learning, and
reward ordering.

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
- Do not introduce a separate acceptance MLP as the default method unless an
  experiment shows that the simpler shared-network/two-head design cannot keep
  proposal and acceptance gradients separated.  A split acceptance MLP is an
  optional boundary-protection ablation, not the primary architecture.
- Do not implement \(\alpha_t\) by changing the environment action dimension.
  It is an auxiliary state-router head trained by supervised rollout labels and
  used by the runner to choose Stable Frame vs HSL/HRL route.

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

## Inertial Compatibility Contract

The acceptance head must not judge a repair only by whether it is closer to the
Clean reference.  Near the stability frontier, the current robot state may have
nontrivial linear and angular momentum.  A clean-oriented repair can then become
anti-inertial: it asks GMT to track a reference that lies against the current
motion, while the corrupted Noisy reference may accidentally preserve a small
stability margin.

The testable failure pattern is:

\[
D(g^{\mathrm{write}}_t,o_t) > D(g^{\mathrm{noisy}}_t,o_t)
\]

or

\[
C(g^{\mathrm{write}}_t,o_t) < C(g^{\mathrm{noisy}}_t,o_t)-m_I,
\]

where \(D\) is the state-reference distance and \(C\) is an inertial
compatibility score based on the inner product between state-reference error
and current anchor velocity/angular velocity.  If this pattern rises before the
observed `episode_frontres` collapse around the high-DR frontier, the failure
is not that HSL failed to estimate the clean direction.  The failure is that
the straight Noisy-to-Clean correction path is dynamically incompatible with
the already-falling robot.

The first implementation should be diagnostic-only.  Log Noisy, Projected,
Candidate, and Clean branch values for state-reference distance, inertial
compatibility, and failure-conditioned inversion rate.  Only after the
diagnostic verifies the hypothesis should the acceptance objective be modified.

The preferred repair preserves the two-head contract:

- HSL still owns the clean-oriented proposal direction.
- Acceptance owns how much of that direction is dynamically admissible under
  the current state and inertia.
- Any inertial prior should suppress or reshape \(\rho_t\), not rotate the HSL
  proposal into a new direction unless a later experiment explicitly studies a
  curved-path proposal.

A minimal inertial repair can shape the branch score:

\[
\tilde{J}^b =
J^b
-
\lambda_I
\left[
C(g^{\mathrm{noisy}}_t,o_t)
-
C(g^b_t,o_t)
+
m_I
\right]_+,
\]

or apply an acceptance suppressor:

\[
\rho^{\mathrm{write}}_t =
\rho_t \odot
\sigma
\left(
\frac{
C(g^\rho_t,o_t)-C(g^{\mathrm{noisy}}_t,o_t)-m_I
}{T_I}
\right).
\]

The score-shaping version should be preferred if time allows, because it lets
the existing acceptance head learn the rule from observed state variables.  The
explicit suppressor is a stronger hand-written prior and should be reported as
an ablation if used.

## Resume Schedule Contract

FrontRES phase schedules must be keyed to the absolute learning iteration, not
to `iteration - start_iter`.  On full resume, `start_iter` is the checkpoint
iteration.  If actor warmup, actor ramp, or critic warmup subtracts
`start_iter`, a run resumed from `model_700.pt` incorrectly replays the warmup
phase with `PPO actor weight = 0` and fixed low DR even though the checkpoint is
already in the PPO phase.

The resume sanity check is:

- `start=700` for `model_700.pt`;
- `Adaptive DR scale restored from checkpoint` near the saved frontier value;
- `PHASE: PPO + WEAK SUPERVISION`, not `CRITIC WARMUP`;
- `PPO actor weight: 1.000`, not `0.000`.

## Rollout-Calibrated Acceptance Preference

The first rollout preference implementation converted quartet branch ordering
into a binary acceptance label:

\[
J_1 > J_{\rho},J_0 \Rightarrow \rho^*=1,
\qquad
J_0 > J_{\rho},J_1 \Rightarrow \rho^*=0.
\]

This solved a real credit-assignment problem: Candidate/full-write is a
counterfactual branch, so PPO alone cannot learn that the current sampled
Projected write under-accepted a good HSL proposal.  However, binary
full/no-op supervision is too coarse for the current high-DR regime.  It does
not represent the central question:

\[
\text{How much of the HSL direction is dynamically admissible now?}
\]

The active design should therefore treat quartet preference as calibration
rather than classification.  Keep the same rollout branches:

\[
\alpha \in \{0,\rho,1\}
\]

for Noisy, Projected, and Candidate.  Do not add new rollout branches.  Let
\(J_0=0\), \(J_{\rho}\) be the projected repair gain, and \(J_1\) be the full
candidate gain.  The acceptance target is a local adjustment around the current
policy output:

\[
\rho^* =
\operatorname{clip}
\left(
\rho
+
\eta
\frac{J_1-J_{\rho}}{|J_1-J_0|+\epsilon},
0,1
\right)
\]

when Candidate clearly beats Projected and Noisy, and

\[
\rho^* =
\operatorname{clip}
\left(
\rho
-
\eta
\frac{J_0-J_{\rho}}{|J_1-J_0|+\epsilon},
0,1
\right)
\]

when Noisy clearly beats Projected and Candidate.  If Projected is the best
branch, the correct target is the current \(\rho\), not an ignored sample:

\[
J_{\rho} > J_1,J_0 \Rightarrow \rho^*=\rho.
\]

This makes `keep_win` semantically meaningful: it is evidence that an
intermediate acceptance value is better than both no-op and full-write.  The
target remains detached.  Rollout preference still trains only the acceptance
head; HSL still owns the clean-oriented proposal direction.  Inertial
compatibility diagnostics may explain why full-write fails, but the training
signal should remain rollout-calibrated rather than a hand-written angle
penalty.

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

## Boundary-Aware Stabilizing Teacher

The current failure is not only an acceptance-head optimization issue.  The
Clean-oriented Candidate can be outside the recoverable route even when the
state itself is still near the GMT frontier.  In that case a full or partial
write toward Clean teaches HRL a polluted signal: the correct short-term target
is not on the straight Noisy-to-Clean path.

This update must not add rollout branches, policy heads, or a separate dynamics
model.  It only reuses the existing quartet evidence and changes the candidate
route that HRL scales:

\[
g^{\rho}_t
=
g^{\mathrm{noisy}}_t
+
\rho_t \odot \Delta g^{\mathrm{selected}}_t,
\qquad
\Delta g^{\mathrm{selected}}_t
\in
\{\Delta g^{\mathrm{HSL}}_t,\Delta g^{\mathrm{stable}}_t\}.
\]

The selection rule has two separate signals:

- `Executable Floor`: the recoverability boundary of the frozen GMT.  It is
  not Noisy and not Clean.  The first implementation should use existing
  `damage_gap`, Safe/Repairable/Broken gates, GMT `ep_len`, survival, and the
  adaptive `dr_scale` frontier.  The controller should sample around the
  frontier instead of increasing `dr_scale` without bound.
- `Candidate Margin`: whether the Candidate/full-HSL branch remains above this
  floor.  It must be computed from the existing Candidate rollout and
  executability score, not from `Candidate - Noisy`.  Noisy is only a no-write
  counterfactual diagnostic; it is not the safety baseline.

The active engineering contract is:

```text
if Candidate is above the executable floor:
    selected candidate = Clean-oriented HSL correction
else if the batch/sample is near the recoverable frontier:
    selected candidate = Stabilizing correction
else:
    keep the update conservative and avoid strong repair supervision
```

The stabilizing correction is deterministic privileged reference construction,
not a new policy.  Its first version should be deliberately small and physical:
upright roll/pitch, conservative yaw, no aggressive upward `z`, and no forced
global `xy` tracking.  It represents a stable standing manifold used as a
short-term recovery route before returning to Clean.

Projected does not need a new rollout branch.  It becomes the policy acceptance
applied to the selected route:

```text
normal:     Projected = Noisy + rho * HSL(Clean route)
dangerous:  Projected = Noisy + rho * Stable route
```

Required diagnostics:

- `frontier mix`: easy/frontier/hard DR sample mode and effective scale;
- `candidate floor`: pass fraction and margin relative to the executable floor;
- `stable route`: fraction of samples for which HRL scaled the stable route.

These diagnostics prove that the mechanism is live.  If `stable_route_frac` is
always zero, the new teacher is inactive.  If `candidate_floor_pass` is always
near zero, the perturbation distribution is too hard.  If it is always near one,
the run is not probing the frontier.

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

## Rollout Preference Learning for Acceptance

The current test run shows a conceptual gap in the quartet reward design.
Candidate rollout ranking can reveal the local ordering
\(J_1 > J_\rho > J_0\), but a pure PPO update only reinforces the sampled
projected action \(g^\rho_t\).  Candidate/full-write \(g^1_t\) is evaluated as a
counterfactual branch, not sampled by the policy, so it does not automatically
become an action target for \(\rho_t\).  The missing bridge is:

\[
\text{rollout ordering}
\rightarrow
\text{acceptance preference target}.
\]

For the active test branch, treat HRL acceptance as a local line-search problem
over the HSL residual line:

\[
g(\rho) = g^{\mathrm{noisy}}_t + \rho \odot \Delta g^{\mathrm{HSL}}_t.
\]

Each sample compares three points under the same current state \(o_t\), noisy
reference, and HSL direction:

- no-write: \(J_0 = J(g^0_t \mid o_t)\);
- current projected-write: \(J_\rho = J(g^\rho_t \mid o_t)\);
- full candidate-write: \(J_1 = J(g^1_t \mid o_t)\).

The test-run learning rule is preference-supervised acceptance:

| Local rollout ordering | Acceptance target |
| --- | --- |
| \(J_1\) clearly beats \(J_\rho\) and \(J_0\) | push \(\rho_t \rightarrow 1\) |
| \(J_0\) clearly beats \(J_\rho\) and \(J_1\) | push \(\rho_t \rightarrow 0\) |
| \(J_\rho\) clearly beats both endpoints | keep current \(\rho_t\) or give low loss |
| score margins are small | ignore or downweight the sample |

This is not a replacement for HSL.  HSL still owns the clean-oriented repair
direction \(\Delta g^{\mathrm{HSL}}_t\).  The preference loss only trains the
acceptance field \(\rho_t\).  PPO can remain as a later fine-tuning or
regularization signal, but the primary test-run HRL signal should be the
counterfactual rollout preference converted into an acceptance target.

### Test-Run Implementation Plan

1. Keep the four rollout branches already implemented:
   Projected, Candidate, Noisy, and Clean.
2. Compute executable scores \(J_\rho\), \(J_1\), and \(J_0\) from the same
   current-state-conditioned executable metric currently used for ranking
   diagnostics.
3. Build a detached acceptance target for the six acceptance channels:
   \(y_\rho = 1\) when full-write clearly wins, \(y_\rho = 0\) when no-write
   clearly wins, and no strong target when the projected branch already wins or
   the margins are small.
4. Apply this loss only to the acceptance channels.  Do not let it update the
   HSL proposal direction.
5. Gate the loss by a margin threshold and the existing sample-reweighting
   window so safe/noisy-equivalent samples and deeply broken samples do not
   dominate the acceptance head.
6. Keep the current PPO acceptance objective available behind a config flag, but
   do not rely on it as the only learning signal in this test branch.

### Acceptance Input Contract

Rollout preference learning is not only a target-construction change.  The
acceptance head must receive the variables that define the local line-search
problem.  Otherwise the target can say "write more" or "write less" during
training, but the network cannot infer the rule at test time.

The shared FEMR representation and acceptance head should represent:

- current physical state \(o_t\): root state, velocity, contact or phase cues,
  and other GMT observation features that determine dynamic admissibility;
- noisy reference context \(g^{\mathrm{noisy}}_t\): the reference frame that GMT
  would execute without repair;
- enough information to infer the HSL candidate direction
  \(\Delta g^{\mathrm{HSL}}_t\) and the corresponding full-write branch
  \(g^1_t = g^{\mathrm{noisy}}_t + \Delta g^{\mathrm{HSL}}_t\).  In a two-head
  implementation this may be carried by the shared representation; if explicit
  proposal features are fed to the acceptance head, they should be detached from
  the acceptance-preference loss;
- optional scalar difficulty features such as damage gap, active perturbation
  family, jump/contact gate, and action-cone saturation.

The rollout scores \(J_0\), \(J_\rho\), and \(J_1\) should not be required at
deployment time.  They are training labels and diagnostics, not policy inputs.
The policy input must instead contain enough state-reference information for
the acceptance head to predict the same preference from observation.

For the current test run, the preferred minimal implementation is a single
FEMR/FrontRES network with two heads:

\[
z_t = \phi(o_t),\qquad
\Delta g^{\mathrm{HSL}}_t = h_{\mathrm{prop}}(z_t),\qquad
\rho_t = h_{\mathrm{accept}}(z_t).
\]

This preserves the architecture without adding a second policy or residual
network: HSL owns the repair direction, while the acceptance head owns the
current-state-conditioned decision of how much of that direction is dynamically
admissible.  If a later implementation passes explicit proposal features into
the acceptance head, they should be detached from the acceptance-preference loss
unless that experiment deliberately changes the gradient contract.

### Required Diagnostics

The next test run should print and log:

- mean \(J_0\), \(J_\rho\), \(J_1\);
- fractions of samples whose preference target is full-write, no-write, keep,
  and ignored;
- acceptance loss magnitude;
- \(\rho_{\mathrm{pos}}\) and \(\rho_{\mathrm{rpy}}\);
- correlation between \(\rho_t\) and the preferred direction;
- current PPO actor weight, so PPO fine-tuning can be separated from preference
  supervision.

The expected early test-run behavior is simple: when \(J_1 \gg J_\rho\) for
local-rp samples, \(\rho_{\mathrm{rpy}}\) should move above 0.5 without needing a
long PPO exploration phase.  If \(J_1\) remains better but \(\rho_t\) stays near
0.5, the preference target or gradient boundary is still not connected to the
acceptance head.

### Class-Balanced Focal Preference Loss

The current rollout-preference implementation can still fail even when the
quartet evidence is correct.  The runner may report a reasonable local ordering
distribution, but after per-dimension masks and sample gates the minibatch loss
can become dominated by no-write labels.  In that case the acceptance head
learns a conservative default instead of the intended local line-search rule.
This is a sample-weighting failure, not evidence that the two-head architecture
or rollout preference design is conceptually wrong.

The active fix is to treat acceptance preference learning as a hard-example and
class-imbalance problem.  Keep the existing target construction:

\[
J_1 \text{ wins} \rightarrow y_\rho=1,\qquad
J_0 \text{ wins} \rightarrow y_\rho=0,
\]

with projected-wins and small-margin samples ignored or downweighted.  Replace
plain masked BCE with class-balanced focal BCE:

\[
L_{\mathrm{pref}}
=
w_y (1-p_t)^\gamma
\operatorname{BCE}(\rho_t, y_\rho),
\]

where \(p_t=\rho_t\) when \(y_\rho=1\), and
\(p_t=1-\rho_t\) when \(y_\rho=0\).  The class weights are computed within the
active minibatch after acceptance masks and active task dimensions are applied:

\[
w_{\mathrm{full}}
=
\frac{M}{2M_{\mathrm{full}}},
\qquad
w_{\mathrm{noop}}
=
\frac{M}{2M_{\mathrm{noop}}},
\]

then clamped to a small range such as \([0.5, 3.0]\).  This makes full-write
and no-write preference labels comparable in optimization mass without changing
which samples are considered valid.  The focal term should start conservatively,
with \(\gamma=1.0\), so that wrong or uncertain acceptance decisions receive
more gradient while easy labels do not dominate.

This mechanism has a narrow authority boundary:

- it may change only the weighting of acceptance preference BCE terms;
- it must not change \(J_0,J_\rho,J_1\), margin rules, rollout branches, or HSL
  proposal targets;
- it must update only the acceptance head or acceptance rows, preserving the
  proposal/acceptance gradient boundary;
- it must not feed rollout scores into deployment inputs.

The diagnostics must expose both raw and effective class balance.  In addition
to the existing full/noop/mask/rho/error/correlation numbers, the runner should
print the focal \(\gamma\), full/noop class weights, and the effective full
fraction after class and focal weights.  A healthy early run should no longer
show an algorithm-side effective target collapsed to no-op when the runner-side
preference distribution contains many full-write wins.

### Design Audit Before Implementation

The preference-learning design is complete only if the following chain is
closed.  Implementing only one link is unsafe.

| Link | Required state |
| --- | --- |
| Policy input | The active method should be one FEMR/FrontRES network with a shared state representation and two semantic heads: proposal and acceptance.  The acceptance head must have access, through the shared representation or explicit detached features, to the state/reference variables that determine dynamic admissibility. |
| Rollout evidence | The runner executes Projected, Candidate, Noisy, and Clean under synchronized motion/frame state. |
| Preference label | The runner converts \(J_0,J_\rho,J_1\) into detached target, mask, margin, and class fractions. |
| Storage | The target and mask must be stored with each rollout transition, or recomputable exactly from stored tensors.  Logging-only tensors are not enough. |
| Algorithm loss | `frontres_unified.py` must add an acceptance preference loss to the minibatch update.  The loss must update only the acceptance head or acceptance rows. |
| Gradient boundary | The preference loss must not update \(\Delta g^{\mathrm{HSL}}_t\).  Detached proposal input is necessary but not sufficient; the loss path must be audited. |
| PPO relation | PPO acceptance loss may remain enabled, but it is secondary in the test branch.  Preference supervision is the primary signal for moving \(\rho_t\). |
| Deployment | \(J_0,J_\rho,J_1\) are not deployment inputs.  The deployed policy must infer acceptance from \(o_t\), noisy reference context, and HSL proposal features. |
| Diagnostics | Console and TensorBoard must prove class balance, target direction, acceptance loss, \(\rho\) movement, and gradient placement. |

The implementation target is therefore not another residual network.  The core
implementation requirement is the preference-label storage and acceptance-only
loss path that turns quartet rollout ordering into a trainable update while
preserving the proposal/acceptance gradient boundary.

### Inertial Preference Teacher

The next validation target is not a harder sample miner.  It is a more faithful
teacher for acceptance.  Local reference perturbations are random, so the noisy
frame is not always dynamically worse than the clean-oriented repair.  Sometimes
the noisy frame lies along the robot's current velocity or angular-velocity
trend and preserves short-horizon stability margin.  A clean-oriented HSL
repair can then be anti-inertial: it moves the reference back toward Clean but
against the current robot motion, consuming margin.

The active acceptance teacher should therefore rank branches by executable gain
after an inertial compatibility penalty, not by rollout gain alone.  For each
branch \(b \in \{0,\rho,1\}\), compute the compatibility \(C^b\) between the
branch state-reference displacement and the current robot velocity/angular
velocity.  Positive compatibility means the branch lies along the current
inertial trend; lower compatibility than Noisy means the branch asks the robot
to reverse harder than the noisy reference does.  Define

\[
P_I^b = [C^0 - C^b + m_I]_+ .
\]

Then shape the preference score:

\[
\tilde J^b = J^b - \lambda_I P_I^b .
\]

The existing acceptance-target rule remains unchanged, but it must use
\(\tilde J_0,\tilde J_\rho,\tilde J_1\) instead of raw \(J_0,J_\rho,J_1\).
Thus full-write pushes \(\rho\) upward only when it is both useful and
inertially admissible.  No-write pushes \(\rho\) downward when the repair branch
is not worth its inertial cost.  Projected-write can remain the local best point
when partial acceptance preserves the useful repair while avoiding an
anti-inertial full write.

This change has a narrow authority boundary:

- it changes only the detached rollout-preference teacher;
- it does not change the HSL proposal, rollout branches, network architecture,
  storage format, or deployed action formula;
- it updates only the acceptance head through the existing preference loss;
- it should be guarded by config so the raw-gain teacher remains available as
  an ablation.

Required diagnostics are the original full/noop/keep/ignore fractions plus the
mean inertial penalty applied to Projected and Candidate.  A healthy validation
run should show fewer false full-write targets on anti-inertial samples and a
less negative correlation between gate and inertial gain.

### Direct Conditional Acceptance Target

The score-shaping teacher above is still an indirect translation of the concept.
The clearer concept separates desired repair from dynamic feasibility:

\[
\rho^* = \Pi_{[0,a]}(n).
\]

Here \(n=\mathrm{repair\_need}\) is the desired write fraction implied by
Clean geometry, and \(a=\mathrm{admissibility}\) is the current-state upper
bound on how much of that repair can be safely accepted.  The one-dimensional
projection has the closed form

\[
\rho_i^* = \min(n_i, a).
\]

This avoids the false conservatism of multiplying two soft quantities.  The
product \(n_i a\) treats repair necessity and dynamic admissibility as two
independent probabilities in an AND gate; medium values then collapse toward
no-op.  The projection view is closer to safe-action and constrained-policy
methods: the proposal gives the desired action strength, and the dynamic
condition clips it to the admissible set.

For task-space FrontRES, compute repair need from Clean geometry, not from
executable reward:

\[
n_i =
\left[
\frac{|e^{0}_i| - |e^{1}_i|}{|e^{0}_i|+\epsilon}
\right]_{0}^{1},
\]

where \(e^0\) is the Noisy-to-Clean task-space error and \(e^1\) is the
Candidate-to-Clean error after the **actual Candidate branch command** has been
written.  Do not compute \(e^1\) from the raw sampled action alone: the action
may be changed by mode masks, active dimensions, scale/clamp, projection, or
command-write constraints before it becomes the Candidate reference.  Samples
with very small \(|e^0|\) should be ignored or heavily downweighted because no
repair is needed.

Compute admissibility from inertial compatibility:

\[
a =
\sigma\left(\frac{C^1-C^0-m_I}{T_I}\right),
\]

where \(C^0\) is the compatibility of the Noisy branch and \(C^1\) is the
compatibility of the actual Candidate branch.  The target becomes

\[
\rho_i^* = \min(n_i, a).
\]

This target is more aligned than endpoint classification:

- it does not add rollout branches;
- it does not fit a numerical surrogate unrelated to the concept;
- it separates "does this repair help Clean?" from "can the current body state
  accept it?";
- it treats admissibility as a constraint on desired repair strength, not as a
  second multiplicative penalty;
- it still updates only the acceptance head through the existing
  `acceptance_target` / `acceptance_mask` path.

For the active validation branch, this direct target should replace
full/noop/keep endpoint calibration.  The older score-shaping rule should remain
available as a config fallback.

Expected diagnostic signature after this change:

- at the per-sample/per-dimension target-construction level,
  `target_i = min(need_i, admissibility)` on active dimensions.  Console means
  are aggregated after masking, so `mean(target)` should be bounded by the
  printed active-dimension `mean(need)` and `mean(admiss)`, but it does not have
  to equal `min(mean(need), mean(admiss))`;
- if debug samples show the actual Candidate command nearly reaches the Clean
  target, `need` should be high rather than near zero;
- if `need` is high but `admiss` is moderate, `tgt` should remain moderate
  instead of being multiplicatively crushed.

## Implementation Design Delta

This delta defines the next code change.  It should be checked before editing
FrontRES training or runtime code.

### What Changes

The active `hsl_hybrid` implementation should make \(\rho_t\) genuinely
current-state-conditioned while keeping the architecture minimal.  The preferred
design is a single FEMR/FrontRES network with a shared state representation and
two heads:

\[
z_t = \phi(o_t),\qquad
\Delta g^{\mathrm{HSL}}_t = h_{\mathrm{prop}}(z_t),\qquad
\rho_t = h_{\mathrm{accept}}(z_t).
\]

This changes the implementation from one undifferentiated residual output into
a two-role structure inside the same model:

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

- `front_residual_actor_critic.py`: owns the single FEMR/FrontRES network, the
  proposal and acceptance heads, output bounding, and any optional observation
  split.  A separate acceptance MLP is optional ablation machinery, not the
  default method.
- `frontres_unified.py`: owns PPO and supervised losses.  It must preserve the
  acceptance-only PPO gradient boundary.
- `on_policy_runner.py`: owns rollout target construction, action-cone
  projection, runtime write, reward diagnostics, and checkpoint probe records.
- Config files own whether optional split-MLP ablations are enabled and which
  observation dimensions are visible to proposal and acceptance paths.

### Observation And Network Plan

The preferred implementation is:

- shared FEMR input: the current observation \(o_t\), including robot state,
  noisy/reference context, and the features already used for HSL direction
  learning;
- proposal output: bounded six-dimensional \(\Delta SE(3)\);
- acceptance output: sigmoid-bounded six-dimensional \(\rho_t\).

The code must make the gradient boundary explicit.  HSL supervision may update
the shared representation and proposal head.  Rollout-preference acceptance loss
should update only the acceptance head, or at minimum must be audited so it
cannot rewrite the proposal direction.  If explicit \(\Delta g^{\mathrm{HSL}}_t\)
features are passed into the acceptance head, they should use a detached tensor
unless a later experiment explicitly changes this contract.

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

## 2026-06-09 GMT Frontier Search And HRL Signal Decoupling

### Design Delta

The next change separates two concepts that were previously entangled:

1. GMT capability frontier discovery.
2. HRL acceptance/noop learning, which remains discussion-only in this edit.

The GMT frontier is a property of the frozen tracker under corrupted references.
It must be estimated from GMT-only evidence, not from FrontRES, HSL, HRL, or a
stable-route fallback.  The HRL acceptance branch then trains near this frontier,
where samples are neither trivially safe nor already unrecoverable.

Current implementation scope: execute GMT frontier search only.  The HRL branch
analysis below is a design contract for discussion, not a live code change in
this edit.

The HRL branch currently has one live learnable authority: the per-axis
acceptance \(\rho_t\).  It can express:

\[
\Delta g_t^{\mathrm{written}}
=
\rho_t \odot \Delta g_t^{\mathrm{HSL}},
\]

but it cannot create a new route.  Therefore the first implementation should not
pretend that HRL has a separate route-selection action unless a new output head
is explicitly added.  In the resume-compatible branch, stable route remains an
external non-credit fallback and a diagnostic.  The learnable change is to give
\(\rho_t\) a clean no-op / partial-write / full-write target from rollout
ordering.

### GMT Frontier Search Contract

The objective is to estimate the largest perturbation scale that GMT can still
execute:

\[
s^\star
=
\max s
\quad
\text{s.t.}
\quad
\mathrm{Exec}_{\mathrm{GMT}}(s) \ge \tau.
\]

Here \(\mathrm{Exec}_{\mathrm{GMT}}\) should be computed from the Noisy/GMT
baseline rollout only.  A practical score is the normalized episode length or
survival rate of the baseline environments:

\[
\mathrm{Exec}_{\mathrm{GMT}}(s)
=
\frac{\mathrm{ep\_len}_{\mathrm{GMT}}(s)}
       {\mathrm{ep\_len}_{\mathrm{ref}}},
\]

where \(\mathrm{ep\_len}_{\mathrm{ref}}\) is a robust running reference such as
the configured episode length, clean/low-DR episode length, or a moving maximum
from recent baseline rollouts.

Because the observed boundary is sharp near \(dr\_scale \approx 2.381\), the
frontier controller should behave like bracketed search rather than a slow
reward PI controller:

- maintain `safe_low`, the largest scale currently judged executable;
- maintain `broken_high`, the smallest scale currently judged broken;
- probe the current scale;
- if GMT score is above the safe threshold, move `safe_low` up;
- if GMT score is below the broken threshold, move `broken_high` down;
- set the training frontier to `safe_low` or a conservative point inside the
  bracket.

For training, the frontier can be surrounded by strength classes:

\[
s_{\mathrm{easy}} = f_e s_f,\qquad
s_{\mathrm{frontier}} = s_f,\qquad
s_{\mathrm{hard}} = f_h s_f,
\]

but diagnostics must distinguish target mixture from realized behavior.  A
single per-iteration label is not enough.  The log must expose:

- current `safe_low`, `broken_high`, and `frontier`;
- probe scale, GMT executable score, and controller decision;
- easy/frontier/hard scales;
- target and realized strength proportions when batch-level mixing is active.

### HRL Reward/Preference Contract

Reward design is treated as rollout sorting under a physical rule:

> In the same current state, the better route is the one that makes frozen GMT
> more executable and less likely to fall.

This gives the clean ranking variables:

\[
J_0 = \mathrm{Exec}(\mathrm{Noisy}) - \mathrm{Exec}(\mathrm{Noisy}) = 0,
\]

\[
J_\rho = \mathrm{Exec}(\mathrm{Projected}) - \mathrm{Exec}(\mathrm{Noisy}),
\]

\[
J_1 = \mathrm{Exec}(\mathrm{Candidate}) - \mathrm{Exec}(\mathrm{Noisy}).
\]

The acceptance head should learn from this ordering only when the ordering is
clean.  No-op is not an afterthought; it is one of FEMR's required behaviors.
Therefore safe samples and deeply broken samples must be allowed to produce a
clear no-op target for \(\rho_t\), while repairable samples may produce
partial/full-write targets.  Ambiguous samples should be masked out.

The live resume-compatible branch should use:

- safe or broken samples: target \(\rho_t \rightarrow 0\) when no route has a
  reliable positive gain;
- repairable samples: compare \(J_0,J_\rho,J_1\) and train toward the winning
  write strength;
- candidate floor failure: do not create a false full-write target from
  Candidate.  Stable route may be applied as a non-credit fallback, but the
  acceptance loss should not reward \(\rho_t\) for an external oracle route.

This separates the learnable authority:

- HSL learns the clean-oriented geometric proposal
  \(\Delta g_t^{\mathrm{HSL}}\);
- HRL learns no-op / partial-write / full-write through \(\rho_t\);
- stable route remains an external route fallback until a route-selection head
  is explicitly added.

### Current Implementation Scope

This edit executes only the GMT frontier-search part.  It should remain
checkpoint-compatible:

- do not change the actor output dimension;
- add frontier-search state and diagnostics in the runner;
- compute the frontier score from Noisy/GMT baseline episode length only;
- print a sentinel that proves the live path:

```text
GMT frontier probe: scale=..., score=..., decision=..., bracket=safe/broken
DR train mix scales e/f/h: ...
```

### Deferred HRL Discussion Items

The following points are conceptual only in this edit and should not be treated
as implemented:

- whether safe/broken samples should produce explicit no-op targets for
  \(\rho_t\) or be separated by a route/mode variable;
- whether Candidate floor failure should mask acceptance learning, create a
  no-op target, or trigger an external stable fallback;
- whether \(\rho_t\)-only HRL is expressive enough, or an explicit route head is
  needed later.

If a later experiment adds explicit route selection, it must be a new guarded
branch with storage/loss/runtime diagnostics for the route action.  It should
not be silently mixed into the current \(\rho_t\)-only HRL branch.

## 2026-06-10 Per-Env Mixed DR Strength Curriculum

### Design Delta

The previous mixed-DR implementation mixed `easy/frontier/hard` strength across
iterations, not inside a single rollout batch.  This is insufficient for HRL
acceptance learning.  When the current global `dr_scale` is judged broken, the
whole batch becomes broken:

```text
Candidate floor pass low -> HRL mostly sees noop -> alpha/stable route dominates.
```

The curriculum should instead sample perturbation strength per paired training
sample.  This mirrors automatic curriculum / domain randomization practice: keep
coverage around the learnable frontier while preventing broken states from
monopolizing the batch.

### Sampling Contract

For every training sample \(i\), sample a class:

\[
c_i \in \{\mathrm{easy}, \mathrm{frontier}, \mathrm{hard}\}.
\]

Then assign a strength:

\[
s_i =
\begin{cases}
f_e s_f, & c_i=\mathrm{easy},\\
f_f s_f, & c_i=\mathrm{frontier},\\
f_h s_f, & c_i=\mathrm{hard},
\end{cases}
\]

with default proportions:

```text
easy/frontier/hard = 0.50 / 0.40 / 0.10
```

The paired rollout branches must share the same \(s_i\):

```text
Train(i), Candidate(i), GMT-baseline(i): same sampled dr_scale
Clean(i): no perturbation
```

This preserves the causal comparison among Noisy, Projected, Candidate, and
Clean.  The change only affects perturbation sampling; it does not add a new
rollout branch or change the FrontRES action dimension.

### Frontier Resume Guard

When resuming from an old checkpoint, the stored frontier may equal a scale that
the current GMT probe immediately judges broken.  In that case the bracket must
not get stuck at:

```text
safe_low == broken_high == current_scale
```

If a probe is broken at or below the current `safe_low`, retreat `safe_low`
conservatively and probe inside the reopened bracket.  This treats the resumed
scale as an unverified prior, not as a proven safe boundary.

### Implementation Contract

- Add per-env DR scales to the motion perturber.
- Keep perturbation family masks per-env as before.
- Let the runner write one vector of scales for Train/Candidate/GMT-baseline
  branches and zero/disable Clean through the existing baseline mask.
- Keep GMT frontier bracket updates tied to frontier-class Noisy/GMT baseline
  episode lengths only.  Easy and hard samples are useful for training coverage,
  but they must not be averaged into the single-scale boundary probe.
- Keep the old scalar DR path as a fallback when per-env mixed strength is
  disabled.
- Print realized batch proportions and mean scale:

```text
DR train mix e/f/h: ...
DR train scale mean: ...
```

### Expected Diagnostic Change

After this change, a broken frontier probe should no longer imply that every
training sample is broken.  A healthy run should show:

- nonzero easy/frontier/hard proportions in the same iteration;
- higher `cand floor pass` than the all-broken batch;
- lower `accept pref noop` if repairable samples re-enter the batch;
- alpha/stable route active only on the high-risk subset, not because the entire
  batch is broken.

## 2026-06-10 Stable-to-Repair HRL Reparameterization

### Design Delta

The previous HRL parameterization made \(\rho_t\) search on the straight line
from the corrupted Noisy reference to the HSL Repair candidate:

\[
g_t^{\rho}
=
g_t^{\mathrm{noisy}}
+
\rho_t \odot
\left(g_t^{\mathrm{repair}}-g_t^{\mathrm{noisy}}\right).
\]

This made \(\rho_t=0\) equivalent to no-op.  Near the dynamic frontier, no-op
can win even when the right short-term action is not "do nothing" but "first
move to a stable executable reference, then return toward the repair target."
Thus \(\rho_t\) was forced to encode route rejection, no-op, and repair strength
at the same time.

The active experiment should instead make \(\rho_t\) search between a
deterministic Stable Frame and the HSL Repair frame:

\[
g_t^{\rho}
=
(1-\rho_t)\odot g_t^{\mathrm{stable}}
+
\rho_t\odot g_t^{\mathrm{repair}}.
\]

Equivalently, in residual form relative to the Noisy reference:

\[
\Delta g_t^{\rho}
=
\Delta g_t^{\mathrm{stable}}
+
\rho_t\odot
\left(
\Delta g_t^{\mathrm{repair}}
-
\Delta g_t^{\mathrm{stable}}
\right).
\]

This changes the concept of \(\rho_t\):

- \(\rho_t=0\): choose the conservative executable Stable Frame;
- \(\rho_t=1\): choose the Clean-oriented HSL Repair frame;
- \(0<\rho_t<1\): release tracking demand from stability toward repair.

This is not a new rollout branch and not a new policy head.  It is a new
coordinate system for the existing six-dimensional acceptance vector.

### Component Ownership

- **HSL** still owns \(g_t^{\mathrm{repair}}\), the Clean-oriented repair
  candidate.
- **Stable Frame constructor** owns \(g_t^{\mathrm{stable}}\), a deterministic
  upright/conservative executable reference.  It is privileged engineering
  structure, not a learnable policy.
- **HRL / PPO acceptance** owns only the interpolation coefficient \(\rho_t\)
  between Stable and Repair.  It no longer treats \(\rho_t=0\) as an empty
  no-op; \(\rho_t=0\) has a physical meaning.
- **State Router Alpha** is demoted to a safety override / diagnostic.  The
  primary continuous route choice is now \(\rho_t\).  If alpha remains enabled,
  it may force the stable endpoint in extreme states, but it must not be
  required for ordinary Stable-to-Repair interpolation.

### Implementation Contract

- Preserve the old Noisy-to-Repair write rule behind a config fallback.  Do not
  delete the old branch.
- Add a config flag:

```text
frontres_rho_space = "stable_to_repair"
```

  with fallback value `"noisy_to_repair"` for old ablations.
- In `hsl_hybrid`, when `frontres_rho_space == "stable_to_repair"`, compute
  the actual Projected correction as:

```text
projected = stable + rho * (repair - stable)
```

  for both position and rotation-vector corrections.
- Candidate rollout must remain the full HSL Repair endpoint.  Noisy/GMT and
  Clean/GMT branches remain unchanged.  The quartet semantics become:

```text
Noisy/GMT:      corrupted reference, zero correction
Projected:      Stable-to-Repair interpolation controlled by rho
Candidate:      full HSL Repair endpoint
Clean/GMT:      uncorrupted reference
```

- The existing preference target path can remain unchanged initially because it
  already compares Noisy, Projected, and Candidate.  Its interpretation changes:
  a low \(\rho\) target now means "prefer the stable endpoint," not "prefer
  no-op."  Console labels should make this visible.
- Stable Route alpha override, if active, should force \(\rho=0\) / stable
  endpoint rather than represent the main routing mechanism.

### Required Diagnostics

The live log must prove the new coordinate system is active:

```text
rho space: stable_to_repair
stable endpoint frac: ...
stable route frac: ...
accept pref repair/stable/keep/ign: ...
```

Here `stable endpoint frac` is the fraction of Projected samples using the
Stable-to-Repair parameterization.  In the active branch it should be near one
whenever `frontres_rho_space == "stable_to_repair"`.  `stable route frac`
remains the alpha-forced safety override fraction.

### Validation Expectation

The first short resume test should not be judged only by episode length.  The
key sanity checks are:

- `rho space` prints `stable_to_repair`;
- `stable endpoint frac` is nonzero and normally near one;
- `accept pref stable` is no longer interpreted as useless no-op;
- `episode_frontres` should not collapse simply because \(\rho\) became low,
  because low \(\rho\) now writes a stabilizing reference.

## 2026-06-10 Tri-Anchor HRL Projection Contract

### Design Delta

The global Stable-to-Repair parameterization above was too aggressive.  It made
low \(\rho\) mean "write Stable" for every sample, including states where Noisy
was already executable by GMT.  This explains why FrontRES could fall below GMT
at moderate perturbation: conservative HRL was no longer close to no-op.

The corrected HRL contract treats Repair as the conceptual origin.  HSL proposes
the ideal Clean-oriented repair \(R\).  If this repair is not fully executable,
HRL should retreat from Repair toward one of two fallback anchors:

```text
N: Noisy / no-op fallback
S: Stable Frame fallback
R: HSL Repair endpoint
```

The projected reference is a semantic three-anchor projection:

\[
P
=
\rho R
+
(1-\rho)\left((1-\alpha)N+\alpha S\right).
\]

In residual form relative to Noisy, where \(\Delta N=0\):

\[
\Delta P
=
\rho \Delta R
+
(1-\rho)\alpha \Delta S.
\]

This keeps the variable meanings clean:

- \(\rho\): Repair retention.  It answers whether full HSL Repair is better
  than Noisy under executable rollout evidence.
- \(\alpha\): fallback direction.  It answers whether the rejected repair mass
  should retreat toward Noisy or Stable.

This is not trajectory interpolation.  It is a minimal reference-space simplex
that keeps FEMR inside its authority boundary: root/task-space correction before
frozen GMT, not a new motion generator.

### Training Signals

The two heads must receive different labels:

- \(\rho\) target comes from Candidate-vs-Noisy:

```text
rho_target = repair_need * sigmoid((E_R - E_N - margin) / tau_rho)
```

  where \(E_N\) is the Noisy/GMT executable score and \(E_R\) is the full HSL
  Candidate executable score.  If Candidate improves over Noisy, Repair should
  be retained.  If Candidate is worse, Repair should be rejected.

- \(\alpha\) target comes from Noisy-vs-floor:

```text
alpha_target = sigmoid((floor - E_N) / tau_alpha)
```

  If Noisy is still executable, rejected repair mass should retreat to Noisy.
  If Noisy is below the executable floor, rejected repair mass should retreat to
  Stable.

Alpha should matter most when Repair is rejected, so the alpha loss mask should
be weighted by:

```text
alpha_weight = 1 - rho_target
```

This prevents alpha from learning fallback direction on samples where Repair is
already clearly valid.

### Implementation Contract

- Preserve old modes:
  - `frontres_rho_space = "noisy_to_repair"` keeps the legacy residual scaling;
  - `frontres_rho_space = "stable_to_repair"` keeps the previous ablation;
  - `frontres_rho_space = "tri_anchor"` activates this contract.
- Default the current experiment to `tri_anchor`.
- Candidate rollout remains full HSL Repair.  Noisy/GMT and Clean/GMT remain
  unchanged.
- Projected rollout in `tri_anchor` must use:

```text
projected = rho * repair + (1 - rho) * alpha * stable
```

  in residual coordinates.
- The existing acceptance head remains \(\rho\).  The existing state-router head
  becomes \(\alpha\).  Do not add a new rollout branch or a new policy head.

### Required Diagnostics

The next resume test should print:

```text
rho space: tri_anchor
stable endpoint frac: 0.000
tri weights R/N/S: ...
accept pref repair/fallback/keep/ign: ...
state alpha p/t/m/r: ...
```

Expected healthy early behavior:

- safe moderate-DR samples should have nontrivial `w_N` instead of being pulled
  to Stable;
- high-risk samples with low Noisy executability should increase `w_S`;
- \(\rho\) should no longer collapse only because Stable is a safe endpoint.

## 2026-06-11 Grouped Executable Projection For 6D Rho

### Design Delta

The active `tri_anchor` branch correctly keeps \(\rho_t\) as a six-dimensional
acceptance vector, but its first target construction still used one scalar
Candidate-vs-Noisy preference and copied it to all six axes.  This makes the
network look six-dimensional while the supervision remains an overall
full-repair/fallback decision.  When one dangerous component makes full
Candidate fail, the scalar target suppresses all components and teaches a
conservative gate.

The next change makes HRL an executable projection layer:

```text
HSL: propose a 6D clean-oriented repair.
HRL: keep the executable components and suppress the harmful components.
```

The action interface does not change.  \(\rho_t\) remains:

```text
[rho_x, rho_y, rho_z, rho_roll, rho_pitch, rho_yaw]
```

The credit assignment is grouped only for target construction:

```text
planar group:   dx, dy, yaw
attitude group: roll, pitch
vertical group: dz
```

This is not a 3D output.  It is a 6D output trained with three cleaner sources
of evidence until fully per-axis counterfactual data is available.

### Signal Contract

The target should no longer ask only:

```text
Does full Candidate beat Noisy?
```

It should ask:

```text
Which repair component group has positive executable evidence?
```

Use existing rollout branches and existing executability components.  Do not
add a rollout branch.

- Planar target uses the branch improvement of the planar/xy/yaw score.
- Attitude target uses the branch improvement of the roll/pitch score.
- Vertical target uses the branch improvement of the z/vertical score.

For each group \(G\), compute a soft repair-retention target from the
group-specific Candidate-vs-Noisy improvement:

```text
rho_target_G = sigmoid((J_R_G - J_N_G - margin_G) / tau_rho)
```

Then write it into the corresponding six-dimensional entries:

```text
dx, dy, yaw      <- rho_target_planar
roll, pitch     <- rho_target_attitude
dz              <- rho_target_vertical
```

### Sample Attribution Contract

Single-family perturbation samples provide the cleanest supervision.  Reuse the
existing per-mode mask:

```text
planar perturbation   -> train dx, dy
yaw perturbation      -> train yaw
local_rp perturbation -> train roll, pitch
global_z perturbation -> train dz
```

Combined perturbations may stay in the batch, but their preference mask should
remain weaker or mode-local.  The current experiment prioritizes single-family
mixup; composite credit assignment is deferred.

### Tri-Anchor Interaction

`tri_anchor` runtime remains:

```text
projected = rho * repair + (1 - rho) * alpha * stable
```

Only the \(\rho\) target changes.  The \(\alpha\) target remains Noisy-vs-floor
and its loss mask should be weighted by rejected repair mass:

```text
alpha_weight = 1 - mean_active(rho_target_6d)
```

This keeps the roles separate:

- \(\rho\): which repair components are executable enough to retain;
- \(\alpha\): where rejected repair mass should retreat, Noisy or Stable.

### Required Diagnostics

The next resume test must prove that the six-dimensional target is no longer a
copied scalar:

```text
rho target grp p/r/z: ...
rho target spread: ...
accept pref repair/fallback/keep/ign: ...
tri weights R/N/S: ...
```

Healthy early behavior:

- `rho target spread` should be nonzero on mixed-mode batches;
- local-rp runs should mainly supervise `roll/pitch` acceptance;
- planar/yaw samples should be allowed to keep planar/yaw repair even if
  roll/pitch or z components are risky;
- HRL should become less globally conservative without increasing harmful
  repair.

## 2026-06-11 Current-Action Regret For Grouped Rho

### Problem

The grouped-rho test proved that the six-dimensional target is no longer a
copied scalar: `rho target grp p/r/z` separates planar, roll/pitch, and vertical
acceptance.  However, the training still remained conservative:

```text
Candidate gain > Projected gain
accept pref fallback high
FrontRES not consistently above GMT
```

This means the grouped target answered the wrong local question.  It asked:

```text
Does Candidate beat Noisy?
```

That is an endpoint-value question.  HRL needs a current-action question:

```text
Did the current Projected action under-write or over-write the Candidate?
```

### Signal Contract

Keep the existing quartet.  Do not add a rollout branch:

```text
Noisy:      J0
Projected:  Jrho     current HRL behavior
Candidate:  J1       full HSL write
Clean:      calibration / diagnostics
```

For each grouped repair component \(G\), compute:

```text
underwrite_G = J1_G - Jrho_G
overwrite_G  = J0_G - Jrho_G
range_G      = |J1_G - J0_G| + eps
```

The detached target is a local movement around the current policy output:

```text
if underwrite_G > margin:
    rho_target_G = rho_G + eta * underwrite_G / range_G
elif overwrite_G > margin:
    rho_target_G = rho_G - eta * overwrite_G / range_G
else:
    rho_target_G = rho_G
```

This changes the meaning of the target from endpoint classification to
current-action regret.  The target says whether the current rho wrote too little
or too much.

### Mask Contract

The previous acceptance mask became too restrictive:

```text
oracle_trust * repair_window * route_mask * per_mode_mask * active_dim_mask
```

For regret learning, boundary samples are useful.  Do not hard-delete them only
because they are outside the repairability window.  The active mask policy is:

- keep `active_dim_mask` as a hard mask;
- keep invalid rollout / missing candidate as a hard mask;
- keep stable-route oracle replacement out of rho credit with `route_mask`;
- convert repair-window gating to a soft weight with a nonzero floor;
- make per-mode attribution a soft weight, not an absolute deletion, when using
  grouped regret.

This preserves the action-cone idea without starving HRL of regret gradients.

### Diagnostics

The next run must print:

```text
rho regret up/dn p/r/z: ...
rho target grp p/r/z: ...
rho target spread/weight: ...
accept pref repair/fallback/keep/ign: ...
```

Healthy behavior:

- `rho regret up` should be positive when Candidate beats Projected;
- `accept pref repair` should increase if Candidate remains better than
  Projected;
- `rho target weight` should rise above the previous ~0.17 unless active dims
  and route credit are truly sparse;
- `rho target spread` should remain nonzero.

## 2026-06-11 Mask Authority Audit For Rho Regret

### Problem

The grouped regret target is now the cleanest available HRL signal:

```text
Did the current Projected action write too little or too much,
relative to Candidate and Noisy?
```

This signal should not be filtered by older masks whose meaning came from
sample selection or perturbation attribution.  Otherwise the system can produce
a correct regret target and still starve the rho head of gradients.

The dangerous chain was:

```text
oracle_trust * repair_window * route_mask * per_mode_mask * active_dim_mask
```

Only some of these terms have authority over the current-action regret target.

### Mask Ownership

Keep these hard boundaries:

- `active_dim_mask`: the model must not learn acceptance for disabled output
  dimensions.
- invalid rollout / missing Candidate: there is no trustworthy comparison.
- stable-route replacement mask: if the runner replaced the normal HRL route
  with an oracle stable route, the observed outcome is not credit-assignable to
  rho.

Remove these as default regret masks:

- `repair_window`: safe, repairable, and near-broken samples all carry useful
  regret.  The current-action question already tells whether rho should move up,
  down, or stay.  The repair window may remain a diagnostic or reward-shaping
  concept, but it should not delete rho gradients.
- `per_mode_mask`: perturbation family attribution is not the owner of a grouped
  rho dimension.  If planar/rp/z current-action regret can be measured from the
  quartet rollout, it is valid even when the sampled corruption family was not
  the same group.

Soften this evidence term:

- `oracle_trust`: this is evidence reliability, not a policy decision.  For rho
  regret it should be a nonzero continuous weight, not a binary delete, unless
  the rollout itself is invalid.

### Implementation Contract

For `frontres_acceptance_regret_target_enabled=True` and
`frontres_grouped_rho_target_enabled=True`:

```text
pref_weight = oracle_trust_soft * route_mask * active_dim_mask
```

where:

```text
oracle_trust_soft = oracle_floor + (1 - oracle_floor) * oracle_trust
```

The default regret floors are:

```text
frontres_acceptance_regret_soft_mask_floor = 1.0
frontres_acceptance_regret_per_mode_soft_floor = 1.0
frontres_acceptance_regret_oracle_trust_floor = 0.25
```

This means `repair_window` and `per_mode_mask` are removed from the default
current-action regret path, while oracle reliability still modulates gradient
strength.

Diagnostics should report the mean effective rho weight, not just whether the
mask is nonzero:

```text
rho target spread/weight: target_group_spread / mean_effective_weight
```

If `mean_effective_weight` rises while `rho regret up/dn` remains nonzero but
`rho` still collapses toward fallback, then the remaining problem is no longer
mask starvation.  It is the regret objective or reward decomposition itself.

## 2026-06-11 Structured Joint RL For Alpha-Rho

### Problem

The previous HRL branch repeatedly converted rollout preference into detached
targets:

```text
rho_target   -> BCE(rho_pred, rho_target)
alpha_target -> BCE(alpha_pred, alpha_target)
```

This is not the original Conditional Preference Learning concept.  It turns a
black-box executable decision into two pseudo-supervised labels.  The execution
result, however, is caused by the joint projected reference:

```text
Projected = f(alpha, rho, HSL proposal, current state).
```

If Projected succeeds or fails, the credit belongs to the sampled joint action,
not to two independently constructed labels.  Separate pseudo labels create a
concept-engineering mismatch: the method says rollout preference, but the live
loss trains target imitation.

### Design Delta

Keep the existing action interface and rollout branches.  Do not add a new
rollout branch, critic, policy head, or action dimension.

Replace pseudo-target ownership with a structured joint RL objective:

```text
alpha: route / fallback direction variable
rho:   repair-retention variable
joint action: (alpha, rho)
training signal: Projected rollout advantage relative to the trusted baseline
```

The design rule is:

```text
concepts stay separated, training responsibility is joint.
```

Alpha and rho should remain separate heads because they describe different
variables.  But when the runner executes a Projected reference produced by both
heads, the same rollout advantage must update the joint action log-probability:

```text
L_joint = - A_projected * log pi(alpha, rho | state)
```

The baseline is the paired Noisy/GMT branch already present in the quartet.
The advantage must be continuous, not binary:

```text
A_projected = U(Projected) - U(Noisy/GMT).
```

The baseline defines the zero point, not the optimization endpoint.  The utility
must preserve pressure toward higher-quality repair by including executable
gain and clean-oriented quality terms already used in the FrontRES reward
decomposition.  A projected reference that is only slightly better than Noisy
should receive a small positive advantage; a projected reference that is stable
and closer to Clean should receive a larger one.

### Implementation Contract

- Add a guarded mode:

```text
frontres_structured_joint_rl_enabled = True
```

- In this mode, do not use `acceptance_target` and `state_alpha_target` as
  conceptual truth for rho/alpha.  They may stay in the code as diagnostics or
  legacy fallbacks, but the active update should use joint-action advantage.
- Reuse the existing PPO sampled action and log-probability path for rho.
- Add alpha log-probability from the state-router Bernoulli distribution:

```text
alpha_sample = sampled route/fallback action derived from alpha probability
logp_alpha   = Bernoulli(alpha_prob).log_prob(alpha_sample)
```

  In structured-joint mode, the executed training projection must use this same
  stored `alpha_sample`, not a separate thresholded route or a detached
  continuous diagnostic probability.  Otherwise the alpha gradient would update
  an action that did not cause the observed rollout outcome.
- Store the joint advantage and alpha log-prob in the existing rollout storage
  channels without changing the action dimension:

```text
acceptance_target[:, 0] = detached joint advantage
acceptance_target[:, 1] = detached alpha log-prob
acceptance_target[:, 2] = detached alpha action in {0, 1}
acceptance_mask[:, 0]   = joint-RL sample weight
```

  In structured-joint mode these fields are not rho targets.  They are a compact
  storage carrier for the joint RL signal.  Diagnostics must make this semantic
  switch visible.
- The loss must update PPO-owned acceptance dimensions and the alpha head, while
  preserving the HSL proposal boundary.  Do not let joint RL update the
  supervised HSL proposal rows.
- Disable or ignore the legacy acceptance BCE and state-alpha BCE in this mode
  unless explicitly requested by config.  Otherwise pseudo-supervision and joint
  RL will fight.

### Diagnostics

The next resume test must print and log:

```text
joint adv pos/neg/near/ign: ...
joint rl adv/w: ...
joint rl alpha a/logp: ...
joint rl loss: ... (enabled=1, ...)
FrontRES grad debug ... ppo_alpha=...
```

Healthy first-run signs:

- `joint rl adv` is not always near zero;
- `joint rl weight` is nonzero on the same samples that execute Projected;
- PPO acceptance gradients and alpha-head gradients are both nonzero;
- `accept pref ...` may remain printed for legacy diagnostics, but should no
  longer be interpreted as the active HRL training signal when joint RL is
  enabled.

### Falsification

If joint advantage is mostly positive but rho/alpha remain conservative, the
issue is optimization or gradient boundary.  If joint advantage is mostly zero
or negative, the current HSL proposal and projected action space do not contain
a useful improvement over Noisy under the present utility.  In that case further
Preference-Learning target engineering is not the bottleneck.

## 2026-06-12 Oracle Upper-Bound Test Branch

### Motivation

Before changing alpha/rho training again, run a zero-intervention test branch
that answers one question:

```text
Does the current rollout evidence contain any candidate that can beat
Noisy/GMT under the executable utility?
```

This is a diagnostic branch, not a new training objective.  It must not feed
back into the loss.  Its job is to separate three failure modes:

1. no candidate in the current search/evaluation space can beat Noisy;
2. a candidate can beat Noisy, but the learned Projected action does not find it;
3. only the feasible-oracle proxy beats Noisy, which means the real executed
   search space still lacks the needed endpoint.

### Existing Evidence Sources

Reuse the current quartet and feasible-oracle diagnostics:

```text
U_noisy     = executable score of Noisy/GMT baseline
U_projected = executable score of executed FrontRES Projected action
U_candidate = executable score of full HSL repair candidate
U_feasible  = existing feasible-oracle proxy score
```

Then compute the diagnostic upper bound:

```text
U_ub  = max(U_noisy, U_projected, U_candidate, U_feasible)
A_ub  = U_ub - U_noisy
src   = argmax(noisy, projected, candidate, feasible)
```

This upper bound is intentionally optimistic because `U_feasible` is a proxy,
not an executed stable-frame rollout.  That is acceptable for a first-pass
falsification test: if even this optimistic bound cannot beat Noisy, further
rho/alpha target engineering is unlikely to help.

### Diagnostics To Print

Add console and TensorBoard diagnostics:

```text
oracle ub gain/src: A_ub / projected / candidate / feasible / noisy
oracle ub pass: fraction(A_ub > margin)
```

Interpretation:

- `oracle ub gain <= 0` or `noisy` wins most samples:
  the current HSL proposal/evaluator does not expose usable improvement.
- `candidate` or `feasible` wins often but `projected` does not:
  the improvement exists, but alpha/rho policy or credit assignment is failing.
- `feasible` dominates while `candidate` and `projected` are weak:
  the real action space is missing the executable endpoint; the feasible proxy
  is not enough as an executed route.
- `projected` wins often:
  the current policy is already finding useful repairs; continue training or
  evaluate fixed-DR test curves.

### Execution Constraint

This branch is a diagnostic test only.  Do not use `U_ub` as a reward, target,
mask, or actor update unless a later note explicitly changes the method.  If a
future implementation cannot preserve this boundary, stop and discuss instead
of silently merging diagnostics into training.

## 2026-06-12 Split Advantage For Simplex Alpha-Rho

### Problem

The current tri-anchor route has the right action geometry but the wrong credit
assignment.  The executed frame is conceptually:

```text
P = rho * R + (1 - rho) * ((1 - alpha) * N + alpha * S)
```

where `N` is the noisy input frame, `S` is the stable fallback endpoint, and
`R` is the HSL repair candidate.

This means:

- `alpha` chooses the fallback endpoint: noisy fallback vs stable fallback.
- `rho` decides how much repair mass to move from that chosen fallback toward
  the repair candidate.

Therefore `alpha` and `rho` must not be trained by the same scalar advantage.
Doing so reintroduces the old mixed-credit problem: a bad projected result may
mean the fallback choice is wrong, the repair amount is wrong, or the repair
candidate is wrong.  A single advantage cannot tell these apart.

### Training Signal Contract

Use two separate policy-gradient signals.

1. `alpha` signal: fallback selection.

```text
U_N = executable score of the noisy branch
U_S = executable score of the stable fallback proxy

if sampled alpha chose S:
    A_alpha = U_S - U_N
else:
    A_alpha = U_N - U_S
```

This is a clean binary routing signal.  It only answers whether the sampled
fallback endpoint was better than the alternative fallback endpoint.

2. `rho` signal: executed repair improvement over the chosen fallback.

```text
U_F = U_S if sampled alpha chose S else U_N
U_P = executable score of the actually executed projected frame

A_rho = U_P - U_F
```

This is a clean repair-strength signal.  It only answers whether the sampled
repair mass improved the frame beyond its chosen fallback.

### Implementation Contract

Do not add rollout branches for this change.  The first implementation uses the
existing feasible/stable proxy as `U_S`, the noisy branch as `U_N`, and the
projected branch as `U_P`.

Store the structured RL carrier as:

```text
acceptance_target[:, 0] = A_rho
acceptance_target[:, 1] = old alpha log-prob
acceptance_target[:, 2] = sampled alpha action
acceptance_target[:, 3] = A_alpha
acceptance_mask[:, 0] = rho update weight
acceptance_mask[:, 1] = alpha update weight
```

The algorithm must compute two PPO-style losses:

```text
rho loss   uses rho log-prob ratio   and A_rho
alpha loss uses alpha log-prob ratio and A_alpha
```

Do not combine alpha and rho log-prob ratios into one joint ratio.  That would
again make the two decisions share one credit-assignment channel.

### Diagnostics

The log should expose both branches:

```text
joint split adv rho/alpha
joint split w rho/alpha
```

The previous `joint rl adv/w` may remain as a backward-compatible rho summary,
but interpretation must change: it no longer means a joint alpha-rho advantage.
