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
