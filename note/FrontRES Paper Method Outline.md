# FrontRES Technical Method Record

This document is a paper-writing source record for FrontRES.  It expands the
current design contract into a technical description with formulas, losses,
sample gates, rollout scores, and training signals.  It is intentionally more
complete than a main-paper Methods section.  Use it as a pool of material:
select the clean core for the paper body and move detailed rewards, gates, and
diagnostics to the appendix.

Audience tags:

- **Main**: likely belongs in the main Methods section.
- **Appendix**: useful for reward, loss, implementation, and reproducibility
  details.
- **Internal**: mainly for our own debugging and design tracking.

## 0. Core Thesis

**Main.**  FrontRES is a front-end residual module placed before a frozen
whole-body motion tracker.  Its central observation is that geometric
restoration and dynamic executability are different decisions.  A clean-oriented
correction tells where a corrupted reference should move, but the current robot
state determines how much of that correction can be safely accepted.

The method separates reference repair into two coupled roles:

\[
\text{proposal: where to move},\qquad
\text{acceptance: how much to apply}.
\]

This separation lets supervised learning anchor a clean-oriented repair
direction, while rollout feedback trains a dynamics-aware acceptance field
without rewriting the proposal direction.

## 1. Problem Setup

**Main.**  Let \( \pi_{\mathrm{GMT}} \) be a frozen whole-body motion tracker.
At time \(t\), the tracker receives a root reference
\(g_t^{\mathrm{noisy}}\in SE(3)\) and robot observation \(o_t\).  The reference
may contain root-frame artifacts from video extraction, retargeting, or
reference-frame noise.  These artifacts can make the reference dynamically
inconsistent with the current robot state even when the intended motion remains
close to a valid clean trajectory.

FrontRES writes a corrected reference before the frozen tracker:

\[
g_t^{\mathrm{write}}
=
\mathcal{F}_{\theta}(o_t,g_t^{\mathrm{noisy}}),
\qquad
a_t^{\mathrm{robot}}
\sim
\pi_{\mathrm{GMT}}(\cdot\mid o_t,g_t^{\mathrm{write}}).
\]

The goal is not to retrain or replace \( \pi_{\mathrm{GMT}} \).  The goal is to
make corrupted references more executable while preserving clean-rollout
behavior.

## 2. FrontRES Parameterization

**Main.**  FrontRES predicts a task-space root residual rather than a full-body
joint correction:

\[
\Delta g_t
=
(\Delta x_t,\Delta y_t,\Delta z_t,
\Delta r_t,\Delta p_t,\Delta \psi_t).
\]

The active method uses one network with a shared representation and two
semantic heads:

\[
z_t=\phi_{\theta}(o_t),
\qquad
\Delta g_t=h_{\mathrm{prop}}(z_t),
\qquad
\rho_t=h_{\mathrm{accept}}(z_t).
\]

Here \(h_{\mathrm{prop}}\) predicts the clean-oriented proposal and
\(h_{\mathrm{accept}}\) predicts a six-dimensional acceptance vector
\(\rho_t\in[0,1]^6\).  The written residual is

\[
\Delta g_t^{\mathrm{write}}
=
\rho_t\odot \Delta g_t,
\]

and the reference passed to the frozen tracker is

\[
g_t^{\mathrm{write}}
=
g_t^{\mathrm{noisy}}
+
\Delta g_t^{\mathrm{write}}.
\]

The acceptance vector is not a confidence score.  It parameterizes a dynamic
projection of the HSL proposal:

\[
g_t^{\mathrm{write}}
\approx
\Pi_{\mathrm{dyn}}
(g_t^{\mathrm{noisy}}+\Delta g_t\mid o_t).
\]

## 3. Action Cone

**Main / Appendix.**  FrontRES only claims to repair artifacts inside a
task-space action cone:

\[
\Delta g_t \in \mathcal{C}_{\mathrm{act}}.
\]

The cone defines active correction dimensions and per-axis bounds:

\[
\Delta p_t \in [-d_{\max},d_{\max}]^3,
\qquad
\Delta \omega_t \in [-r_{\max},r_{\max}]^3.
\]

In the current root-reference repair setting, upward \(z\)-correction is
constrained because artificial root lifting can introduce dynamics
discontinuities.  The projected \(z\)-target uses a jump/penetration gate:

\[
\Delta z_t
\in
[-d_{\max},\ z_t^{+}],
\qquad
z_t^{+}
=
\operatorname{clip}
(j_t\,p_t^{\mathrm{pen}},0,d_{\max}),
\]

where \(j_t\in[0,1]\) is the jump degree and \(p_t^{\mathrm{pen}}\) is the
estimated anchor penetration depth.  When the motion is not in a jump/penetration
condition, \(z_t^{+}\approx 0\), so ordinary upward lifting is suppressed.

For inactive action dimensions, the corresponding residual dimensions are set
to zero:

\[
\Delta g_{t,k}=0,\qquad k\notin\mathcal{A}.
\]

This cone should be described as part of the method: it defines the repairable
reference-artifact family and prevents FrontRES from becoming a free tracker.

## 4. HSL Proposal Supervision

**Main.**  HSL trains the proposal head to predict a clean-oriented task-space
repair direction.  Let \(g_t^{\mathrm{clean}}\) be the clean reference and
\(g_t^{\mathrm{noisy}}\) the corrupted reference.  A raw clean-oriented target
can be written abstractly as

\[
\Delta g_t^{\star}
=
\operatorname{Log}
\left(
(g_t^{\mathrm{noisy}})^{-1}
g_t^{\mathrm{clean}}
\right),
\]

then projected into the action cone:

\[
\bar{\Delta g}_t^{\star}
=
\Pi_{\mathcal{C}_{\mathrm{act}}}
(\Delta g_t^{\star}).
\]

The proposal loss uses weighted Huber losses over position and orientation:

\[
\mathcal{L}_{\mathrm{pos}}
=
\frac{\sum_t w_t\,\alpha_t^{p}\,
\operatorname{Huber}
(\Delta p_t,\bar{\Delta p}_t^{\star})}
{\sum_t w_t\,\alpha_t^{p}+\epsilon},
\]

\[
\mathcal{L}_{\mathrm{rot}}
=
\frac{\sum_t w_t\,\alpha_t^{r}\,
\operatorname{Huber}
(\Delta \omega_t,\bar{\Delta \omega}_t^{\star})}
{\sum_t w_t\,\alpha_t^{r}+\epsilon}.
\]

The basic supervised proposal loss is

\[
\mathcal{L}_{\mathrm{sup}}
=
\mathcal{L}_{\mathrm{pos}}
+
\lambda_{\mathrm{rpy}}\mathcal{L}_{\mathrm{rot}}.
\]

The valid-dimension weights \(\alpha_t^{p},\alpha_t^{r}\) increase the weight of
samples whose target has nontrivial position or orientation signal.

**Appendix.**  Additional proposal regularizers are available:

\[
\mathcal{L}_{\mathrm{mag}}
=
\operatorname{Huber}
(\|\Delta g_t\|,\|\bar{\Delta g}_t^{\star}\|),
\]

\[
\mathcal{L}_{\mathrm{over}}
=
\left[
\|\Delta g_t\|-\|\bar{\Delta g}_t^{\star}\|
\right]_{+}^{2},
\]

\[
\mathcal{L}_{\mathrm{dir}}
=
1-\cos(\Delta g_t,\bar{\Delta g}_t^{\star}),
\]

and temporal smoothness:

\[
\mathcal{L}_{\mathrm{smooth}}
=
\operatorname{Huber}
\left(
(\Delta g_{t+1}-\Delta g_t),
(\bar{\Delta g}_{t+1}^{\star}-\bar{\Delta g}_{t}^{\star})
\right).
\]

The complete supervised proposal objective is

\[
\mathcal{L}_{\mathrm{HSL}}
=
\mathcal{L}_{\mathrm{sup}}
+
\lambda_{\mathrm{mag}}\mathcal{L}_{\mathrm{mag}}
+
\lambda_{\mathrm{over}}\mathcal{L}_{\mathrm{over}}
+
\lambda_{\mathrm{dir}}\mathcal{L}_{\mathrm{dir}}
+
\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}
+
\lambda_{\mathrm{harm}}\mathcal{L}_{\mathrm{harm}}.
\]

## 5. Harmful-Proposal Suppression

**Appendix.**  Some clean-oriented proposal directions can be harmful under the
current dynamics.  FrontRES uses a harmful-proposal loss to suppress the
proposal magnitude on samples where the repaired branch is worse than the noisy
baseline.

Let \(h_t\ge 0\) be a harmful sample weight derived from rollout evidence.  In
the active `hsl_hybrid` contract, this penalty applies to the proposal, not the
acceptance head:

\[
\mathcal{L}_{\mathrm{harm}}
=
\frac{
\sum_t h_t \|\Delta g_t\|_2^2
}{
\sum_t h_t+\epsilon
}.
\]

This term protects the frozen tracker from unsafe proposal directions while
preserving the conceptual split: HSL owns direction, acceptance owns strength.

## 6. Executability Score

**Appendix.**  FrontRES does not use the full environment reward as the main
repair signal.  Instead, it uses a narrower executability score designed to
measure whether the current reference is trackable by the frozen GMT.

The score is a weighted sum of planar, vertical, and task-consistency
components:

\[
J(g_t\mid o_t)
=
w_{\mathrm{pl}}J_{\mathrm{pl}}
+
w_{\mathrm{ver}}J_{\mathrm{ver}}
+
w_{\mathrm{task}}J_{\mathrm{task}}.
\]

The planar component averages an \(xy\) score and a yaw score:

\[
J_{\mathrm{pl}}
=
\frac{1}{2}
\left(
J_{xy}+J_{\psi}
\right).
\]

The \(xy\) score combines anchor position, anchor velocity, and foot-phase
terms:

\[
J_{xy}
=
\frac{
w_{xy}S_{xy}
+
w_{\dot{x}y}S_{\dot{x}y}
+
w_{\mathrm{foot}}S_{\mathrm{foot}}
}{
w_{xy}+w_{\dot{x}y}+w_{\mathrm{foot}}+\epsilon
}.
\]

The yaw score combines yaw angle and yaw-rate tracking:

\[
J_{\psi}
=
\frac{
w_{\psi}S_{\psi}
+
w_{\dot{\psi}}S_{\dot{\psi}}
}{
w_{\psi}+w_{\dot{\psi}}+\epsilon
}.
\]

The basic bounded tracking scores are

\[
S_{xy}
=
\operatorname{clip}
\left(
1-\frac{\|p_{xy}^{\mathrm{ref}}-p_{xy}^{\mathrm{robot}}\|}
{\tau_{xy}},
-1,1
\right),
\]

\[
S_{\psi}
=
\operatorname{clip}
\left(
1-\frac{|\operatorname{wrap}(\psi^{\mathrm{ref}}-\psi^{\mathrm{robot}})|}
{\tau_{\psi}},
-1,1
\right),
\]

and velocity terms use exponential kernels:

\[
S_{\dot{x}y}
=
\exp
\left(
-\frac{\|\dot{p}_{xy}^{\mathrm{ref}}-\dot{p}_{xy}^{\mathrm{robot}}\|^2}
{\sigma_{xy}^{2}}
\right),
\]

\[
S_{\dot{\psi}}
=
\exp
\left(
-\frac{(\dot{\psi}^{\mathrm{ref}}-\dot{\psi}^{\mathrm{robot}})^2}
{\sigma_{\psi}^{2}}
\right).
\]

The foot-phase score gates foot \(xy\) error by foot height:

\[
\eta_i
=
\sigma
\left(
\frac{\tau_z-|z_i^{\mathrm{ref}}-z_i^{\mathrm{robot}}|}
{T_{\mathrm{foot}}}
\right),
\]

\[
S_{\mathrm{foot}}
=
\frac{
\sum_i \eta_i
\operatorname{clip}
\left(
1-\frac{\|p_{i,xy}^{\mathrm{ref}}-p_{i,xy}^{\mathrm{robot}}\|}
{\tau_{\mathrm{foot}}},-1,1
\right)
}{
\sum_i \eta_i+\epsilon
}.
\]

The vertical component combines root \(z\), roll/pitch, and end-effector height:

\[
J_{\mathrm{ver}}
=
\frac{
w_zS_z+w_{rp}S_{rp}+w_{ee}S_{ee}
}{
w_z+w_{rp}+w_{ee}+\epsilon
}.
\]

where

\[
S_z=
\operatorname{clip}
\left(
1-\frac{|z^{\mathrm{ref}}-z^{\mathrm{robot}}|}{\tau_z},
-1,1
\right),
\]

\[
S_{rp}
=
\operatorname{clip}
\left(
1-\frac{\|
\operatorname{Log}
((R^{\mathrm{robot}})^{-1}R^{\mathrm{ref}})_{rp}
\|}{\tau_{rp}},
-1,1
\right),
\]

and \(S_{ee}\) is the maximum end-effector height consistency score.  The
task-consistency component uses body and anchor velocity kernels:

\[
J_{\mathrm{task}}
=
\frac{1}{3}
\left(
S_{\mathrm{body\ lin}}
+
S_{\mathrm{body\ ang}}
+
S_{\mathrm{anchor\ lin}}
\right).
\]

**Main option.**  The main paper can summarize this as an executable tracking
score over anchor pose, anchor velocity, foot phase, root orientation, and
end-effector height, leaving equations to the appendix.

## 7. Mode-Specific Executability

**Appendix.**  The executable score can be selected according to the active
repair family.  Let \(c_k(g_t,o_t)\) denote executable components:

\[
c_t=
(J_{xy},J_{\psi},J_z,J_{rp},J_{\mathrm{task}}).
\]

For a perturbation mode set \(M_t\), FrontRES uses a cone-aligned score

\[
J_M(g_t\mid o_t)
=
\frac{
\sum_{k\in M_t} \beta_k c_{t,k}
+
\beta_{\mathrm{task}}J_{\mathrm{task}}
}{
\sum_{k\in M_t} \beta_k+\beta_{\mathrm{task}}+\epsilon
}.
\]

This avoids asking one scalar reward component to judge correction dimensions
that are inactive or irrelevant for the current perturbation family.

## 8. Double-Sigmoid Sample Selection

**Main / Appendix.**  Rollout evidence defines a continuous repairability window.
Let

\[
d_t
=
\left[J(g_t^{\mathrm{clean}}\mid o_t)
-
J(g_t^{0}\mid o_t)\right]_{+}
\]

be the damage gap between the clean branch and the no-write/noisy branch.  Safe
samples have too little damage to require correction; broken samples are too far
outside the action cone; repairable samples lie between them.

FrontRES uses a double-sigmoid gate:

\[
\mu_t^{\mathrm{enter}}
=
\sigma
\left(
\frac{d_t-d_{\mathrm{safe}}}{T_g}
\right),
\qquad
\mu_t^{\mathrm{exit}}
=
\sigma
\left(
\frac{d_{\mathrm{broken}}-d_t}{T_g}
\right),
\]

\[
\mu_t
=
\operatorname{clip}
\left(
\frac{
\mu_t^{\mathrm{enter}}\mu_t^{\mathrm{exit}}
}{
\mu_{\mathrm{peak}}
},
0,1
\right).
\]

The side gates are

\[
\mu_t^{\mathrm{safe}}
=
1-\mu_t^{\mathrm{enter}},
\qquad
\mu_t^{\mathrm{repair}}
=
\mu_t,
\qquad
\mu_t^{\mathrm{broken}}
=
1-\mu_t^{\mathrm{exit}}.
\]

These gates implement a three-regime objective:

\[
\text{safe: abstain},\qquad
\text{repairable: repair},\qquad
\text{broken: abstain or conservative repair}.
\]

## 9. Quartet Rollout Branches

**Main.**  FrontRES evaluates four rollout/reference branches:

\[
g_t^0
=
g_t^{\mathrm{noisy}},
\qquad
g_t^1
=
g_t^{\mathrm{noisy}}+\Delta g_t,
\qquad
g_t^\rho
=
g_t^{\mathrm{noisy}}+\rho_t\odot\Delta g_t,
\]

plus the clean reference \(g_t^{\mathrm{clean}}\).  The corresponding executable
scores are

\[
J_0=J(g_t^0\mid o_t),
\qquad
J_1=J(g_t^1\mid o_t),
\qquad
J_{\rho}=J(g_t^\rho\mid o_t),
\qquad
J_c=J(g_t^{\mathrm{clean}}\mid o_t).
\]

The main rollout gains are

\[
A_{\mathrm{full}}=J_1-J_0,
\qquad
A_{\rho}=J_{\rho}-J_0,
\qquad
A_{\mathrm{proj}}=J_{\rho}-J_1.
\]

Candidate \(g_t^1\) is a counterfactual full-write branch.  Projected
\(g_t^\rho\) is the branch actually induced by the current acceptance head.

## 10. Acceptance Preference Learning

**Main.**  Counterfactual candidate rollouts are not policy-sampled actions.
Therefore, their ordering must be converted into an explicit acceptance
preference target.

With margin \(m\), define

\[
y_t =
\begin{cases}
1, &
J_1>J_{\rho}+m
\ \land\
J_1>J_0+m,\\
0, &
J_0>J_{\rho}+m
\ \land\
J_0>J_1+m,\\
\varnothing, & \text{otherwise}.
\end{cases}
\]

The preference mask is gated by repairability and oracle trust:

\[
q_t
=
\tau_t^{\mathrm{oracle}}\mu_t^{\mathrm{repair}}
\mathbb{1}[y_t\ne\varnothing],
\]

where \(\tau_t^{\mathrm{oracle}}\in[0,1]\) suppresses samples where the feasible
oracle is not trusted.  Per-mode and active-dimension masks further restrict the
loss to dimensions that can be repaired:

\[
M_{t,k}
=
q_t
\mathbb{1}[k\in\mathcal{A}_t].
\]

The active implementation uses class-balanced focal BCE:

\[
\mathcal{L}_{\mathrm{pref}}
=
\frac{
\sum_{t,k}
M_{t,k}
w_{y_t}
(1-p_{t,k})^\gamma
\operatorname{BCE}(\rho_{t,k},y_t)
}{
\sum_{t,k}
M_{t,k}
w_{y_t}
(1-p_{t,k})^\gamma
+\epsilon
},
\]

where

\[
p_{t,k}
=
\begin{cases}
\rho_{t,k}, & y_t=1,\\
1-\rho_{t,k}, & y_t=0.
\end{cases}
\]

The minibatch class weights are

\[
w_{\mathrm{full}}
=
\operatorname{clip}
\left(
\frac{M}{2M_{\mathrm{full}}},
w_{\min},w_{\max}
\right),
\qquad
w_{\mathrm{noop}}
=
\operatorname{clip}
\left(
\frac{M}{2M_{\mathrm{noop}}},
w_{\min},w_{\max}
\right).
\]

This prevents the acceptance head from collapsing to no-write when valid
full-write preferences are underrepresented after masks.

## 11. Inertial Compatibility Test Plan

**Internal -> Appendix.**  The high-DR collapse suggests a failure mode that is
not captured by clean-reference closeness alone.  Once the robot is already
tilted or moving away from the clean tracking manifold, a clean-oriented repair
can become anti-inertial: it may point against the current body momentum and
consume the small stability margin that the corrupted reference accidentally
preserves.  In this regime, the useful restoration path may be curved rather
than a straight interpolation from Noisy to Clean.

For each branch
\[
b \in \{0,\rho,1,c\},
\]
corresponding to Noisy, Projected, Candidate, and Clean, define the
state-reference displacement
\[
e^b_p = p^b_{\mathrm{ref}} - p_{\mathrm{robot}},
\]
and the rotational displacement
\[
e^b_R =
\mathrm{Log}\left(
R_{\mathrm{robot}}^\top R^b_{\mathrm{ref}}
\right).
\]
The branch distance to the current robot state is
\[
D^b =
w_p \|e^b_p\|_2
+
w_R \|e^b_R\|_2.
\]
For local roll/pitch perturbations, the most important rotational term is
\[
e^b_{rp}=e^b_R[0:2].
\]

Let \(v_{\mathrm{robot}}\) and \(\omega_{\mathrm{robot}}\) be the current
anchor linear and angular velocities.  Define an inertial compatibility score:
\[
C^b =
\frac{\langle e^b_p, v_{\mathrm{robot}}\rangle}
{\|e^b_p\|_2\|v_{\mathrm{robot}}\|_2+\epsilon}
+
\lambda_{\omega}
\frac{\langle e^b_R, \omega_{\mathrm{robot}}\rangle}
{\|e^b_R\|_2\|\omega_{\mathrm{robot}}\|_2+\epsilon}.
\]
Positive \(C^b\) means the reference branch lies along the current inertial
trend.  Negative \(C^b\) means the branch asks the robot to reverse its current
motion, which can be dangerous near the stability boundary.  For the local-rp
case, also log
\[
C^b_{rp} =
\frac{\langle e^b_{rp}, \omega_{\mathrm{robot}}[0:2]\rangle}
{\|e^b_{rp}\|_2\|\omega_{\mathrm{robot}}[0:2]\|_2+\epsilon}.
\]

The key diagnostic is the state-reference inversion:
\[
\Delta D^\rho = D^\rho - D^0,
\qquad
\Delta C^\rho = C^\rho - C^0.
\]
The suspected failure pattern is:
\[
\Delta D^\rho > 0
\quad\mathrm{or}\quad
\Delta C^\rho < -m_I,
\]
especially when it appears before a sharp drop in episode length.  This means
Projected is geometrically closer to Clean but dynamically farther from the
current robot state than Noisy.

The test should be diagnostic-only first:

1. Freeze checkpoints around the observed transition, especially the
   \(699\sim701\) and \(750\sim755\) range.
2. Evaluate fixed DR bins, e.g. \(d\in\{2.0,2.2,2.36,2.48,2.6\}\), using the
   same motions and seeds.
3. For each branch, log \(J^b\), \(D^b\), \(C^b\), and \(C^b_{rp}\).
4. Report
   \[
   \mathrm{angle\_inv}
   =
   \mathbb{E}[\mathbb{1}(D^\rho>D^0)],
   \]
   \[
   \mathrm{anti\_inertia}
   =
   \mathbb{E}[\mathbb{1}(C^\rho<C^0-m_I)],
   \]
   and the same quantities for Candidate and Clean.
5. Condition the same statistics on falls or short-horizon failures within the
   next \(K\) steps.  The hypothesis is supported only if inversion and
   anti-inertia rise before failure, not merely after failure.
6. Add counterfactual mixed branches:
   \[
   g^{p0,r1}=(p^{\mathrm{noisy}}, R^{\mathrm{candidate}}),
   \qquad
   g^{p\rho,r1}=(p^{\rho}, R^{\mathrm{candidate}}),
   \]
   to test whether repairing orientation while delaying position is easier for
   GMT than full clean-oriented restoration.

Current diagnostic-only implementation lives in
`source/rsl_rl/rsl_rl/runners/on_policy_runner.py` inside
`_maybe_print_frontres_restore_debug`.  It prints `[FrontRES inertial debug]`
lines together with the existing restore/gate debug logs.  This implementation
does not change the rollout branch, reward, preference target, or PPO gradient;
it only exposes \(D^b\), \(C^b\), \(C^b_{rp}\), `angle_inv`, and
`anti_proj/cand/clean`.  It also logs mixed state metrics for
\(g^{p0,r1}\) and \(g^{p\rho,r1}\).  These mixed values are currently
state/inertia diagnostics, not additional counterfactual rollout scores
\(J^{p0,r1}\) or \(J^{p\rho,r1}\).

If this hypothesis is verified, the smallest repair is to inject an inertial
compatibility prior into acceptance rather than changing the HSL proposal
direction.  Define
\[
P^b_I =
\left[
C^0 - C^b + m_I
\right]_+.
\]
Then either shape the preference score
\[
\tilde{J}^b =
J^b - \lambda_I P^b_I,
\]
or apply a deployment-time acceptance suppressor
\[
\rho^{\mathrm{write}}
=
\rho
\odot
\sigma\left(
\frac{C^\rho-C^0-m_I}{T_I}
\right).
\]
The first option lets the acceptance head learn the rule from the existing
state observation.  The second option is a stronger hand-written prior and
should be treated as an ablation or emergency stabilizer.

An intermediate option is branch selection along the already evaluated
correction path:
\[
\rho^*
=
\arg\max_{\alpha\in\{0,\rho,1\}}
\left(
J^\alpha-\lambda_I P^\alpha_I
\right).
\]
This keeps the proposal fixed and changes only how much of the proposal is
accepted.  It is therefore consistent with the current two-head contract:
HSL owns direction, while acceptance owns current-state dynamic admissibility.

## 12. Candidate Ranking Reward

**Appendix.**  In addition to direct preference supervision, the rollout reward
can include a candidate-ranking term:

\[
R_{\mathrm{rank}}
=
A_{\rho}
+
\lambda_{\mathrm{proj}}A_{\mathrm{proj}}
-
\lambda_{\mathrm{under}}
\left[A_{\mathrm{full}}-A_{\rho}\right]_{+}
\mathbb{1}[A_{\mathrm{full}}>0]
-
\lambda_{\mathrm{harm}}
\left[-A_{\rho}\right]_{+}.
\]

This term penalizes under-writing a useful candidate, rewards projection when
projected is better than full-write, and penalizes harmful projected writes.
Preference supervision is the cleaner acceptance signal; ranking reward can be
viewed as auxiliary PPO shaping.

## 13. Selective FrontRES Reward

**Appendix.**  Let \(r_t^{\mathrm{exec}}\) be the executable signal.  In the
default gain mode,

\[
r_t^{\mathrm{exec}}
=
A_{\rho}
=
J_{\rho}-J_0.
\]

Alternative modes use normalized repair ratio:

\[
r_t^{\mathrm{ratio}}
=
\operatorname{clip}
\left(
\frac{A_{\rho}}{d_t+\epsilon},
-1,1
\right),
\]

or family-normalized preference:

\[
r_t^{\mathrm{pref}}
=
s
\left[
\alpha
\tanh
\left(
\frac{A_{\rho}}{\sigma_{\mathrm{family}}\tau}
\right)
+
(1-\alpha)r_t^{\mathrm{ratio}}
\right].
\]

The selective reward combines positive terms and constraint penalties:

\[
R_t^{+}
=
\lambda_{\mathrm{exec}}
\mu_t^{\mathrm{repair}}
r_t^{\mathrm{exec}}
+
\lambda_{\mathrm{gain}}
\mu_t^{\mathrm{repair}}
\left[A_{\rho}-a_{\min}\right]_{+}
+
\lambda_{\mathrm{geom}}R_t^{\mathrm{geom}}
+
\lambda_{\mathrm{rank}}R_t^{\mathrm{rank}}
+
\lambda_{\mathrm{rescue}}R_t^{\mathrm{rescue}}.
\]

The constraint penalty is

\[
C_t
=
\lambda_{\mathrm{harm}}C_t^{\mathrm{harm}}
+
\lambda_{\mathrm{int}}C_t^{\mathrm{int}}
+
C_t^{\mathrm{bound}}
+
C_t^{\mathrm{under}}.
\]

The final reward delta used by PPO is scheduled by curriculum progress:

\[
r_t^{\Delta}
=
\eta_t R_t^{+}
-
\kappa_t C_t,
\]

where \(\eta_t\) increases with DR and actor-takeover progress, and
\(\kappa_t\) is a stronger constraint-progress schedule.

The harmful executable penalty uses action-gated negative repair gain:

\[
C_t^{\mathrm{harm}}
=
\omega_t^{\mathrm{harm}}
\left[-A_{\rho}-\epsilon_h\right]_{+}
g_t^{\mathrm{act}},
\]

where \(g_t^{\mathrm{act}}\in[0,1]\) suppresses false harmful labels when the
policy essentially performed no correction.

## 14. Intervention, Boundary, and Under-Repair Costs

**Appendix.**  FrontRES includes several practical costs to keep the front-end
correction conservative.

The intervention cost penalizes correction magnitude:

\[
C_t^{\mathrm{int}}
=
\|\Delta g_t^{\mathrm{write}}\|^2
\quad
\text{or a weighted task-space variant}.
\]

The clean-boundary cost penalizes moving beyond the clean target along the HSL
line:

\[
C_t^{\mathrm{bound}}
=
\left[
\langle \Delta g_t^{\mathrm{write}}, \hat{\Delta g}_t^{\star}\rangle
-
\|\Delta g_t^{\star}\|
-
m_{\mathrm{over}}
\right]_{+}^{2}
+
C_t^{\mathrm{side}}.
\]

The under-repair penalty is used when a minimum restoration ratio is expected:

\[
C_t^{\mathrm{under}}
=
\lambda_{\mathrm{under}}
\mu_t^{\mathrm{repair}}
\left[
\rho_{\min}
-
\frac{e_t^{\mathrm{raw}}-e_t^{\mathrm{frontres}}}
{e_t^{\mathrm{raw}}+\epsilon}
\right]_{+}^{2}.
\]

These terms are better suited for the appendix unless an ablation shows that one
is essential to the method.

## 15. Hierarchical Reinforcement Learning Boundary

**Main / Appendix.**  The current system is hierarchical in authority, not in
the sense of a second policy that proposes new motions.  HSL/proposal learning
owns the repair direction:

\[
\Delta g_t=h_{\mathrm{prop}}(z_t).
\]

HRL/PPO owns only acceptance:

\[
\rho_t=h_{\mathrm{accept}}(z_t),
\qquad
\Delta g_t^{\mathrm{write}}
=
\rho_t\odot\Delta g_t.
\]

The PPO objective is standard clipped PPO over sampled actions, but its actor
gradient is restricted to the acceptance outputs:

\[
\mathcal{L}_{\mathrm{PPO}}
=
-
\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)\hat{A}_t,
\operatorname{clip}
(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t
\right)
\right].
\]

The gradient contract is

\[
\nabla_{\theta_{\mathrm{prop}}}
\left(
\mathcal{L}_{\mathrm{pref}}
+
\mathcal{L}_{\mathrm{PPO}}
\right)
=0,
\]

while

\[
\nabla_{\theta_{\mathrm{accept}}}
\left(
\mathcal{L}_{\mathrm{pref}}
+
\mathcal{L}_{\mathrm{PPO}}
\right)
\ne 0.
\]

Thus rollout feedback can learn dynamic strength without corrupting the
clean-oriented proposal direction.

## 16. Total Training Objective

**Main.**  The full optimization objective can be summarized as

\[
\mathcal{L}
=
\lambda_{\mathrm{HSL}}\mathcal{L}_{\mathrm{HSL}}
+
\lambda_{\mathrm{pref}}\mathcal{L}_{\mathrm{pref}}
+
\lambda_{\mathrm{PPO}}\mathcal{L}_{\mathrm{PPO}}
+
\lambda_V\mathcal{L}_V
-
\lambda_H\mathcal{H}.
\]

The value loss uses the modified FrontRES reward delta \(r_t^{\Delta}\):

\[
\mathcal{L}_V
=
\|V_{\theta}(o_t)-\hat{R}_t\|^2.
\]

The important conceptual constraint is that the proposal and acceptance heads
receive different learning signals:

\[
\mathcal{L}_{\mathrm{HSL}}
\rightarrow
\theta_{\mathrm{shared}},\theta_{\mathrm{prop}},
\qquad
\mathcal{L}_{\mathrm{pref}},\mathcal{L}_{\mathrm{PPO}}
\rightarrow
\theta_{\mathrm{accept}}.
\]

## 17. Training Schedule

**Main / Appendix.**  Training uses three phases:

1. **Critic/proposal warmup.**  Low, controlled perturbations train the HSL
   proposal and value function.  PPO actor updates are disabled or strongly
   downweighted.
2. **Actor takeover.**  The PPO/acceptance actor weight ramps up after the
   proposal direction becomes stable.
3. **Frontier fine-tuning.**  The perturbation curriculum samples easy,
   frontier, and hard regions.  Easy samples preserve clean restoration,
   frontier samples train acceptance, and hard samples expose failure
   boundaries without dominating the batch.

The curriculum target is not simply maximum perturbation strength.  It is to
keep enough samples near the repairable frontier while preserving clean
restoration behavior.

## 18. Perturbation Curriculum and Failed Cases

**Main / Appendix.**  The perturbation curriculum is a method component, not a
minor training detail.  Several failed training regimes motivated the current
design.

Failed cases:

1. **Fixed high `dr_scale`.**  Training for a long time at one high perturbation
   value, such as the observed local-rp frontier near
   \(\mathrm{dr\_scale}\approx 2.3\), can teach FrontRES to survive or reject
   difficult repairs while washing out the low-perturbation clean-restoration
   behavior learned during warmup.
2. **Monotonic difficulty ramp only.**  A simple increasing schedule moves the
   batch distribution away from easy and medium perturbations.  This improves
   frontier exposure but can degrade demo-quality restoration.
3. **Hard samples dominating actor updates.**  Once the robot leaves the
   recoverable neighborhood, root-level \(SE(3)\) correction cannot reliably
   restore clean behavior.  Letting these samples dominate encourages no-op,
   conservative acceptance, or noisy reward chasing.
4. **Composite perturbations too early.**  Mixing perturbation families before
   the single-family correction signals are clear creates conflicting reward
   and supervision signals, especially when the active action cone cannot repair
   all sampled families.

The resulting rule is to train over a distribution of perturbation difficulties
rather than a single scalar:

\[
p(d)
=
w_e p_{\mathrm{easy}}(d)
+
w_f p_{\mathrm{frontier}}(d)
+
w_h p_{\mathrm{hard}}(d).
\]

The regions have distinct roles:

\[
\mathcal{D}_{\mathrm{easy}}:
\text{ low perturbation for clean restoration},
\]

\[
\mathcal{D}_{\mathrm{frontier}}:
\text{ near the repairable boundary for acceptance learning},
\]

\[
\mathcal{D}_{\mathrm{hard}}:
\text{ slightly beyond the boundary for robustness exposure}.
\]

A practical mixture is

\[
w_e\approx 0.5,\qquad
w_f\approx 0.4,\qquad
w_h\approx 0.1,
\]

with easy and frontier samples dominating and hard samples kept as a tail.  The
frontier center should adapt to rollout evidence.  It should increase when
FrontRES remains comfortably better than GMT, and decrease when FrontRES becomes
harmful or episode length drops sharply below the GMT baseline:

\[
\mathrm{epLen}_{\mathrm{frontres}}
\approx
\mathrm{epLen}_{\mathrm{gmt}}.
\]

The current implementation also contains a practical `dr_scale` ramp used to
align supervised proposal learning with the target perturbation range:

\[
d_t
=
d_{\mathrm{start}}
+
\alpha_t
(d_{\mathrm{end}}-d_{\mathrm{start}}),
\qquad
\alpha_t
=
\operatorname{clip}
\left(
\frac{t}{T_{\mathrm{ramp}}},
0,1
\right).
\]

Current recorded values for the active local-rp setting are:

\[
d_{\mathrm{start}}=1.25,\qquad
d_{\mathrm{end}}=4.375,\qquad
T_{\mathrm{ramp}}=1400.
\]

The runtime bounds are

\[
d_{\min}=1.25,\qquad d_{\max}=4.50.
\]

For local roll/pitch perturbations, the MOSAIC base perturbation is
approximately \(0.08\ \mathrm{rad}\), so

\[
\mathrm{dr\_scale}=4.375
\quad\Rightarrow\quad
\epsilon\approx 0.35\ \mathrm{rad},
\]

which matches the RobotBridge-like perturbation level used for presentation and
validation.  These numbers are experiment-specific and should be reported in
the experimental setup or appendix rather than framed as universal constants.

**Appendix.**  Perturbation-family curriculum should also be staged:

1. Early: balanced single-family perturbations.
2. Middle: occasional pairs once single-family signals are stable.
3. Late: rare three/full combinations for robustness exposure.

The sampled perturbation families must respect the active action cone.  Sampling
a family that the current action dimensions cannot repair creates false
negative reward and should be avoided.

## 19. Main-Paper Minimal Version

**Main.**  If space is limited, the method can be compressed to:

1. FrontRES is a front-end residual module before a frozen tracker.
2. It predicts a task-space proposal \(\Delta g_t\) and dynamic acceptance
   \(\rho_t\).
3. The written reference is
   \(g_t^{\mathrm{write}}=g_t^{\mathrm{noisy}}+\rho_t\odot\Delta g_t\).
4. HSL supervision trains the proposal direction.
5. Quartet rollout preference compares no-write, full-write, and projected
   write to train acceptance.
6. PPO/preference gradients are restricted to acceptance, so rollout feedback
   cannot rewrite the proposal direction.

## 20. Appendix Candidates

**Appendix.**  The following details are good appendix material:

- full executability score components;
- mode-specific executable score selection;
- double-sigmoid sample selection;
- jump/penetration \(z\)-gate;
- class-balanced focal preference loss;
- candidate-ranking reward;
- inertial compatibility diagnostics and the curved-path failure hypothesis;
- selective reward composition;
- intervention, boundary, under-repair, and harmful costs;
- gradient-boundary implementation;
- curriculum and checkpoint-selection details;
- failed cases behind the `dr_scale` curriculum;
- diagnostic metrics and what failure each metric detects.

## 21. Notation Table

| Symbol | Meaning | Suggested location |
| --- | --- | --- |
| \(o_t\) | current robot/tracker observation | Main |
| \(g_t^{\mathrm{clean}}\) | clean root reference | Main |
| \(g_t^{\mathrm{noisy}}\) | corrupted root reference | Main |
| \(\Delta g_t\) | FrontRES proposal residual | Main |
| \(\rho_t\) | dynamics-aware acceptance vector | Main |
| \(g_t^{0}\) | no-write branch | Main |
| \(g_t^{1}\) | full candidate branch | Main |
| \(g_t^{\rho}\) | projected/write branch | Main |
| \(J(\cdot\mid o_t)\) | executable score | Appendix |
| \(D^b\) | distance between branch \(b\) and the current robot state | Appendix |
| \(C^b\) | inertial compatibility between branch \(b\) and current anchor velocity | Appendix |
| \(P_I^b\) | inertial incompatibility penalty for branch \(b\) | Appendix |
| \(d_t\) | clean-vs-noisy damage gap | Appendix |
| \(\mu_t\) | double-sigmoid repairability gate | Appendix |
| \(d_t\) | perturbation scale / `dr_scale` when used in curriculum context | Appendix |
| \(y_t\) | acceptance preference target | Main |
| \(M_{t,k}\) | preference mask for dimension \(k\) | Appendix |
| \(\mathcal{C}_{\mathrm{act}}\) | task-space action cone | Main |

## 22. Writing Guidance

Do not present every reward term as a separate contribution.  The contribution
is the architectural decomposition:

\[
\text{corrupted reference}
\rightarrow
\text{task-space proposal}
\rightarrow
\text{dynamic acceptance}
\rightarrow
\text{frozen tracker rollout}.
\]

The reward terms, gates, and diagnostics support this decomposition.  They
should be used in the main paper only when they clarify why the decomposition is
trainable; otherwise they belong in the appendix.
