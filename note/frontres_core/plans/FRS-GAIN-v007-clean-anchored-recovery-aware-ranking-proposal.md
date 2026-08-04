# FRS-GAIN-v007 Proposal: Clean-Anchored Recovery-Aware Ranking

Status: closed after human confirmation on 2026-08-01. Retained as design
rationale; active semantics live in `FRS-METHOD-v017`, `FRS-GAIN-v007`,
`FRS-PPO-v005`, and `FRS-TRAIN-v012`. Created: 2026-07-30. Updated: 2026-08-01.

Governance note, 2026-08-03: the perturbation-frontier and K-transition text
below records the currently active TRAIN-v012 reading, but E-FI-107 has reopened
that narrow interaction through the TRAIN-v013 proposal. It must not authorize
new `g_K` implementation or Phase A continuation before Design Inspector review
and contract activation.

Affected design points:

- `FRS-DP-01` / `M-02`: Perturbation Data
- `FRS-DP-02` / `SR-01`: Segment Replay
- `FRS-DP-04` / `M-04`: FrontRES 6D Repair
- `FRS-DP-06` / `Q-PAIR`: Paired Rollouts
- `FRS-DP-07` / `Q-01`: Repair Gain
- `FRS-DP-09` / `M-05`: Actor & Critic Warmup

The coordinated active contracts are `FRS-METHOD-v017`, `FRS-GAIN-v007`,
`FRS-PPO-v005`, and `FRS-TRAIN-v012`. This rationale note does not supersede or
activate any contract.

## Confirmed Concept

Gain is divided into Intent Gain and Physics Gain. Recovery-Aware is the
continuous interaction between those two terms inside `G_total`; it is not an
explicit Physics/Intent ratio or a hard handoff between two objectives. Segment
Replay does not define that interaction. It collects the M comparable
`G_total` values under one sealed scenario. Every valid attempt remains a PPO
row; their different returns and advantages provide the ordering signal without
a winner-only or argmax update.

```text
Clean Rollout
  -> indicates the desired motion semantics and recovery direction

Noisy zero-action Rollout
  -> defines the no-repair zero point

Clean-anchored Gain
  -> combines Intent improvement and pressure-weighted Physics improvement
     into comparable recovery evidence

Segment Replay
  -> lets all M Repair outcomes contribute ordered credit for the current
     one-action reachable frontier
```

The method should learn this rule from rollout consequences. It must not encode
the rule as a manually scheduled proportion, a state switch, a new actor output,
or a second network.

## Preserved Perturbation Strength Curriculum

Perturbation strength uses the existing frontier-envelope curriculum rather
than a monotonic ramp. Because the same corruption can have a different
executable consequence under `K=8`, `K=16`, and `K=32`, each active horizon
owns its own frontier `g_K`, identified from the frozen GMT Noisy baseline at
that horizon. The curriculum keeps easier restoration cases present,
concentrates evidence near `g_K`, and retains only a small stress tail beyond
it:

\[
\begin{aligned}
p(d\mid K)={}&0.20\,\mathcal U(0,0.25g_K)
+0.30\,\mathcal U(0.25g_K,0.70g_K)\\
&+0.40\,\mathcal U(0.70g_K,g_K)
+0.10\,\mathcal U(g_K,\min(1.10g_K,d_{\max})).
\end{aligned}
\]

The first two regions jointly form the historical `easy` mass; the third is
the repair frontier and the fourth is the capped hard tail. Training progress
does not force every sample toward maximum corruption. `g_8`, `g_16`, and
`g_32` are calibrated independently; the method does not impose a monotonic
ordering among them. Repair Gain and PPO outcomes do not become a second
adaptive sampling controller.

At every K transition, Actor and action std remain frozen while the frozen-GMT
Noisy baseline calibrates the new `g_K`. That K-conditioned strength
distribution is then frozen before Critic-only recalibration begins. The
training order is therefore:

```text
new K
-> freeze Actor/std
-> calibrate and freeze g_K from frozen-GMT Noisy executability
-> Critic-only recalibration
-> Actor-ramp
-> Joint Optimize
```

The active K and its frontier identity/state belong to the persisted curriculum
identity. A single global frontier must not silently carry across K stages.

Each sealed Segment samples one strength and one single-family artifact at
scenario selection. The zero-action Noisy rollout and all M Repair rollouts
reuse that exact strength, artifact, protocol, application point, and hash;
Clean remains uncorrupted. The active family remains `local_rp`, and this
strength curriculum does not authorize composite perturbations or narrow the
full-6D repair action.

## Clean Rollout Authority

Clean Rollout is restored as evaluator evidence. Its role is to identify what
the original motion normally requires at the current phase, including expected
support changes, planned dynamic lean, phase-ZMP behavior, and the intended
pose. It is an anchor for interpreting Physics and demo quality, not a condition
that Repair must numerically equal.

Clean Rollout remains forbidden from:

- FEMR actor observations or future context;
- HSL targets;
- PPO policy rows;
- deployment inputs;
- changing `x_t` from a dynamics-only replay reset.

The Noisy zero-action rollout remains the do-nothing counterfactual. Each
Repair rollout remains the consequence of one policy-sampled full-6D
`Delta SE(3)` action. Clean, Noisy, and Repair therefore have distinct
evaluator roles and must not be collapsed into interchangeable baselines.

## Preserved Deployment No-Feedback Boundary

One-action-K training and per-frame deployment have different clocks but the
same reference authority. At deployment frame `t`, FrontRES reads the current
frame and future q29 Intent from the sealed Noisy/deployment stream and applies
its full-6D `Delta SE(3)_t` only to that current reference frame. The repaired
reference is not written back as the next frame's actor input or as a replacement
for the next frame in the sealed stream.

The robot's physical state continues from one environment step to the next, so
the effect of earlier repairs remains visible through dynamics. What is forbidden
is recursive reference feedback, where `Repair_t` becomes the reference base for
`Repair_{t+1}` and lets local residuals accumulate into an unintended trajectory.
This preserves the active evaluation contract's no-feedback composition boundary
without changing one-action-K training or introducing Clean deployment input.

## Confirmed Paired-Rollout Lifecycle And Noise Boundary

Each sealed Segment scenario uses exactly this evaluator set:

```text
one Clean Rollout
one fixed zero-action Noisy Rollout
M Repair Rollouts
```

Clean and Noisy are each executed once, then their observed K-step evidence is
sealed and read-only for all M Repair comparisons. The Noisy artifact,
corruption protocol, application point, `noisy_segment_hash`, `x_t`, q29
Intent, Clean continuation, K, and frozen `pi_old` remain fixed. Resetting or
starting another Repair attempt must not regenerate, mutate, or mix either
shared baseline.

This lifecycle controls the avoidable source of comparison noise: every Repair
attempt is judged against the same observed Clean anchor and the same observed
Noisy zero point. It does not claim that simulator dynamics become
deterministic or that all rollout noise cancels. The shared Clean/Noisy outcome
is a common offset inside one Segment, so it cannot create a different ranking
between that Segment's M attempts. Residual Repair dynamics noise and
cross-Segment target variance remain ordinary rollout noise and are accepted
unless runtime evidence later shows that they erase the learning signal.

## Recovery-Aware Ordering Principle

Segment Replay supplies multiple Repair candidates for the same sealed
scenario. A single full-6D repair has a dynamics-limited reachable set, so a
Repair is not expected to equal Clean in one action. The three references have
different roles:

- Clean indicates which recovery direction preserves the original motion;
- Noisy defines zero improvement under no repair;
- the M Repair outcomes expose the current policy's empirical one-action
  reachable frontier.

Gain gives every attempt one Recovery-Aware score by combining Intent Gain and
Physics Gain. Segment Replay preserves every valid attempt with equal structural
mass, so PPO receives both better and worse outcomes from the same sealed
scenario. Their score differences move learning toward the best currently
reachable direction without turning Clean equality into a positive-Gain
threshold or discarding non-winning attempts.

Within one scenario, every attempt shares the same current observation and
Critic baseline. Its scalar credit is therefore

\[
A^{(m)}=G_{\mathrm{total}}^{(m)}-V(o_t).
\]

Subtracting the same baseline preserves the `G_total` ordering across M
attempts. Grouped PPO may apply the existing sign-preserving scale, but it must
not mean-center the group, select only the maximum, or multiply row mass by the
score.

The resulting ordering must let learning distinguish at least these outcomes:

1. A Repair that still depends on unplanned support changes or compensating
   steps has not completed Physics recovery, even if it survives.
2. A Repair that no longer depends on extra support compensation but remains
   persistently tilted has a remaining Intent/demo-quality error.
3. When Repair candidates have comparable physical recovery, the candidate
   closer to the Clean motion's intended pose should be preferred.

These statements define the required candidate ordering. The Physics/Intent
interaction must already be present in each attempt's Gain. Segment Replay
supplies only the same-scenario ordering operation; it must not add another
Physics/Intent rule.

## Confirmed Pressure-Weighted Ranking Form

Each retained Intent or Physics evidence term `j` first defines a
Clean-conditioned remaining problem:

\[
r_j(X\mid C)=\frac{D_j(X,C)}{S_j}.
\]

`D_j(X,C)` is the channel-specific remaining problem relative to the Clean
motion semantics. It is not a single generic Euclidean distance: the relevant
meaning may be intended pose, expected support behavior, phase-conditioned ZMP,
survival, or another retained channel-specific quantity.

`S_j` is the fixed semantic unit of that evidence term. The normalization has
four distinct responsibilities:

1. quantities measured in metres, radians, seconds, or contact events become
   dimensionless semantic units instead of competing through their raw units;
2. the meaning remains fixed across Segments, attempts, and curriculum stages,
   because `S_j` is not recomputed from the current M candidates;
3. it preserves absolute recovery pressure, because `S_j` is not the current
   `Noisy-Clean` gap used by a recovery ratio; a near-zero gap would amplify
   rollout noise, while equal recovery ratios would erase the difference
   between severe and mild states;
4. it does not clip the result to `[0,1]`, so severe remaining problems do not
   saturate into the same value.

The normalization therefore does more than align units. It creates a stable
cross-Segment comparison coordinate while preserving severity. It does not by
itself decide the Intent/Physics preference; that interaction is introduced by
the recovery-pressure coefficient below.

## Confirmed K-Step Aggregation

Every continuous retained channel uses a direction-sensitive K-step cumulative
consequence normalized by its weighted semantic exposure:

\[
D_j^{\rightarrow}(X,C)
=
\frac{\sum_{k=1}^{K}a_{j,k}\,\tau_k\,e_{j,k}(X,C)}
     {\sum_{k=1}^{K}a_{j,k}\,\tau_k},
\qquad \tau_k=\frac{k}{K}.
\]

`e_{j,k}` is the frame-level Clean-conditioned problem for channel `j`, and
`a_{j,k}` states whether that channel is meaningful at step `k`. The fixed
normalized position `tau_k` gives later residuals more influence without
discarding earlier consequences. It therefore distinguishes a trajectory whose
problem is shrinking from one whose problem is growing. For example, residuals
`[4,3,2,1]` and `[1,2,3,4]` have the same ordinary mean `2.5`, but their
direction-sensitive values are `2.0` and `3.0`, respectively. `tau_k` is not a
learned parameter, temperature, or curriculum schedule.

This is a weighted-effective-time cumulative quantity, not a raw sum and not
an endpoint-only measurement. It preserves the complete K-step consequence
while preventing the same behavior from receiving a larger numerical score
merely because the curriculum changes from `K=8` to `K=16` or `K=32`.

A raw sum would make the Intent contribution grow approximately with `K` and
the pressure-weighted Physics contribution grow approximately with `K^2`, while
the one-action repair cost would not grow with `K`. That would silently change
the Recovery-Aware ordering at every K transition. Effective-time
normalization keeps the meaning of one semantic unit stable across the
curriculum.

The applicability denominator must describe the intended evidence exposure,
not reward missing evidence or early termination:

- continuous pose, velocity, support-foot drift, and applicable phase-ZMP
 channels use the direction-sensitive weighted exposure above;
- Contact keeps mismatch-event/exposure semantics, so an early illegal support
 switch cannot be discounted merely because it occurred early;
- survival keeps lost-horizon semantics, so early failure remains maximally
 consequential rather than receiving a small `tau_k` weight;
- phase-ZMP uses loaded-support phase exposure only;
- valid actual no-load remains a Contact violation and phase-ZMP `N/A`.

## Confirmed Intent Evidence Dictionary

Intent compares the executed Noisy or Repair trajectory with the executed Clean
rollout. It does not compare the Noisy and Clean q29 reference arrays, because a
root-only artifact can leave those arrays numerically identical while the
executed motion is visibly wrong. Global horizontal translation is deliberately
excluded: Demo quality is more sensitive to body attitude and relative pose
than to harmless global drift.

| Intent item | `D_j(X,Clean)` means | Active fixed `S_j` |
| --- | --- | --- |
| Root orientation | K-step root-orientation geodesic error; directly exposes sustained lateral lean | `0.087 rad` (`5 deg`) |
| Joint pose | Per-joint RMS error between executed q29 and the Clean rollout | `0.087 rad` (`5 deg`) |
| Key-body pose | Hand, foot, and torso position error after removing horizontal global drift | `0.10 m` |
| Linear velocity | Root-local body linear-velocity error | `0.75 m/s` |
| Angular velocity | Joint/body angular-velocity error | `2.0 rad/s` |
| Root height | Clean-relative root-height error for sink or float artifacts | `0.05 m` |

Acceleration is retained as a diagnostic rather than an initial Gain channel,
because finite-difference acceleration is substantially noisier than the pose
and velocity evidence above.

## Confirmed Physics Evidence Dictionary

Physics measures whether Repair preserves the Clean rollout's expected support
semantics. It does not reward generic stability without regard to the intended
motion. Sustained lean without extra support compensation belongs to Intent;
extra stepping, dragging, or changed support belongs to Physics.

| Physics item | `D_j(X,Clean)` means | Active fixed `S_j` |
| --- | --- | --- |
| Contact phase | Fraction of intended foot-time containing extra/missed contact or an illegal support switch | `0.10` exposure fraction |
| Support-foot drift | Loaded-foot slip, drag, or unplanned support displacement | `0.03 m` |
| Phase-ZMP | Mean depth outside the Clean-conditioned support/recovery envelope during loaded-support applicability | `0.02 m` |
| Survival | Fraction of the K-step horizon lost after failure | `0.10` horizon fraction |

These fixed semantic units are not copied reward weights. They separate pose,
velocity, end-effector, Contact, and balance channels because their physical
units and acceptable errors differ, and remain fixed across Segments, attempts,
and K stages.

`S_j` affects optimization sensitivity, not just displayed units. Intent Gain
scales approximately as `1/S_I`. Because Physics improvement is multiplied by
normalized remaining pressure, its contribution scales approximately as
`1/S_P^2`: halving a Physics scale can approximately quadruple that channel's
effect. A scale that is too small amplifies rollout noise and can suppress
Intent; a scale that is too large can hide sustained lean, unplanned stepping,
or support drift. `S_j` must therefore remain a fixed, physically interpretable
unit rather than a dataset standard deviation, candidate-dependent gap,
learned weight, or clipping threshold.

## Confirmed Within-Family Aggregation

Intent channels and Physics channels are each aggregated with the same smooth
worst-item operator:

\[
\mathcal M(z_1,\ldots,z_n)
=\log\left(\frac{1}{n}\sum_{j=1}^{n}e^{z_j}\right).
\]

Each `z_j` is one normalized remaining problem `r_j`; it is not a new metric.
Because every `r_j` is already expressed in one fixed semantic unit, the
operator uses an implicit temperature of one semantic unit rather than adding
another tunable coefficient. The `1/n` term makes the result invariant to
family size when all residuals are equal: if every `z_j=c`, then
`M(z_1,...,z_n)=c`.

The two families remain separate:

\[
I_X=\mathcal M\bigl(\{r_{I,j}(X\mid C)\}_j\bigr),\qquad
P_X=\mathcal M\bigl(\{r_{P,j}(X\mid C)\}_j\bigr).
\]

This implements limited compensation within each family. A severe residual
increasingly controls the family score, so several small improvements cannot
freely hide one visibly bad pose or support failure. Unlike a hard maximum,
all retained channels still influence the ordering and near-tied channels vary
smoothly. This is not the retired final scalarization
`max_j r_j + eta sum_j r_j`: the operator is applied separately inside Intent
and Physics before paired improvement and Recovery-Aware interaction. Raw
per-channel evidence remains available for diagnostics and fail-closed checks.

## Confirmed Signed Improvement And Recovery Pressure

The signed improvement form is the result of two refinements to the older
`(Repair-Clean)-(Noisy-Clean)` intuition. First, raw state subtraction was
replaced by two Clean-conditioned semantic evaluations, `D_j(Noisy,Clean)` and
`D_j(Repair,Clean)`. Clean therefore defines how planned support changes,
dynamic lean, and intended pose are interpreted instead of cancelling
algebraically. Second, the unstable per-scenario recovery ratio was replaced by
fixed-scale normalized differences. This avoids a noisy `Noisy-Clean`
denominator while retaining absolute recovery pressure.

After within-family aggregation, attempt `m` receives the paired improvements

\[
G_I^{(m)}=I_N-I_R^{(m)},\qquad
G_P^{(m)}=P_N-P_R^{(m)}.
\]

Clean defines the correct direction, Noisy defines the zero-action baseline,
and Repair supplies the candidate consequence. Positive improvement only
requires Repair to reduce the corresponding remaining problem relative to
Noisy; it does not require Repair to reach Clean.

`G_P` records how much aggregated Physics pressure changed, but not whether the
same change occurred during severe recovery or near an already stable state.
The Recovery-Aware pressure coefficient is therefore

\[
\lambda_{RA}^{(m)}=\frac{P_N+P_R^{(m)}}{2}.
\]

It is the average remaining Physics pressure across the Noisy-to-Repair
transition. Its primary role is to give Physics changes more weight while the
robot is strongly imbalanced, and less weight after the robot is already near
the Clean physical regime. Multiplying it by signed Physics improvement gives
the exact decrease of a quadratic recovery potential:

\[
\lambda_{RA}^{(m)}G_P^{(m)}
=\frac{P_N^2-(P_R^{(m)})^2}{2}.
\]

The endpoint average is necessary because `G_P` alone records only the size of
the change. A transition from pressure `10` to `8` and a transition from `3`
to `1` both have `G_P=2`, although the former occurs while Physics recovery is
still urgent. The pressure-weighted contributions become `9 x 2 = 18` and
`2 x 2 = 4`, respectively.

Using only `P_N` as the coefficient would under-penalize a Repair that starts
near stability and creates a new severe imbalance. Using only `P_R` would
under-credit a Repair that starts in severe imbalance and ends near stability.
The symmetric endpoint average represents the whole transition and is exactly
the coefficient required by the quadratic-potential difference above.

The corresponding candidate ranking form is

\[
G_{\mathrm{total}}^{(m)}
=G_I^{(m)}
+\lambda_{RA}^{(m)}G_P^{(m)}
-\beta C_{\mathrm{repair}}^{(m)}.
\]

This construction gives the same amount of Physics improvement more credit
when the remaining pressure is high. Near a stable state, a small Physics
degradation produces only a small negative contribution, so a sufficiently
valuable Intent improvement may determine the ordering. If Repair creates a
new severe imbalance, its own `P_R` raises `lambda_RA` while `G_P` becomes
negative, so the harmful contribution is automatically amplified. A candidate
does not need to reach Clean or satisfy a minimum improvement threshold.

`lambda_RA` is computed independently for every attempt from that attempt's two
endpoints. It therefore changes continuously with observed recovery pressure
instead of being selected from a hand-authored recovery stage. The same Clean
anchor is used for `P_N` and `P_R`, so the coefficient cannot change by
switching reference semantics between candidates.

The Noisy subtraction preserves the do-nothing counterfactual. The different
values across M attempts expose the best currently reachable one-action
direction, including the least harmful direction when all attempts are adverse;
all valid attempts still remain training rows.

## Confirmed Full-6D Repair Cost

Repair cost does not judge whether the action improves Intent or Physics; the
Clean-anchored rollout terms above own that decision. Its only role is to prefer
the smaller intervention when two Repair outcomes are otherwise comparable.
Translation and rotation are therefore converted into fixed semantic repair
units before they are combined:

\[
C_{\mathrm{repair}}^{(m)}
=
\sqrt{
\left(\frac{\lVert\Delta t^{(m)}\rVert_2}{S_t}\right)^2
+
\left(\frac{\lVert\Delta\theta^{(m)}\rVert_2}{S_\theta}\right)^2
},
\qquad
S_t=0.10\ \mathrm m,
\quad
S_\theta=5^\circ\approx0.087\ \mathrm{rad}.
\]

The square root combines translation and rotation, not Physics and Intent.
`S_t` and `S_theta` state that 10 cm of translation and 5 degrees of rotation
each count as one repair unit; they are fixed across Segments, attempts, and K
stages. They are not per-axis Actor scales, dataset statistics, or
candidate-dependent normalization. No mask, `tanh`, `clip`, or `clamp` is
introduced.

Under one-action-K training, this cost measures only the one full-6D action at
`t`. Adjacent-frame action change remains a deployment diagnostic rather than a
Stage-3 Gain term, because one-action-K contains no consecutive FEMR action
sequence from which to define that penalty without expanding the method.

The first bounded live calibration starts from one explicit provisional global
value:

\[
\beta_{\mathrm{init}}=0.02.
\]

This value is the initial price of one semantic repair unit, not a claim that
`0.02` is already optimal and not a reused old raw-L2 coefficient. A bounded
live calibration run must retain the cost-free Recovery-Aware score

\[
R^{(m)}=G_I^{(m)}+\lambda_{RA}^{(m)}G_P^{(m)}
\]

and `C_repair` separately for every same-scenario attempt. A pair is informative
for calibration only when the attempt with greater recovery also uses a larger
repair action. For such a recovery-cost trade-off pair `a,b`, the observed
break-even coefficient is

\[
\beta_{ab}^{\star}
=
\frac{R^{(a)}-R^{(b)}}
     {C_{\mathrm{repair}}^{(a)}-C_{\mathrm{repair}}^{(b)}}>0,
\]

where the numerator and denominator have the same sign. If one attempt has
both higher recovery and no larger cost, it dominates the other and does not
define a positive cost-recovery trade-off. Pairs with numerically negligible
cost differences likewise do not estimate `beta` because their ratio is
ill-conditioned.

The live metrics therefore show at which coefficient a smaller action would
overtake a more effective but larger action. The same report must expose
`R`, `C_repair`, `beta * C_repair`, `G_total`, and whether adding the cost term
changes the within-Segment attempt order. These metrics do not automatically
mutate the objective. Human review may revise the single global `beta` between
explicit bounded calibration runs, starting from `0.02`, so that cost changes
only near-tied recovery rankings. Once accepted, that value is frozen across
Segments, attempts, and K stages. Updating `beta` inside a run or independently
per Segment or K would create a moving ranking standard and is forbidden.

## Confirmed PPO And Critic Consumption

The active scalar Stage-3 target is the complete Recovery-Aware score:

\[
\mathrm{return}_K^{(m)}=G_{\mathrm{total}}^{(m)},
\qquad
V^{RA}_K(o_t)\approx
\mathbb E[G_{\mathrm{total}}^{(m)}\mid o_t,\text{active }K].
\]

Every valid Repair attempt contributes one policy row and its own scalar
return. Clean and Noisy remain evaluator-only rows. Motion -> Segment ->
attempt grouped equal-mass reduction, zero optimizer steps during collection,
and exactly one optimizer step after the complete transaction seals remain
unchanged. Segment Replay ordering is therefore realized by all attempt
advantages, not by a winner-only update, replay-priority weight, or best-of-M
multiplier.

The old independent Physics-gradient projection is retired on this active
route. Contact phase, support-foot drift, phase-ZMP, and survival remain named,
fail-closed evidence, but their formal learning consumer is now `P_X -> G_P ->
G_total`; they do not also create a second actor constraint, projection cone,
KKT gate, or fallback update. Keeping both mechanisms active would count
Physics twice and could veto the mild Physics sacrifice that the confirmed
Recovery-Aware score intentionally allows near stability.

There is still one full-6D Actor, one scalar Critic, and one optimizer. The
Critic target changes from `paired-intent-minus-repair-v1` to the new
Recovery-Aware `G_total` identity; it does not become a Physics-specific or
multi-head Critic.

Critic recalibration follows the K-step Curriculum, not an independent M
Curriculum. Increasing K changes the executable-evidence horizon and therefore
changes the return distribution that the same Critic must predict. On every K
increase, the same Critic retains its weights but re-enters critic-only
recalibration while Actor/std remain frozen; actor-ramp and joint training then
resume, and the Critic continues updating in both phases. M remains the number
of Repair attempts drawn from one sealed Segment for ranking. A change in M
does not define a new Critic target and does not independently trigger Critic
warmup.

## Preserved Evidence

The active design retains ordered, role-specific rollout evidence until the final
Gain/optimization boundary:

- expected and actual Contact phase;
- phase-conditioned ZMP applicability, violation, and recovery;
- survival and terminal facts;
- pose/Intent deviation needed to detect stable but visibly tilted repairs;
- scenario, `x_t`, K, valid-step, and rollout identity.

Clean supplies the motion-phase anchor. Noisy supplies the no-action
counterfactual. Repair supplies candidate consequences. Segment Replay supplies
the repeated candidate set. None of these roles implies a new actor input.

## Remaining Engineering Calibration

The active contracts close the method semantics. Two implementation/calibration
facts remain deliberately outside this rationale note:

- the final global `beta` selected after bounded live break-even telemetry; the
  first calibration run uses the active provisional value `0.02`;
- whether implementation restores any extra simulator RNG state beyond the
  sealed scenario and one-time Clean/Noisy lifecycle; exact stochastic
  cancellation is not a method requirement.

The old expression `(Repair-Noisy)-(Clean-Noisy)`, a recovery ratio, a fixed
weighted sum, the retired `max_j r_j + eta sum_j r_j` scalarization, and a hard
Physics-to-Intent switch remain rejected alternatives.

## Contract-Version Boundary

Human confirmation activated the coordinated semantic migration on 2026-08-01:

```text
FRS-METHOD-v017  active Recovery-Aware method authority
FRS-GAIN-v007    Clean-anchored Intent/Physics Gain owner
FRS-PPO-v005     grouped scalar all-attempt PPO without Physics projection
FRS-TRAIN-v012   new scalar-Critic target and persistence identity
```

The former `FRS-METHOD-v016`, `FRS-GAIN-v006`, `FRS-PPO-v004`, and
`FRS-TRAIN-v011` are historical. Checkpoint-v6 cannot be treated as a strict
full-resume artifact for the changed Critic target; `FRS-TRAIN-v012` defines the
checkpoint-v7 boundary. Contract activation is not permission to edit source:
the existing runtime remains contract-mismatched until a separately authorized
Engineering Plan is accepted and implemented.
