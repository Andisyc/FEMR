---
contract_id: FRS-GAIN-v007
status: active
effective_date: 2026-08-01
updated_date: 2026-08-03
supersedes: FRS-GAIN-v006
scope: Clean-conditioned Intent and Physics evidence, continuous remaining-Physics pressure, full-6D repair cost, and one scalar Recovery-Aware G_total for every valid Repair attempt
---

# Clean-Anchored Recovery-Aware Repair Gain

## Design Delta

FRS-GAIN-v006 supplied a scalar paired Intent objective and separate Contact,
phase-ZMP, and survival constraint evidence. FRS-GAIN-v007 keeps the validated
physical evidence authority but changes its formal consumer. Physics and Intent
now interact continuously inside one scalar candidate ordering:

```text
Clean  -> defines correct phase, support, dynamic lean, and intended pose
Noisy  -> defines zero improvement under no repair
Repair -> supplies the consequence of one full-6D action
```

The active scalar is neither the old `(Repair-Clean)-(Noisy-Clean)` state
difference nor a recovery ratio. It evaluates two Clean-conditioned remaining
problems, subtracts Repair from the Noisy zero point, weights Physics change by
the remaining Physics pressure, and subtracts one full-6D repair cost.

## Concept Figure Mapping

| Design ID | Canonical human name | Figure block ID | Contract section |
| --- | --- | --- | --- |
| `FRS-DP-06` | Paired Rollouts | `Q-PAIR` | Evidence Authority And Lifecycle |
| `FRS-DP-07` | Repair Gain | `Q-01` | Recovery-Aware Total Gain |

## Evidence Authority And Lifecycle

Every sealed Segment provides one observed Clean Rollout, one observed fixed
zero-action Noisy Rollout, and M observed Repair Rollouts. Clean and Noisy are
executed once and reused read-only for every Repair comparison. They never
become policy rows.

Expected Contact phase, expected foot pose/support envelope, and legitimate
dynamic lean come from the same Clean continuation used by frozen GMT. Actual
Contact and contact-wrench ZMP come from the validated filtered foot-to-ground
ContactSensor route. Missing APIs, malformed shapes/counts, non-finite valid
values, or inconsistent sensor payloads fail the transaction closed.

Valid actual no-load during expected support is a Contact violation. Phase-ZMP
is `N/A` for that role/step because no loaded-support resultant exists. Flight
is likewise semantic ZMP `N/A`. Neither case is silently converted to numeric
zero or excluded from the Contact consequence.

Clean evidence is evaluator-only. It never enters actor observation, HSL
target, PPO policy rows, deployment input, or the q29 future-intent carrier.

## Clean-Conditioned Remaining Problems

For role `X` in `{Noisy, Repair}` and retained channel `j`:

\[
r_j(X\mid C)=\frac{D_j(X,C)}{S_j}.
\]

`D_j` is the channel-specific remaining problem relative to the executed Clean
motion semantics. It is not one generic Euclidean distance. `S_j` is one fixed,
physically interpretable semantic unit. It is never recomputed from the current
Noisy-Clean gap, the M candidates, dataset standard deviation, K stage, or live
batch. No `clip`, `clamp`, `tanh`, or `[0,1]` saturation is applied.

This normalization aligns unlike physical units while preserving absolute
severity. It avoids the near-zero denominator and loss of severity produced by
per-scenario recovery ratios.

## K-Step Consequence

For every continuous retained channel:

\[
D_j^{\rightarrow}(X,C)
=
\frac{\sum_{k=1}^{K}a_{j,k}\tau_k e_{j,k}(X,C)}
     {\sum_{k=1}^{K}a_{j,k}\tau_k},
\qquad \tau_k=\frac{k}{K}.
\]

`e_{j,k}` is the frame-level Clean-conditioned problem and `a_{j,k}` is its
semantic applicability. Later residuals matter more, so equal ordinary means
can still distinguish recovery from deterioration. Dividing by weighted
applicable exposure keeps one semantic unit stable as K changes.

Contact retains mismatch-event/exposure semantics; an early illegal switch is
not discounted by `tau_k`. Survival retains lost-horizon semantics; early
failure remains maximally consequential. Phase-ZMP uses loaded-support phase
exposure only. Missing evidence never shrinks a denominator to improve a score.

## Intent Evidence Dictionary

Intent compares executed Noisy or Repair motion with the executed Clean
rollout. Global horizontal translation is removed because demo quality is more
sensitive to body attitude and relative pose than harmless world drift.

| Intent item | `D_j(X,Clean)` | Fixed `S_j` |
| --- | --- | --- |
| Root orientation | K-step root-orientation geodesic error, including sustained lateral lean | `0.087 rad` (`5 deg`) |
| Joint pose | Per-joint RMS q29 execution error | `0.087 rad` (`5 deg`) |
| Key-body pose | Hand, foot, and torso position error after horizontal-drift removal | `0.10 m` |
| Linear velocity | Root-local body linear-velocity error | `0.75 m/s` |
| Angular velocity | Joint/body angular-velocity error | `2.0 rad/s` |
| Root height | Clean-relative root-height error | `0.05 m` |

Acceleration remains diagnostic-only because finite-difference noise is not
required for the initial candidate ordering.

## Physics Evidence Dictionary

Physics asks whether Repair preserves the Clean rollout's expected support
semantics. It does not reward generic survival detached from the intended
motion.

| Physics item | `D_j(X,Clean)` | Fixed `S_j` |
| --- | --- | --- |
| Contact phase | Extra/missed Contact or illegal support-switch exposure | `0.10` exposure fraction |
| Support-foot drift | Loaded-foot slip, drag, or unplanned support displacement | `0.03 m` |
| Phase-ZMP | Mean depth outside the Clean-conditioned support/recovery envelope during loaded support | `0.02 m` |
| Survival | Fraction of the K-step horizon lost after failure | `0.10` horizon fraction |

Sustained lean without extra support compensation is an Intent error. Extra
stepping, dragging, missed support, or changed support is a Physics error. The
Clean anchor distinguishes planned dynamic support changes from reward hacking.

## Within-Family Aggregation

Intent and Physics are aggregated separately with the same smooth worst-item
operator:

\[
\mathcal M(z_1,\ldots,z_n)
=\log\left(\frac1n\sum_{j=1}^{n}e^{z_j}\right).
\]

Each `z_j` is one normalized `r_j`. A severe channel increasingly controls its
family while every retained channel remains visible. The `1/n` term keeps the
equal-residual baseline independent of family size. The implicit temperature is
one fixed semantic unit; there is no additional tunable temperature.

\[
I_X=\mathcal M(\{r_{I,j}(X\mid C)\}_j),\qquad
P_X=\mathcal M(\{r_{P,j}(X\mid C)\}_j).
\]

Raw signed, per-step, per-channel evidence remains available for diagnostics and
fail-closed validation. The aggregate must not erase applicability or identity.

## Signed Improvement And Recovery Pressure

For Repair attempt `m`:

\[
G_I^{(m)}=I_N-I_R^{(m)},\qquad
G_P^{(m)}=P_N-P_R^{(m)}.
\]

Positive Gain requires improvement over the fixed Noisy zero-action baseline,
not equality with Clean. The same-scenario M outcomes expose the currently
reachable direction even when every candidate remains far from Clean.

Physics change is weighted by the average remaining Physics pressure across
the transition:

\[
\lambda_{RA}^{(m)}=\frac{P_N+P_R^{(m)}}2,
\qquad
\lambda_{RA}^{(m)}G_P^{(m)}
=\frac{P_N^2-(P_R^{(m)})^2}{2}.
\]

Thus the same Physics improvement receives more credit while imbalance remains
large and less near the Clean physical regime. A Repair that creates severe
imbalance raises its own `P_R`, makes `G_P` negative, and amplifies the harmful
term. This continuous endpoint rule replaces both hard handoff and manually
scheduled Physics/Intent ratios.

## Full-6D Repair Cost

Repair cost compares intervention size only; it does not decide whether the
action improves Intent or Physics:

\[
C_{\mathrm{repair}}^{(m)}
=\sqrt{
\left(\frac{\lVert\Delta t^{(m)}\rVert_2}{0.10\,\mathrm m}\right)^2
+
\left(\frac{\lVert\Delta\theta^{(m)}\rVert_2}{5^\circ}\right)^2
}.
\]

Ten centimetres of translation and five degrees of rotation each define one
repair unit. The root combines translation with rotation, not Physics with
Intent. The scales do not become per-axis actor scales or output bounds.

## Recovery-Aware Total Gain

\[
G_{\mathrm{total}}^{(m)}
=G_I^{(m)}
+\lambda_{RA}^{(m)}G_P^{(m)}
-\beta C_{\mathrm{repair}}^{(m)}.
\]

This is the unique scalar return and candidate ordering. High Physics pressure
prioritizes recovery; near physical stability, Intent determines demo quality;
repair cost breaks otherwise comparable outcomes in favor of the smaller
intervention. Segment Replay supplies only all-attempt comparison and must not
add another score, gate, or winner rule.

## Beta Initialization And Live Calibration

The first bounded live calibration uses one provisional global value:

\[
\beta_{\mathrm{init}}=0.02.
\]

Define the cost-free score:

\[
R^{(m)}=G_I^{(m)}+\lambda_{RA}^{(m)}G_P^{(m)}.
\]

For a same-Segment pair where the attempt with greater recovery also uses a
larger action, the positive break-even coefficient is:

\[
\beta_{ab}^{\star}
=\frac{R^{(a)}-R^{(b)}}
       {C_{\mathrm{repair}}^{(a)}-C_{\mathrm{repair}}^{(b)}}>0.
\]

Dominated pairs, where one attempt has higher recovery and no larger cost, do
not calibrate beta. Numerically negligible cost differences likewise do not
estimate it. Live telemetry must expose `R`, `C_repair`, `beta*C_repair`,
`G_total`, and whether cost changes within-Segment attempt ordering.

Live execution never mutates beta. Human review may revise the single global
value between explicit bounded calibration runs. Once accepted, it remains
fixed across Segments, attempts, and K stages. The final numeric beta remains
an empirical calibration boundary, not a learned parameter or per-stage
controller.

## PPO And Critic Consumption

\[
\mathrm{return}_K^{(m)}=G_{\mathrm{total}}^{(m)}.
\]

Every valid Repair contributes one return and one policy row. The scalar Critic
owned by FRS-TRAIN-v014 predicts the expected complete `G_total` at the active K.
Contact, support-foot drift, phase-ZMP, and survival reach learning only through
`P_X -> G_P -> G_total`. They do not simultaneously produce independent actor
constraints, projection gradients, KKT gates, or fallback updates.

## Required Diagnostics

- raw per-step and aggregated Intent/Physics channels with applicability;
- `I_N`, `I_R`, `P_N`, `P_R`, `G_I`, `G_P`, and `lambda_RA`;
- full-6D action, `C_repair`, beta, cost-free `R`, and `G_total`;
- Contact preservation, support-foot drift, phase-ZMP recovery, survival,
  sustained lateral lean, and unplanned support changes;
- same-Segment attempt ordering with and without cost;
- scenario, hash, role, K, valid-step, transaction, and policy-row identity.

## Forbidden Behavior And Stop Conditions

- old scalar Intent-minus-cost target;
- independent Physics projection or KKT actor gate;
- recovery ratio or candidate-derived normalization;
- hard Physics/Intent switch, threshold, clamp, or winner selection;
- missing evidence filled with zero or invalid rows silently dropped;
- Clean/Noisy as actor observations or PPO rows;
- dynamic beta or Gain-driven sampling curriculum.

Stop if raw paired evidence and the scalar ordering disagree for a confirmed
case, a severe channel is erased by aggregation, applicability changes the
meaning of a valid outcome, an old consumer computes another Gain, or runtime
cannot preserve the complete raw evidence needed to falsify `G_total`.
