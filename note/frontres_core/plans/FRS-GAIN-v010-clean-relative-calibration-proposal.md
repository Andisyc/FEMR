# FRS-GAIN-v010 Clean-Relative Calibration Proposal

Status: proposal / not active / offline design only  
Human decision: Clean is a Scenario-specific execution anchor, never an
absolute safety oracle.  
Supersession boundary: none. `FRS-GAIN-v009` remains active until the complete
v010 module and formal route are proved.

## Problem

FRS-GAIN-v009 uses independently handwritten thresholds for L3 membership and
one global numerical resolution for Pareto comparison. The current log shows
that `angular_momentum_error <= 0.10` and `stable_hold_steps >= 4` are never
satisfied, while a `1e-6` comparison resolution treats physically negligible
continuous differences as distinct. The hierarchy is retained; only the
source of ordinary continuous tolerances changes.

## Preserved hard Physics

The following evidence remains absolute and non-compensable:

\[
survival=0,\quad no\_load>0,\quad unplanned\_switch>0,\quad
illegal\_contact>0.
\]

Capture and applicable phase-ZMP must also remain on their physically valid
side. Clean similarity never converts an absolute violation into admissibility.

## Calibration artifact

One immutable artifact is produced before training from repeated executions of
the same sealed Clean Scenario identities. For every continuous field `j`, it
stores a positive, finite resolution

\[
\delta_j=Q_{1-\alpha}\!\left(
 |\phi_j(C^{(1)})-\phi_j(C^{(2)})|
\right),
\]

plus the field units, Scenario/robot/GMT/cache identities, sample counts,
coverage `1-alpha`, horizon convention and content hash. A field with no
repeated evidence, zero/non-finite resolution or mismatched identity is
invalid. No training batch, Repair outcome, Actor checkpoint or learned policy
may determine `delta_j`.

The current single-Clean transaction is insufficient to build this artifact.
The offline public producer can now construct and hash the artifact from
repeated same-Scenario window observations, but no real repeated-Clean artifact
has been collected. The current transaction supplies only the current Scenario
anchor.

## Scenario-relative evidence

For the current sealed Scenario, Clean supplies the center `c_j`; the artifact
supplies the resolution `delta_j`. After orienting every coordinate so larger
means better, define the globally anchored integer evidence

\[
b_j(R\mid C)=
\operatorname{round}\!\left(\frac{\phi_j(R)-c_j}{2\delta_j}\right).
\]

For one Scenario, all M Repairs use exactly the same `C`, `delta`, field order
and applicability domain. Pair-specific epsilon comparisons are forbidden.

## Levels

\[
level(R)=
\begin{cases}
L_0,&\text{Survival fails},\\
L_1,&\text{a hard contact/support rule is violated},\\
L_3,&\text{absolute Physics is valid and the final }W\text{ steps remain in the Clean tube},\\
L_2,&\text{otherwise}.
\end{cases}
\]

The Clean tube is the intersection of the calibrated per-field normal bins;
it is not a weighted sum. `W` is specified in physical duration and converted
to steps from the simulator timestep. The only human semantic controls are
the Clean coverage level `1-alpha` and required stable duration `W`.

## Relation

Cross-level comparison remains lexicographic. Within L1, the exact discrete
severe vector remains Pareto ordered. Within L2, exact Pareto is applied to
`b(R|C)`. Thus one coordinate that crosses a calibrated bin can establish an
improvement when all other coordinates remain in the same or better bins; a
real cross-coordinate tradeoff remains `INCOMPARABLE`. L3 retains Intent
Pareto followed by repair cost only when Intent is tied.

## Public boundary and state

The future owner accepts:

```text
Clean trajectory + Repair trajectory + expected support + Repair action
+ immutable CleanCalibration artifact
-> Outcome + BETTER/WORSE/SAME/INCOMPARABLE/INVALID
```

The comparator is deterministic and stateless. The artifact is read-only and
must be identity-bound in configuration and checkpoints. Missing, stale or
mismatched calibration fails the transaction before edge construction and
before optimizer mutation. There is no online calibration, fallback threshold,
zero-fill, second scalar score or policy-dependent update.

## Required offline proof

1. identical Clean repeats produce SAME bins;
2. sub-resolution differences remain SAME;
3. one supra-resolution improvement with no other degradation is BETTER;
4. supra-resolution tradeoffs remain INCOMPARABLE;
5. absolute Physics violations remain L0/L1 despite Clean similarity;
6. calibration identity, units, field order, positivity and sample coverage
   fail closed;
7. row and edge order do not change relations;
8. a controlled handwritten-threshold mutant is detected.

## Stop boundary

Do not modify the active Gain route or start training until a public repeated-
Clean calibration producer exists, the artifact has real data, the candidate
module is `MODULE-CORRECT`, and formal audit proves the artifact reaches the
official v010 comparison without training-state mutation.
