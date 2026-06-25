# FrontRES Repair Paper Closure

Date: 2026-06-24

This note records the paper-level interpretation that emerged from the
FrontRES method-boundary discussion.  Its purpose is to help close the current
work as a small, clean paper instead of continuing to expand it into dynamic
recovery, rho acceptance, or recovery-reference generation.

## 1. Core Decision

The current paper should be framed as a **reference repair** paper, not a
dynamic recovery paper.

The central idea is:

> Some motion-tracking failures are not caused by a weak tracker or by a robot
> state that needs recovery.  They are caused by corrupted reference frames that
> push a frozen tracker outside its robust input interface.  FrontRES repairs
> this reference interface before the frozen tracker consumes it.

Under this framing, HSL is not the paper's intellectual contribution by itself.
HSL is the simplest training mechanism that follows naturally from the problem
definition: if the clean reference is known and the noisy reference is
corrupted, the repair direction is observable in task space.

Therefore, the current paper should not sell:

> We train a supervised Delta SE(3) network.

It should sell:

> We identify reference-space corruption as a distinct failure source in frozen
> motion tracking and introduce a front-end task-space repair interface that
> restores executable references without retraining the tracker.

## 2. Why This Boundary Is Necessary

The discussion started from a recurring tension:

- The early HRL version already showed that a front-end residual module can
  repair corrupted references.
- HRL was noisy and its demo improvement was not always visually obvious.
- HSL improved the repair direction because Delta SE(3) is directly observable
  from clean/noisy reference pairs.
- High perturbation still exposed failures.
- Those high-perturbation failures tempted the method to expand into rho,
  authority critic, burst perturbation, k-step return, stable frames, and
  dynamic acceptance.

The key realization is that the high-perturbation failure does not necessarily
belong to the same paper.  It may be a different problem:

- **Reference repair** asks whether a corrupted reference can be mapped back
  toward a clean, tracker-compatible reference.
- **Dynamic recovery** asks whether the robot can actively recover from a body
  state that is already outside the direct execution neighborhood of the
  original motion.

Both can appear as tracking failure.  Both can reduce episode length.  But they
have different causal variables and require different evidence.

This is the point where the paper boundary should be cut.

## 3. Repair vs Recovery

The strongest conceptual distinction is:

> Repair looks at the reference side.  Recovery looks at the robot-state side.

### Reference Repair

Reference repair handles the case where:

1. The clean motion is still executable by the frozen tracker.
2. The reference fed to the tracker is corrupted.
3. The tracker fails because it is chasing the wrong reference, not because it
   lacks a recovery behavior.
4. The correct action is to repair the reference before the tracker consumes it.

The causal chain is:

```text
corrupted reference
  -> frozen tracker receives a wrong target
  -> originally executable motion becomes hard to track
  -> front-end repair maps the reference back toward clean-compatible space
  -> frozen tracker can reuse its existing robustness
```

The method variable is:

> task-space reference residual, Delta SE(3).

The evidence is:

> clean/noisy reference pairs.

The natural training mechanism is:

> supervised task-space residual learning.

### Dynamic Recovery

Dynamic recovery handles the case where:

1. The robot state has been disturbed by external force, bad contact, high
   velocity, severe phase mismatch, or loss of balance.
2. The original clean reference may no longer be immediately executable from
   the current physical state.
3. The policy must generate an active recovery transition before it can return
   to the original motion sequence.

The causal chain is:

```text
disturbed robot state
  -> current body state leaves the original motion's execution neighborhood
  -> even clean reference may be temporarily inadmissible
  -> policy must generate recovery behavior
  -> tracking resumes after the state returns toward the motion manifold
```

The method variable is not Delta SE(3) repair direction alone.  It is something
like:

> dynamic admissibility, recovery reference, acceptance authority, or recovery
> policy.

The evidence is no longer clean/noisy reference pairs alone.  It requires:

- rollout outcome,
- state-conditioned acceptance,
- counterfactual comparison,
- recovery trajectory quality,
- or another dynamic evidence source.

This is a different research object.

## 4. Why HSL Is the Right Mechanism for This Paper

The main concern was that HSL feels too simple or too low-level.  The resolution
is that HSL should not be presented as the contribution.  HSL is the appropriate
mechanism once the paper is defined as reference repair.

In the reference-repair boundary:

- The clean reference is known.
- The corrupted reference is known.
- The desired correction direction is directly observable.
- The output should stay in task space so it remains compatible with the frozen
  tracker.

Thus, the minimal mechanism is:

```text
Noisy reference -> task-space Delta SE(3) residual -> repaired reference
```

HSL is natural because the paper is not asking the model to discover a recovery
strategy.  It is asking the model to restore a corrupted input interface.

This turns simplicity into a strength:

> Because the problem is correctly cut as reference repair, the repair target is
> directly observable and does not require noisy RL credit assignment.

The paper should not apologize for using HSL.  It should explain why HSL is the
cleanest mechanism for isolating the reference-interface hypothesis.

## 5. Why HRL and Rho Should Not Be the Main Paper

HRL and rho became attractive because high perturbation exposed cases where a
clean-oriented repair may be dynamically harmful.  But this is exactly the
moment where the problem changes from repair to recovery/acceptance.

Rho asks:

> Given a proposed repair, how much authority should it have under the current
> dynamic state?

This is not the same as:

> What task-space residual maps the corrupted reference back toward the clean
> reference?

Rho depends on state-conditioned dynamic admissibility.  It requires rollout
evidence, baseline comparison, critic learning, or another feedback mechanism.
It is a legitimate future direction, but adding it to the current paper creates
conceptual mixing:

- The paper would no longer cleanly study reference repair.
- The training signal would mix clean/noisy supervised geometry with dynamic
  rollout acceptance.
- The method would become harder to explain and harder to validate.
- A failed rho run would make the whole paper look unstable even if the repair
  idea itself is valid.

Therefore, the current paper should treat rho/acceptance as a next-layer
problem:

> Once reference repair is established, a natural next step is to learn
> state-conditioned dynamic acceptance for cases where the clean-oriented repair
> may not be immediately executable.

This keeps the current paper clean and preserves rho as a future paper rather
than letting it consume the present one.

## 6. The Paper's Natural Claim

A strong minimal claim is:

> Under moderate reference-frame corruption where the clean reference remains
> executable by a frozen tracker, FrontRES recovers a substantial fraction of
> the performance lost by corrupted references, without retraining the tracker.

This claim has four important boundaries:

1. It is about **reference-frame corruption**, not generic physical recovery.
2. It assumes the clean reference remains within the frozen tracker's executable
   neighborhood.
3. It repairs the input reference, not the tracker policy.
4. It does not claim to solve dynamically inadmissible recovery cases.

The paper should not claim:

- universal robustness under arbitrary perturbations,
- recovery from severe physical disturbances,
- synthesis of a new recovery reference,
- dynamic acceptance of all clean-oriented repairs,
- replacement of GMT or other strong trackers.

The paper should claim:

- reference artifacts are a real and under-isolated failure source;
- a frozen tracker can fail because its reference interface is corrupted;
- task-space front-end repair is a lightweight, tracker-compatible solution;
- supervised Delta SE(3) learning is sufficient inside the reference-repair
  boundary;
- dynamic recovery is complementary and should be studied separately.

## 7. How To Make the Boundary Convincing

The boundary cannot be only a text statement.  It must be shown with figures and
curves.

The most important experiment should separate three conditions:

1. **Clean + GMT**
2. **Noisy + GMT**
3. **Repaired + GMT**

The intended curve should show a region where:

- Clean + GMT remains strong.
- Noisy + GMT drops.
- Repaired + GMT recovers performance.

That region is the reference-repair problem.

The high-perturbation region should be interpreted carefully.  If Clean + GMT
or oracle-style repair also becomes weak, then the problem has moved toward
dynamic recovery.  That does not invalidate FrontRES.  It marks the boundary of
the current paper.

The key boundary condition can be written as:

> This work studies perturbations for which the clean reference remains largely
> executable by the frozen tracker, while the corrupted reference degrades
> tracking.

This is much stronger than saying:

> We do not handle high perturbation.

The first version defines a scientific object.  The second sounds like a
weakness.

## 8. Suggested Figure Logic

The paper should use figures to make the problem split visible.

### Figure 1: Repair vs Recovery

Show two pipelines:

```text
Reference repair:
corrupted reference -> FrontRES repair -> frozen tracker -> robot

Dynamic recovery:
disturbed robot state -> recovery policy -> return to motion manifold -> tracker
```

Caption message:

> Reference repair handles corruption before the tracker consumes the reference.
> Dynamic recovery handles physical state deviations after the robot has already
> left the direct execution neighborhood.

### Figure 2: Frozen Tracker Interface

Show:

```text
Clean reference ---------------> GMT -> good tracking
Noisy reference ---------------> GMT -> failure
Noisy reference -> FrontRES ---> GMT -> recovered tracking
```

Caption message:

> FrontRES is not a replacement for GMT.  It is a front-end repair layer that
> restores a cleaner input interface for a frozen tracker.

### Figure 3: Boundary Curve

Plot perturbation scale on the x-axis and performance on the y-axis:

- Clean + GMT
- Noisy + GMT
- FrontRES + GMT

The important region is where Clean + GMT is still high, Noisy + GMT drops, and
FrontRES + GMT recovers.

Caption message:

> This region isolates reference repair from dynamic recovery.

### Figure 4: Demo Pair

Show short visual comparisons:

- corrupted reference causes GMT to chase a wrong target;
- FrontRES repaired reference brings the motion closer to clean execution.

The demo should not try to look like push recovery.  It should look like
reference correction.

## 9. Reviewer Question: Why Not Dynamic Recovery?

Expected reviewer question:

> Why does the paper not solve dynamic recovery under high perturbation?

Recommended answer:

> Dynamic recovery and reference repair address different causal failures.
> Recovery methods handle physical state disturbances after the robot has left
> the original motion's execution neighborhood.  Our work studies a different
> failure source: corrupted references that degrade a frozen tracker even when
> the clean motion remains executable.  We isolate this reference-interface
> problem because it is measurable, directly correctable, and complementary to
> recovery.  Once reference repair is established, state-conditioned dynamic
> acceptance or recovery-reference generation becomes a natural next step.

Short version:

> We do not replace recovery policies.  We repair the reference interface before
> recovery becomes necessary.

## 10. Reviewer Question: Is This Just HSL?

Expected reviewer question:

> Is the method just supervised residual learning?

Recommended answer:

> The contribution is not a new supervised loss.  The contribution is the
> problem decomposition: a class of tracking failures can be treated as
> reference-interface corruption rather than policy insufficiency.  Under this
> decomposition, the clean/noisy reference pair makes the task-space repair
> direction directly observable, so supervised Delta SE(3) learning is the
> minimal and appropriate mechanism.  Using RL here would entangle reference
> repair with downstream recovery credit assignment, which is deliberately
> outside the current problem boundary.

Short version:

> HSL is simple because the problem has been cut correctly.

## 11. Reviewer Question: Why Not HRL/Rho?

Expected reviewer question:

> Since repairs may be dynamically harmful, why not learn a rho/acceptance
> policy?

Recommended answer:

> Rho represents dynamic acceptance authority, not reference repair direction.
> It asks whether a clean-oriented repair should be executed under the current
> physical state.  That is a second problem requiring rollout evidence and
> state-conditioned admissibility.  The present work first establishes the
> reference-repair layer under the assumption that the clean reference remains
> executable.  Dynamic acceptance is complementary and should be studied after
> the repair interface is isolated.

Short version:

> Rho belongs to recovery/acceptance.  This paper isolates repair.

## 12. What To Do With High Perturbation Results

High perturbation should not be hidden, but it should be used carefully.

It can serve as:

- a boundary diagnostic;
- a motivation for future dynamic acceptance;
- a demonstration that reference repair and recovery are different problems.

It should not be used as the main success region if HSL cannot solve it
reliably.

The paper can say:

> At stronger perturbation scales, failures increasingly involve dynamic
> inadmissibility rather than only reference corruption.  This regime requires
> recovery or state-conditioned acceptance and is outside the present
> reference-repair assumption.

This makes the limitation intellectually controlled instead of looking like an
unhandled failure.

## 13. Practical Closing Strategy

The current work should be closed as a small, clean paper if the following can
be shown:

1. Reference corruption causes measurable degradation for GMT.
2. Clean + GMT remains strong in the target perturbation region.
3. Noisy + GMT drops in that same region.
4. FrontRES + GMT recovers a meaningful fraction of the loss.
5. The demo visibly shows reference repair, not recovery.
6. The method is described as a frozen-tracker-compatible front-end repair
   architecture.
7. High perturbation is presented as a boundary/future-work regime.

If these conditions hold, the work does not need rho or dynamic recovery to be
paper-worthy.

## 14. Final Paper Spine

The cleanest spine is:

```text
Observation:
  Motion tracking failures can arise from corrupted references, not only from
  insufficient policy robustness.

Problem:
  Reference-space corruption consumes the robustness budget of a frozen tracker.

Boundary:
  We study the regime where the clean reference remains executable, but the
  corrupted reference degrades tracking.

Architecture:
  Add a front-end task-space repair layer before the frozen tracker.

Mechanism:
  Predict Delta SE(3) residuals to map noisy references back toward clean
  tracker-compatible references.

Training:
  Use HSL because the repair target is directly observable from clean/noisy
  reference pairs inside the repair boundary.

Evidence:
  Clean/Noisy/Repaired curves and demos show that the repair layer restores
  tracking performance without retraining the tracker.

Boundary/Future:
  Dynamic recovery and state-conditioned acceptance are complementary next
  problems when clean-oriented repair is no longer immediately executable.
```

## 15. One-Sentence Summary

> FrontRES should be closed as a reference-repair paper: it identifies corrupted
> motion references as a distinct interface-level failure source and shows that
> a lightweight task-space residual layer can repair this interface for a frozen
> tracker; HSL is not the novelty itself, but the natural minimal mechanism once
> the problem is correctly separated from dynamic recovery.

---

## 16. Updated Direction: Recovery-Aware Repair, Not HSL-Only

The later branch discussion revised the HSL-only closure.  HSL-only repair is
clean, but likely too weak as the central contribution because it only solves a
geometric reference-repair problem.  The stronger and more natural problem is
the coupled regime:

```text
the robot is lightly destabilized, and the reference frame is still corrupted.
```

This is not full Recovery.  The method should not learn a new recovery policy
or synthesize a new recovery reference.  It should repair corrupted references
while respecting the frozen GMT tracker's ongoing recovery tendency.

The updated paper problem is:

> How can a front-end repair module correct corrupted references without
> interrupting the frozen tracker when the robot is already lightly
> destabilized?

This shifts the method from HSL-only geometric repair to recovery-aware repair.

## 17. New Component Boundary

The cleanest current factorization is:

```text
HSL = dynamics-aware proposal generation
HRL = rollout-based admissibility selection
GMT = frozen execution policy
```

HSL no longer means "blindly predict the full Clean-Noisy residual."  HSL should
produce a locally executable repair proposal:

```text
Delta r_HSL = conservative Clean-oriented repair proposal
```

It owns continuous repair magnitude.  This is the part that old HRL/rho tried
to learn but could not learn reliably from endpoint rollout comparisons.

HRL should no longer own continuous per-dimension rho strength.  It should own
binary or near-binary admissibility:

```text
m_HRL = should this proposal be written under the current state?
```

The execution rule becomes:

```text
r_out = r_noisy + m_HRL * Delta r_HSL
```

or, for dimension/family-level decisions:

```text
r_out = r_noisy + m_HRL \odot Delta r_HSL
```

## 18. Why This Fixes The Old Preference / Advantage Problem

The old continuous-rho formulation asked Noisy/Clean/Candidate endpoint
rollouts to identify a continuous write strength.  This was not identifiable.
Those rollouts can answer:

```text
Was applying this proposal better than doing nothing?
```

They cannot reliably answer:

```text
What exact continuous rho value should each dimension use?
```

The updated design matches evidence to variable:

- HSL receives the continuous magnitude problem because it can be trained as a
  supervised, locally constrained proposal.
- HRL receives only the accept/reject problem because Noisy-vs-Candidate
  rollout comparison can provide that label.

This does not discard HRL.  It makes HRL responsible for the part that rollout
evidence can actually support.

## 19. What HSL Is And Is Not

HSL is:

```text
a locally executable Clean-oriented proposal generator
```

It may use local dynamics priors such as smoothness, bounded magnitude, local
GMT action consistency, or family-specific conservative targets.

HSL is not:

```text
a recovery policy
```

It does not guarantee that the proposal will improve rollout.  It only prevents
the proposal from being a crude geometric jump.  Long-horizon rollout success
or failure is still handled by HRL.

Short version:

```text
HSL makes the repair gentle.
HRL decides whether the gentle repair is admissible.
```

## 20. HRL's Remaining Necessity

HRL exists because HSL has failed corners.  Even a locally gentle proposal can
be harmful when:

- it is locally smooth but rollout-harmful;
- it is geometrically correct but conflicts with GMT's current recovery trend;
- the robot state is near the boundary where any reference correction increases
  control burden;
- the active perturbation family interacts with contact, phase, or velocity in
  a way that local smoothness cannot predict.

Therefore HRL's paper role is:

```text
reject proposal failures using rollout evidence
```

A suitable HRL label is based on Noisy-vs-Candidate comparison:

```text
Candidate better than Noisy -> accept
Candidate worse than Noisy  -> reject
margin region               -> ignore or low weight
```

Clean remains useful for proposal construction and upper-reference analysis,
but the HRL decision should be tied to whether Candidate improves over the
Noisy baseline.

## 21. Updated Main Method Sentence

The strongest current method sentence is:

> FEMR separates reference repair into dynamics-aware proposal generation and
> rollout-based admissibility selection.

Expanded:

> FEMR first converts a clean-oriented geometric correction into a locally
> executable repair proposal, then learns a binary admissibility policy to
> reject proposal failures under rollout evidence.

This should replace the weaker HSL-only closing story if the implementation is
updated accordingly.

## 22. Implementation Contract For The Next Code Pass

The next code pass should follow these constraints:

1. HSL owns continuous repair magnitude.
2. HRL owns accept/reject admissibility, not continuous rho strength.
3. Noisy/Candidate rollout evidence should train binary admissibility labels.
4. Old continuous-rho branches must be removed from the active path or hard
   gated as explicit ablations.
5. Diagnostics should report HSL proposal magnitude, HRL accept/reject rate,
   Candidate-vs-Noisy margin, accepted-beneficial fraction, and
   rejected-harmful fraction.

