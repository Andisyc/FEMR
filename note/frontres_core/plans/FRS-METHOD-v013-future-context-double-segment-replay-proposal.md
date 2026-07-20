# FRS-METHOD-v013 Proposal: Future-Conditioned Double Segment Replay

Status: resolved historical proposal
Date: 2026-07-18
Activated by: `FRS-METHOD-v013` on 2026-07-19
Workflow stage: Stage 2 complete; retained only as discussion provenance

## Problem

The current FrontRES actor receives the current GMT-compatible reference and a
short observation history, but no explicit future reference. Two motion
segments can therefore look equivalent at the current frame while requiring
opposite repairs because their upcoming motion intent differs. In that case an
MLP cannot infer the correct repair direction from the available observation;
averaging toward a small or no-op correction is a rational consequence of an
ambiguous input.

The current formal Stage 3 transaction also does not implement the accepted
Double Segment Replay semantics. It performs one PPO-eligible policy attempt
per Segment and updates immediately, while later trials are search evidence
only. This prevents repeated on-policy attempts under one frozen old policy
from establishing a reliable local repair direction before a cross-Segment PPO
update.

## Confirmed Conceptual Decisions

### FRS-DP-10 / M-11: Future Motion Context

FrontRES must receive enough deployable future reference context to distinguish
segments whose current state is similar but whose required repair direction is
different.

The intended starting point is:

```text
current reference
+ sparse future reference over approximately 0.5 seconds
+ current robot/tracking state
+ existing short history and balance context
-> full-6D FrontRES repair
```

The future context informs repair direction. It does not authorize FrontRES to
replace GMT, generate a new motion, or plan the whole sequence. GMT remains the
frozen joint-control owner.

GMT uses immediate motion targets together with a much longer future window
(approximately two seconds) for general tracking. FrontRES has a narrower
reference-repair role, so this proposal starts from a sparse local window rather
than copying GMT's full tracking horizon.

Reference: [GMT: General Motion Tracking for Humanoid Whole-Body Control](https://gmt-humanoid.github.io/resources/gmt.pdf).

### FRS-DP-02 / SR-01: Double Segment Replay

One policy version must remain frozen while the collector obtains repeated
on-policy attempts for each selected Segment. Multiple Segments may then be
combined into one PPO batch because the observation, including future motion
context, conditions the repair on each Segment's motion intent.

```text
freeze old policy
-> select multiple Segments
-> repeat on-policy attempts per Segment
-> establish local action-to-Gain evidence
-> aggregate eligible attempts across Segments
-> one PPO update
-> update Segment replay evidence and priorities
```

Search or counterfactual actions may inform Segment evidence, but they cannot
receive PPO credit as if sampled by the old policy.

## Joint Method Meaning

The two changes are coupled:

- Future Motion Context makes opposite repair requirements distinguishable in
  the policy input.
- Double Segment Replay reduces the chance that one stochastic attempt is
  mistaken for a stable repair direction.
- Cross-Segment batching is meaningful only after both conditions hold: each
  attempt is on-policy under one frozen policy, and the observation contains
  the information needed to condition different repair directions.

Neither change alone closes the learning problem. Repeated trials cannot resolve
an observation ambiguity, and future context cannot make a single noisy trial
reliable PPO evidence.

## Forbidden Assumptions

- Current frame plus past history is always sufficient to identify repair
  direction.
- Two visually similar current frames necessarily require the same repair.
- Future context turns FrontRES into a motion generator or second tracker.
- One policy attempt per Segment is sufficient Double Segment Replay evidence.
- Search trials can be relabeled as on-policy actions.
- Repeated visits after optimizer updates are equivalent to repeated attempts
  under one frozen old policy.
- Opposite actions across distinguishable observations imply an intrinsic
  gradient conflict.

## Resolved During Activation

1. Future input uses the deployed GMT-compatible command representation, not a
   new Clean-root/contact/noise-truth channel.
2. Positive sparse offsets are a serialized configuration within an
   approximately 0.5-second local window; exact numeric offsets are
   implementation configuration, not a second method branch.
3. One fixed Noisy reference segment supplies current, future, and K-step
   execution reference for every repeated attempt.
4. Every selected Segment contributes M_s >= 2 policy-sampled attempts under
   one frozen old policy; no optimizer mutation occurs during collection.
5. A transaction contains multiple selected Segments and exactly one PPO step.
6. Nested motion -> Segment -> attempt -> valid-step averaging, plus
   non-amplifying sign-preserving group scaling, prevents M/K/priority mass
   dominance.

## Non-Scope

- No observation tensor or network change in this proposal step.
- No sampler, storage, PPO, Gain, or optimizer modification.
- No live test or long training.
- No final commitment to a two-second GMT-style future encoder.

## Activation Gate

This proposal was activated as `FRS-METHOD-v013`. The active implementation
route is now `FRS-v013-future-context-double-segment-replay-engineering-plan.md`.
The existing formal path remains a code-confirmed contract mismatch, so code,
tests, and live execution are governed by that plan rather than this proposal.
