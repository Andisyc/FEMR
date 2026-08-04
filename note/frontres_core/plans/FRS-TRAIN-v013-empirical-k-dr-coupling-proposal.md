# FRS-TRAIN-v013 Proposal: Empirical K-DR Curriculum Coupling

Status: resolved and extracted into active `FRS-TRAIN-v013` on 2026-08-03.
Created: 2026-08-03. Retained only as the design-rationale record for
`contracts/active/training/FRS-TRAIN-v013-nested-k-dr-curriculum.md`.

Affected design points:

- `FRS-DP-01` / `M-02`: Perturbation Data
- `FRS-DP-03` / `M-06`: K-step Curriculum
- `FRS-DP-09` / `M-05`: Actor & Critic Warmup

## Problem

Active TRAIN-v012 assumes that each K first calibrates and freezes an
independent frozen-GMT Noisy frontier `g_K`, then starts Critic recalibration.
It does not define the statistic, observation budget, or freeze rule. More
importantly, frozen-GMT survival alone cannot determine when K8 evidence stops
being sufficient for ranking the Repair attempts learned by FEMR.

The unresolved object is therefore not a better pre-training estimator for
`g_K`. It is the empirical interaction between corruption strength and the
horizon required to preserve Segment Replay ordering.

## Candidate Design

### Perturbation reference for the first campaign

The historical frozen-GMT survival cliff near `dr_scale=2.381` is retained as a
provisional reference ceiling for the first campaign. It bounds the initial DR
sampling support; it is not a claim that every sample at 2.381 must become
solvable, a graduation threshold, or a universal per-K frontier.

The first campaign restores four explicit perturbation-strength classes. For
the current K-stage explicit ceiling `d_cap`, the fixed mixture is:

\[
\begin{aligned}
\mathcal D_{easy}   &: d\in[0,0.25d_{cap}),          & w&=0.20,\\
\mathcal D_{medium} &: d\in[0.25d_{cap},0.70d_{cap}),& w&=0.30,\\
\mathcal D_{hard}   &: d\in[0.70d_{cap},d_{cap}],    & w&=0.40,\\
\mathcal D_{broken} &: d\in(d_{cap},\min(1.10d_{cap},2.381)], & w&=0.10.
\end{aligned}
\]

`Easy` preserves low-disturbance restoration and Demo quality. `Medium`
provides clear, normally repairable corruption and is the ordinary learning
region. `Hard` approaches the current executable/repair frontier and supplies
the main difficult recovery evidence. `Broken` is only a capped tail beyond
that frontier; it exposes the failure boundary but may not dominate actor
learning.

These names classify the sampled perturbation strength. They must not be
confused with the separate observed-outcome taxonomy such as Safe, Repairable,
Broken, or Harmful: a sampled `hard` row can execute successfully, while a row
from a lower strength class can still produce a broken outcome. Curriculum
progress changes the evidence collected under this four-class mixture; it does
not monotonically replace all samples with the maximum corruption.

### Nested K-DR curriculum

The existing fixed K8/M2 -> K16/M3 -> K32/M4 schedule remains the provisional
outer curriculum. Each K owns an inner DR curriculum that begins from a lower,
non-degenerate corruption distribution and then advances toward harder
corruption. "Lower" must still leave a clear Noisy-to-Clean repair signal; it
does not mean zero or nearly Clean corruption.

At a K transition, the current Transaction is committed first. The DR
curriculum then returns to its lower starting distribution, Actor/std remain
frozen, and the same Critic is recalibrated for the new executable-evidence
horizon. Once the Critic is calibrated, Actor ramp and joint optimization
resume while DR advances again inside the new K stage. This resets curriculum
difficulty, not the Actor, Critic, optimizer identity, or previously learned
policy.

### Evidence needed for the final coupling rule

The first campaign records, without feeding back into PPO or the sampler,
whether the same sealed scenario and Repair attempts keep their relative
ordering when evaluated over K8, K16, and K32. Delayed sustained lean,
unplanned support changes, phase-ZMP recovery, survival, and Intent recovery
are part of that longer-horizon outcome.

Cross-horizon ordering remains useful for diagnosing whether the chosen K
stages expose delayed consequences, but it no longer decides whether the next
K must inherit the previous stage's high DR distribution. The lower starting
distribution and its advancement rate remain empirical engineering parameters.
If all observed strengths preserve ordering, 2.381 remains the latest reference
boundary rather than an assumed mastery target.

This cross-horizon fact is diagnostic-only in the first campaign. The proposal
does not introduce an online adaptive K controller, another optimizer, a new
Gain, winner-only replay, or a sampler feedback loop.

## Preserved Boundaries

- K remains the one-action executable-evidence horizon; it is not actor future
  context or a count of PPO rows.
- DR remains corruption severity; it does not alter the full-6D action space.
- Segment Replay keeps every valid attempt and provides grouped ordering; it
  does not choose K or DR by argmax.
- each K transition still re-enters Critic-only recalibration before Actor ramp;
- Clean, Noisy, Repair, H=2, exact M, sealed transaction, grouped PPO, HSL,
  checkpoint identity, and deployment authority remain unchanged;
- no historical episode-length controller is restored.

## Rejected Readings

- increase DR until the policy converges at 2.381, then increase K;
- require a pre-training `g_K` calibration to complete before Critic training;
- increase K and DR difficulty simultaneously;
- carry the previous K stage's high DR distribution directly into Critic
  recalibration at the next K;
- treat frozen-GMT survival as sufficient evidence for learned Repair ranking;
- add an online adaptive controller before the first campaign supplies data.

## Confirmed Human Decisions

1. The first campaign uses 2.381 only as a fixed reference ceiling and retains
 the four-class broad DR mixture.
2. K8/M2 -> K16/M3 -> K32/M4 remains the outer curriculum; every K restarts an
 independent lower-to-higher DR curriculum and same-Critic recalibration.
3. Cross-horizon attempt-order preservation is diagnostic-only in the first
 campaign and cannot feed the sampler or determine an online transition.

## Stop Condition

Activation completed. Source work remains blocked until the rebased Engineering
Plan and affected Module Test Cards are ready. Phase B/live training remains a
separate material boundary. Stop if implementation requires Clean actor input,
a second policy/Critic, winner-only replay, sampler feedback from Gain/PPO, or
revival of the retired global episode-length controller.
