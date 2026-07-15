contract_id: FRS-EVAL-v001
status: superseded
effective_date: 2026-07-13
updated_date: 2026-07-13
supersedes: none
superseded_by: FRS-EVAL-v002
scope: Segment Replay periodic and offline sequence evaluation

# Segment Evaluation Contract

## Evaluation Boundary

Evaluation measures whether the current full-6D repair policy improves frozen
GMT execution relative to the paired Noisy/GMT baseline. Perturbation family
describes the corruption source; it does not reduce the policy output to a
family-specific action mask.

Evaluation data must remain isolated from policy updates and replay-priority
state changes.

## Periodic Evaluation

Each periodic evaluation event must independently:

```text
sample evaluation segments
-> build an evaluation trial plan
-> reset or preroll the selected segments
-> execute paired Noisy/Repaired rollouts
-> aggregate and print evaluation evidence
```

It must not silently reuse a training capture or repeat one stale batch at every
checkpoint. Output must identify motion IDs, start frames, perturbation
families, strengths, horizons, and trial roles.

Sampler RNG/state and replay priority must be unchanged by evaluation unless a
separate, explicitly named review protocol is active.

## Offline Sequence Evaluation

- Sample the requested number of distinct motion identities when available.
- Establish the intended dynamic start state by reset or frame-zero preroll.
- Apply the declared perturbation protocol.
- Begin scoring only at the declared evaluation boundary.
- Preserve paired Noisy/Repaired comparability.
- Aggregate per-motion results before the global summary.
- Report style quality, physics quality, repair cost, and their paired gains
  using `../reward/FRS-GAIN-v001-style-physics-repair.md`.
- Print raw MPJPE, root-orientation, velocity, acceleration, success/fall,
  survival, ZMP/support, contact, Delta SE norm, and temporal repair change.
- Report unavailable quantities as `UNCONFIRMED`, not zero.

For quartet layouts, row identity and offsets must come from the attached trial
plan rather than raw `num_envs` assumptions.

## Interpretation

Positive short-window total gain does not prove long-sequence stability. Style
gain, physics gain, and repair cost remain separately visible. K coverage must
remain visible so delayed regret is not hidden by one fixed short horizon.

## Acceptance Gates

1. Implementation gate: sampling, batch construction, reset, paired scoring,
   aggregation, and state isolation pass offline contract tests.
2. Integration gate: the actual periodic/offline entrypoint invokes that route
   with fresh identities and prints the required metadata.
3. Runtime gate: repeated evaluations show non-stale metadata and valid paired
   metrics in a live environment.

Implementation without entrypoint connectivity is
`implemented-not-integrated`; connectivity without runtime evidence remains
`integration-unconfirmed`.
