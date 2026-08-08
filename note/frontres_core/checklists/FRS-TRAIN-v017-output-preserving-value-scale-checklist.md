# FRS-TRAIN-v017 Output-Preserving Value Scale Checklist

Status: offline-ready; bounded official transaction pending
Date: 2026-08-08
Plan: `../plans/FRS-TRAIN-v017-output-preserving-value-scale-one-shot-engineering-plan.md`

## Governance

- [x] Preserve FRS-GAIN-v007, raw `G_total`, raw `V(s)`, Actor credit, networks,
  fixed split LR, K/M/DR, GMT and simulator semantics.
- [x] Assign scale preview to Segment PPO, atomic commit to the formal
  transaction, and exact persistence to checkpoint-v12.
- [x] Activate FRS-PPO-v007 / FRS-TRAIN-v017 / checkpoint-v12 and archive the
  superseded v006/v016 authorities.

## Module Gates

- [x] TEST-15 proves hand-computed moments/scale, raw-output and Actor
  invariance, permutation invariance, floor behavior and invalid-state reject.
- [x] TEST-16 proves checkpoint-v12 exact roundtrip and pre-mutation rejection
  of v11, missing, non-finite or iteration-inconsistent state.
- [x] TEST-18 proves committed-only state transition and owner-produced
  telemetry without recomputation or feedback.
- [x] TEST-02 proves the official composition fixes identity, decay `0.9` and
  scale floor `1.0` and rejects partial/drifted identities.

## Verification Gates

- [x] Python compile and focused v017 tests pass.
- [x] Impacted interface, entrypoint, checkpoint-quality and formal-audit
  regressions pass.
- [x] JSON/contract references, diff checks and construction review pass.
- [x] Formal Runtime Audit Phase A confirms config -> loss -> transaction ->
  checkpoint -> telemetry connectivity.
- [ ] One bounded official transaction confirms runtime identity, finite scale,
  exact-one update and checkpoint-v12 roundtrip.
- [ ] Fresh long training is authorized only after the bounded gate; no v016 or
  sentinel checkpoint resume.

## Stop Conditions

Stop before long training if raw targets/values or Actor facts differ, the
scale is non-finite or below one, statistics advance without a committed
receipt, checkpoint count differs from committed iteration, any old identity
is accepted, or the official route lacks the new diagnostics.
