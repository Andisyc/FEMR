# TRAIN-v016 State-Value Module Test Execution

Date: 2026-08-08
Status: `18 passed / 0 partial / 0 blocked`; Formal Runtime Audit Phase A human-confirmed

## Scope

The user confirmed TEST-05, TEST-15, TEST-16 and TEST-18 before implementation.
Their accepted oracle remained fixed during execution. The other fourteen cards
retain their prior passing evidence because v016 does not change their semantic
owner. This ledger records module correctness only; it is not simulator/live,
policy-quality or long-training evidence.

## Changed Cards

| Card | Executable evidence | Observed fact | Limitation |
| --- | --- | --- | --- |
| TEST-05 Observation Layout | `frontres_v016_state_value_observation_contract.py` | `[289 current | 58 sealed tail] = 347`; the same 58D tensor is prepended to Actor input; Actor prefix is 158D and GMT suffix remains exactly 770D; malformed, non-finite and attached tensors reject | No IsaacLab timing evidence |
| TEST-15 Segment PPO | `frontres_v016_state_value_ppo_contract.py` | exact-M mean is computed per `source_index`; per-attempt Actor advantages remain distinct; permutation and repeated local segment IDs are stable; Actor/Critic gradients clip independently at 0.5 | No convergence or calibration-quality claim |
| TEST-16 Checkpointing | `frontres_v016_checkpoint_contract.py` | v11 round-trips Actor, 347D Critic, split Adam, normalizers, curriculum and receipt atomically; v10 cannot resume; strict v10 read-only inspection remains isolated; malformed v11 rejects pre-mutation; HSL-v2 remains Actor-only | No server filesystem or live resume evidence |
| TEST-18 Runtime Diagnostics | `frontres_v016_runtime_telemetry_contract.py` | one complete offline formal update serializes v018/v006/v016, 158/347/770, Segment targets, Actor advantages, separate finite clip facts, exact-one update and v11 receipt; missing/non-finite/misaligned fields reject | Serializer connectivity is offline only |

## Construction Review Findings

- P1 fixed: policy-quality inspection now separates historical read-only v10
  from mutable v11 resume. No legacy optimizer, sampler, curriculum or receipt
  state can initialize TRAIN-v016.
- P1 fixed: Segment value targets group by transaction `source_index`, not the
  motion-local numeric `segment_id`; two sources may legally reuse a local ID.
- Test translation fixes were limited to current public Contract strings,
  package-preserving import fixtures, stale v017/v005/v015 identities and two
  indentation errors. The formal transaction fixture now keeps its toy Actor
  input separate from an exact 347D Critic state. No accepted numeric oracle
  changed.
- The generic owner line-count rejection was removed because active engineering
  discipline requires named responsibility/dependency gates, which still pass:
  the facade remains thin, dependencies are acyclic, and owners import no peer
  private symbols.

## Fresh Commands

The four changed-card tests pass. The Stage-3 pseudo suite passes 13/13 and the
updated deterministic aggregate passes 52/52 with all four v016 contracts in
the active manifest. Python compilation and the focused
entrypoint/config/HSL/interface/grouped-PPO/formal-transaction/read-only-quality
regressions pass. The complete Atlas `npm run check` and `git diff --check` also
pass after the final evidence refresh.

## Next Gate

Formal Runtime Audit Phase A and the Phase B probe plan were human-confirmed on
2026-08-08. The audit-only probes are inserted and pass the 13/13 Stage-3 pseudo
suite, 52/52 aggregate suite, compilation and Atlas checks. The next gate is one
authorized bounded official transaction; long training remains a later
evidence-and-cost gate.
