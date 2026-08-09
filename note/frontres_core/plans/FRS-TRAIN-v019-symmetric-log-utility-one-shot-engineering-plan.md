# FRS-TRAIN-v019 Symmetric-Log Utility One-Shot Plan

```yaml
plan_id: FRS-TRAIN-v019-symmetric-log-utility-one-shot
status: phase-b-ready
date: 2026-08-10
owners: [workflow-governance, frontres-return-utility, frontres-stage3]
human_authority: user-approved fixed signed-log Actor/Critic utility through fresh training start
```

## Decision And Change Contract

Requested behavior: preserve the complete raw FRS-GAIN-v007/v008
`G_total` evidence, then map each valid Repair attempt independently through

```text
U(G) = sign(G) * log1p(abs(G)), G0 = 1
```

before any exact-M reduction. The Critic predicts the exact-M mean of `U(G)`
for one sealed state. Each Actor row uses `U(G_m) - V_old(s)`. This is an
explicit robust expected-utility objective, not a policy-invariant rescaling.

Preserved behavior: Actor/GMT/Critic inputs remain 158D/770D/449D; the Actor
still emits one direct world-frame full-6D Repair; FRS-GAIN raw formulas,
M=4, K/DR, phase counts, beta, fixed Actor/Critic LRs, grouped equal mass,
separate clipping, exact-one Adam, and frozen GMT/simulator semantics remain
unchanged. Raw Gain components and hard Physics diagnostics remain visible.

Single owner: `frontres_return_utility.py` owns the fixed transform identity,
scale, finite validation, and tensor mapping. Segment storage calls that owner
only to form the carried pre-normalization Actor advantage while retaining raw
returns. PPO calls the same owner, verifies the carried advantage, forms the
utility-space Segment target, and remains the only loss/update owner.

Public input/output: a detached finite tensor of raw per-attempt `G_total`
values maps to a same-shape detached finite utility tensor. Zero maps to zero,
sign and strict finite ordering are preserved, and row permutation is
covariant. Missing/non-finite evidence fails closed; there is no clip, clamp,
adaptive `G0`, candidate-derived scale, or inverse transform in training.

Dependency direction: Gain producer -> raw return carrier -> return-utility
owner -> PPO target/advantage -> disjoint Actor/Critic gradients -> exact-one
commit. Gain, sampler, simulator, evaluation metrics, and checkpoint transport
may consume the immutable identity but may not recompute or tune the transform.

State and persistence: the transform has no mutable state. The existing
non-amplifying Critic loss-scale state is retained, but it now observes
utility-space Segment targets. Checkpoint-v14 binds the transform identity and
G0=1, and rejects checkpoint-v13 or raw-target identities before mutation.

Legacy boundary: raw-target METHOD-v019 / PPO-v007 / TRAIN-v018 /
checkpoint-v13 is historical only. HSL-v2 remains an Actor-only initializer;
no old Stage-3 Critic, optimizer, normalizer, sampler, or receipt is migrated.

Forbidden behavior: modifying FRS-GAIN arithmetic; transforming after the M4
mean; Critic-only transform; raw-Gain Actor advantage; dropping raw telemetry;
using `abs`, unsigned `log`, clipping, dynamic scale, winner selection, or a
second objective/optimizer; modifying MOSAIC, GMT, networks, observations,
action, K/M/DR, beta, or LRs.

## Semantic Migration Table

| Object | New owner and consumer | Retired interpretation | Proof |
| --- | --- | --- | --- |
| raw `G_total_m` | GAIN-v008 -> telemetry and utility owner | raw value used directly by PPO | TEST-13 and TEST-18 |
| `U(G_total_m)` | fixed utility owner -> storage/PPO | no active utility transform | TEST-14 and TEST-15 |
| Critic target | PPO-v008 `mean_m(U(G_m))` | `mean_m(G_m)` | TEST-15 |
| Actor advantage | PPO-v008 `U(G_m)-V_old(s)` | `G_m-V_old(s)` | TEST-15 |
| persistence | TRAIN-v019 checkpoint-v14 | checkpoint-v13 | TEST-16 |
| evaluation | EVAL-v004 compares `V(s)` with utility target and reports raw mean separately | raw mean treated as Critic target | TEST-17 |

## Coarse Execution

1. Activate METHOD-v020 / GAIN-v008 / PPO-v008 / TRAIN-v019 and update the
   Design Inspector, register, registry, Architecture and Module Test Cards.
2. Add failing semantic tests, then implement the single pure utility owner,
   storage/PPO consumption, telemetry, checkpoint-v14 and EVAL-v004 alignment.
3. Run focused tests, aggregate contracts, compilation, structured Atlas
   validation, construction/final review, official offline transaction and
   strict persistence round trip.
4. Push the coherent scope, pull it on the server, execute one bounded official
   critic-only transaction, inspect existing structured telemetry, and start a
   fresh HSL-v2 K8/M4 cold-start run only after the long-training gate passes.

## Confirmed Module Test Cards

- `TEST-13 Repair Gain`: raw FRS-GAIN evidence and component ordering remain
  byte-for-byte outside the new utility owner; non-finite input rejects.
- `TEST-14 Segment Storage`: hand-calculated K-return remains raw while stored
  advantage equals per-row `symlog(return)-V_old`; permutation preserves rows.
- `TEST-15 Segment PPO`: asymmetric two-Segment x M4 values prove per-attempt
  transform-before-mean, shared value target, Actor advantage, strict ordering,
  tail compression, grouped mass, separate gradients and sensitivity to the
  wrong `symlog(mean(raw))` construction.
- `TEST-16 Checkpointing`: checkpoint-v14 round-trips transform ID/G0, model,
  optimizer and value-scale state; v13 or wrong transform rejects pre-mutation.
- `TEST-17 Evaluation`: EVAL-v004 reports raw M4 mean separately and calibrates
  the Critic against `mean_m(symlog(G_m))` without writes.
- `TEST-18 Runtime Diagnostics`: formal telemetry exposes raw returns, utility
  returns, utility targets/advantages, transform identity, finite values,
  split LR, exact-one receipt and committed-only normalizer state.

Independent oracles use hand-computed `log1p` values, an explicit wrong
transform-after-mean counterexample, row permutations, model/state snapshots,
optimizer step counts and strict checkpoint identity sentinels.

## Stop Conditions

Stop before live work on any missing raw evidence, non-finite transform,
sign/order loss, transform-after-mean result, raw-target Actor/Critic consumer,
identity drift, old-checkpoint acceptance, partial mutation, shared clipping,
more than one optimizer step, or missing official-route/persistence receipt.
Stop before long training if the bounded critic-only transaction lacks finite
raw/utility telemetry, a nonzero Critic delta, frozen Actor/std, exact-one step,
or checkpoint-v14 fresh readback. A sentinel remains lifecycle evidence only.
