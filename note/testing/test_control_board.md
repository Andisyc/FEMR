# FEMR Current Test Control Board

Updated: 2026-07-20

This is a current-state view, not a chronological test log. Dated command
evidence belongs in evidence ledgers.

## Evidence Levels

```text
S0 Static             syntax, import, path, symbol, config, schema
S1 Module Semantic    deterministic value, sign, shape, mask, invariance
S2 Offline Connection fake runner/env/storage/model formal-route connection
S3 Persistence        checkpoint, resume, normalizer, eval/export state
S4 Live Sentinel      minimal real IsaacLab runtime evidence
```

Each mechanism has separate implementation and integration gates. Runtime-only
claims additionally require S4.

## Current Baseline

| Surface | Current evidence | Status | Limitation |
| --- | --- | --- | --- |
| Historical v013 Segment aggregate suite | 59/59 contract markers, 2026-07-19 | historical offline evidence for fixed-65D-tape lifecycle, command/reset/H connectivity, and Atlas anchors | Does not prove v015 local scenario, q29 intent, or frozen-FEMR Clean-continuation semantics. |
| Historical Survival Gain v002 suite | Gain owner, connectivity, live probe/training/sequence/diagnostic/sampler contracts, 2026-07-17 | historical | Does not prove v015 q29 intent, two-role, or frozen-FEMR Clean-continuation semantics. |
| Architecture viewer | JSON valid; viewer imports; 62 owner paths exist; Runtime Audit Atlas has 22 cards and Quality Audit Atlas has 8 source-linked cards | covered S0 | Does not prove runtime routing or policy quality. |
| Full-6D/no active mask | dedicated static contract plus rollout/PPO tuple tests | covered S0-S2 | S4 full-6D log proof remains. |
| K curriculum | 8/16/32/64 implementation, explicit Stage 3 max horizon, and formal-route connectivity | covered S1-S2 | Live horizon distribution remains S4. |
| Segment PPO | clipped surrogate, exact KL, raw-Gaussian/tanh log-prob identity, ratio source, scale-only advantage, rollback | covered S1-S2 | Gain consumer alignment remains S4; long-run learning quality is deferred until after training. |
| v015 Intent Segment Gain | q29-intent/physics/cost owner and local two-role consumers | S1 candidate chain complete, `E-FI-10`--`E-FI-13` | Real evaluation, sampler state, formal PPO/update, checkpoint/resume, and live consumers remain unconnected. |
| v015 grouped candidate adapter | sealed candidate return -> local metadata -> grouped PPO batch | S1 complete, `E-FI-13` | Candidate-only batch/loss evidence; formal runner, optimizer, checkpoint/resume, and live transaction remain unconnected. |
| v015 deployment composition | ordinary reference `.npz` -> planned selection-time fixed carrier -> paired frozen-GMT baseline vs per-frame FEMR+GMT -> report | interfaces S1/S2 at `E-FI-28`--`E-FI-31`; test-path rebase `E-FI-32` | Current code still requires an external pre-materialized file and missing trained checkpoint. CLI is implemented-not-runnable; G1--G6 precede S4. |

## Historical v002 Gain Change Matrix

The matrix below remains evidence history. The v015 matrix is the current
acceptance surface and supersedes it for all new implementation work.

| Object | Owner | Required tiers | Required T kinds | Current status |
| --- | --- | --- | --- | --- |
| Style components | `frontres_gain.py` | S1 | T-value, T-sign, T-normalize, T-clean-target | S1 complete; S4 population open |
| Physics components | `frontres_gain.py` + balance/contact helpers | S1/S2 | T-value, T-done, T-contact, T-zmp, T-pair | S1/S2 offline complete; contact is height proxy; S4 population open |
| Repair regularizer | `frontres_gain.py` + live probe | S1/S2 | T-full6, T-executed-action, T-temporal, T-K-mask, T-clean-noop | offline complete; live population remains S4 |
| Paired K accumulation | live probe + segment reward | S1/S2 | T-pair, T-role, T-done, T-mixed-K, T-permute, T-component-total | offline complete; live population remains S4 |
| Rollout transaction identity | capture -> Gain -> storage -> update-loop diagnostics | S1/S2 | T-connect, T-role, T-meta, T-oracle | local confirmed; S4 formal equality open |
| PPO return | segment storage/PPO | S2 | T-connect, T-sign, T-invalid-row, T-same-formula, T-no-legacy-fallback | formal route offline complete; live training remains open |
| Single Gain owner / sampler evidence | `frontres_gain.py` + live sampler/sampler | S2 | T-single-owner, T-formal-gain-source, T-no-legacy-score, T-evidence-isolation, T-priority, T-state | 6A/6B plus all eval consumers offline covered; S4 remains; E14/E15/E16 |
| Periodic eval | live training eval owner | S2/S4 | T-fresh-sample, T-state, T-same-formula, T-owner-isolation, T-live | S2 offline covered by E15; S4 live population open |
| Sequence eval | sequence eval owner | S2/S4 | T-preroll, T-motion, T-K, T-same-formula, T-owner-isolation, T-live | S2 offline covered by E16; S4 live population open |
| Checkpoint/resume | `frontres_checkpointing.py` formal runner owner | S3/S4 | T-state, T-version, T-scale, T-missing-identity, T-live | live-confirmed by E69: model_220 full-resume restored training state and saved model_221 after 4/4 updates |
| Full-weight joint PPO | warmup owner -> Segment PPO -> checkpoint owner | S4 | T-phase, T-gradient-boundary, T-trust, T-persist | live-confirmed by E70: joint actor_weight=1.0, valid=13/14/16/16, accepted updates, frozen GMT, complete model_701 |
| Diagnostics | Segment diagnostics/logging | S1/S2/S4 | T-unconfirmed, T-nonstale, T-decompose, T-legacy-isolation, T-live | canonical train-effect route and isolation offline-covered; raw ZMP/contact and S4 live evidence open |

## v015 Current Migration Matrix

| Object | Owner | Required tiers | Required T kinds | Current status |
| --- | --- | --- | --- | --- |
| Local root-artifact / q29 invariant | perturbation + scenario owner | S1 | T-invariant, T-differential, T-hash | S1 complete; `E-FI-2` |
| Future intent H layout | command/runtime/actor/checkpoint | S1/S2/S3 | T-provenance, T-shape, T-clean-isolation, T-resume | S1/S2 fake connectivity plus CPU-fake S3 persistence complete; `E-FI-3`, `E-FI-7`, `E-FI-15`; generic/live resume pending |
| Two-role one-action K lifecycle | pair layout/reset/command/live probe | S1/S2/S4 | T-role, T-state, T-action-count, T-frozen, T-continuation | S1/S2 candidate-only fake lifecycle complete; `E-FI-8`, `E-FI-9`; formal/live pending |
| Intent/physics/cost Gain | Gain + storage/eval consumers | S1/S2/S4 | T-value, T-noop, T-root-exclusion, T-single-owner | S1 pure owner, candidate return/priority, and candidate-only diagnostics/isolation complete, `E-FI-10`-`E-FI-12`; real evaluation/formal consumers pending |
| Grouped PPO | PPO plus v015 scenario metadata | S1/S2/S4 | T-value, T-sign, T-permute, T-mass, T-source | v003 reduction and metadata S1 complete; CPU fake-S2 transaction/update plus S3 persistence barrier complete, `E-FI-13`/`E-FI-14`/`E-FI-15`; generic/live pending |
| Formal v015 route | sealed plan, config, runner, update loop, probe, grouped PPO | S0/S2/S3/S4 | T-route, T-order, T-exact-one-update, T-retirement, T-resume, T-diagnostic, T-live | CPU fake S2/S3 complete, `E-FI-14`/`E-FI-15`; generic train, real checkpoint cadence/resume, and live route remain pending |

## Required Gain Test Order

1. S1 component fixtures with hand-computed Style, Physics, and Repair values.
2. S1 paired sign and mixed-K aggregation tests.
3. S1 candidate pseudo route into immutable return/priority evidence (`E-FI-11`); formal storage/PPO/sampler update remains pending.
4. S2 periodic/sequence eval formula-identity tests.
5. S3 persistence for scales/state entering formal checkpoints.
6. S4 env8/short-K live sentinel before formal training.

## Current Training Gate

```text
v015 Step 0 documentation closure: COMPLETE
v015 local scenario, future q29-intent, two-role K lifecycle, pure Gain, candidate return/priority, and diagnostic/isolation S1: COMPLETE
v015 injected formal transaction fake-S2 plus CPU-fake persistence S3: COMPLETE; generic transaction/real-checkpoint/live route: BLOCKED by S4
long training: BLOCKED pending reviewed v015 Step 5 live identity sentinel
FRS-GAIN-v003 is active semantics; current v002 code/evidence does not
substitute for v015 future-intent, lifecycle, or grouped-loss evidence
```

Unblock only when the current checklist records implementation, formal-route
integration, and live evidence separately.
