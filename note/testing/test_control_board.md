# FEMR Current Test Control Board

Updated: 2026-07-17

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
| Segment aggregate suite | 44/44 contract markers, 2026-07-15 | covered S0-S3/S4-named offline contracts | Does not prove real simulator quality. |
| Survival Gain v002 offline suite | Gain owner, connectivity, live probe/training/sequence/diagnostic/sampler contracts, 2026-07-16 | passed | Same rollout/source policy-row batch is confirmed by E63; formal numeric alignment of K, done mask, components, and step sum remains open; long-run quality is post-training evidence. |
| Architecture viewer | JSON valid; viewer imports; 62 owner paths exist; Runtime Audit Atlas has 20 owner cards | covered S0 | Does not prove runtime routing. |
| Full-6D/no active mask | dedicated static contract plus rollout/PPO tuple tests | covered S0-S2 | S4 full-6D log proof remains. |
| K curriculum | 8/16/32/64 implementation, explicit Stage 3 max horizon, and formal-route connectivity | covered S1-S2 | Live horizon distribution remains S4. |
| Segment PPO | clipped surrogate, exact KL, raw-Gaussian/tanh log-prob identity, ratio source, scale-only advantage, rollback | covered S1-S2 | Gain consumer alignment remains S4; long-run learning quality is deferred until after training. |
| Current Segment Gain | shared Style/Physics/Repair owner reaches formal policy-row returns and periodic/sequence eval | covered S1-S2 | Real root/ZMP/contact population remains open. |

## Active Gain Change Matrix

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
| Checkpoint/resume | `frontres_checkpointing.py` formal runner owner | S3 | T-state, T-version, T-scale, T-missing-identity | offline covered by E17; actual server artifact resume remains open |
| Diagnostics | Segment diagnostics/logging | S1/S2/S4 | T-unconfirmed, T-nonstale, T-decompose, T-legacy-isolation, T-live | canonical train-effect route and isolation offline-covered; raw ZMP/contact and S4 live evidence open |

## Required Gain Test Order

1. S1 component fixtures with hand-computed Style, Physics, and Repair values.
2. S1 paired sign and mixed-K aggregation tests.
3. S2 live-probe pseudo route into storage/PPO and sampler evidence.
4. S2 periodic/sequence eval formula-identity tests.
5. S3 persistence for scales/state entering formal checkpoints.
6. S4 env8/short-K live sentinel before formal training.

## Current Training Gate

```text
formal policy-row Gain route: implemented and offline-connected
active FRS-GAIN-v002: Physics/Repair/sampler/eval formal route is
offline-complete; tiny official-route S4 audit is authorized, while long
full-Gain training remains BLOCKED pending reviewed S4 evidence
single active Gain owner: confirmed design; sampler 6A/6B migration offline-complete
sampler priority/state: 6A/6B offline-covered; periodic and sequence consumers migrated; S2 cross-consumer acceptance closed by E16
```

Unblock only when the current checklist records implementation, formal-route
integration, and live evidence separately.
