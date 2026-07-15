# FEMR Current Impact Rules

Updated: 2026-07-13

Report every change as:

```text
changed owner -> expanded semantic objects -> required S tiers/T kinds
-> observed evidence -> unconfirmed boundary
```

## Gain Component Or Scale Change

Expand to: Clean/Noisy/Repaired pairing, mixed K, live probe, Segment return,
sampler priority, periodic eval, sequence eval, diagnostics, and checkpoint if
state/scales persist.

Required: S1 hand-computed components and signs; S2 formula identity across
training/sampler/eval; S3 persistence when applicable; S4 populated live data.

## Observation Or Normalizer Change

Expand to: observation producers, 870D layout, policy, Stage 2/3 checkpoint,
resume, eval, export, and play.

Required: S1 layout/value; S2 actor connectivity; S3 save/load/sinks; S4 when
real observation production changes.

## Full-6D Action Or Distribution Change

Expand to: policy mean/sigma, bounded action transform, rollout write, old
log-prob/stats, storage, PPO, checkpoint, eval, diagnostics.

Required T kinds: full6, same-source, bounded-logprob, exact-KL, detach,
permutation, ratio decomposition, small-sigma sensitivity, post-mean-delta.

## Segment Sampler Or Trial Change

Expand to: source proportions, priority/state, row budget, trial roles, quartet
construction, reset metadata, PPO validity, evidence isolation, checkpoint,
diagnostics.

Required: S1 priority/state/role; S2 formal sampling and PPO boundary; S3
persistence; S4 real distribution evidence.

## K-Step Change

Expand to: sampler assignment, quartet replication, reset, max-K stepping,
per-row temporal masks, returns, evidence, eval, diagnostics.

Required: S1 mixed-K math; S2 formal-route connectivity; S4 live horizon
distribution. A planner-only test proves implementation, not integration.

## Reset Or Reference-Window Change

Expand to: cache state, dataset payload, reset request/result, command hook,
preroll, row-domain masks, live probe, sequence eval.

Required: S1 schema/value; S2 fake command/reset connection; S3 cache/checkpoint
compatibility; S4 real dynamic-state restoration.

## PPO Update Change

Expand to: storage tuple, advantages, old stats, ratio/KL, optimizer order,
adaptive LR/rollback, diagnostics, sampler isolation.

Required T kinds: clip, KL-exact, detach, permutation, invalid-row isolation,
advantage-sign, update-order, state-change-once, trust-region evidence.

## Checkpoint Change

Expand to: cold start, resume, Stage 2 -> Stage 3 migration, policy head/sigma,
normalizer, optimizer, sampler, eval/export/play sinks.

Required: S3 round-trip and migration; one-batch forward; S4 only when a real
artifact/runtime loader changes.

## Evaluation Change

Expand to: independent sampling, sampler/RNG restoration, reset/preroll,
paired roles, shared Gain components, aggregation, metadata, `UNCONFIRMED`.

Required: S1 formatting/aggregation; S2 fresh-route and state isolation; S4
non-stale motion/perturbation/component evidence.
