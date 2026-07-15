# K-Step Curriculum Formal Integration

## Problem

The Segment Replay sampler already plans horizons `8/16/32/64`, but formal
Stage 3 training clamps every sampled row to `frontres_segment_k=8`. The local
curriculum is therefore implemented-only rather than integrated-live.

## Design Delta

Old design:

```text
frontres_segment_k=8
-> sampler max horizon falls back to 8
-> every reset, rollout, return, and evidence row uses K=8
```

New design:

```text
frontres_segment_k=8                # default/unknown-state horizon
frontres_segment_max_horizon_k=64   # formal curriculum ceiling
-> sampler state chooses 8/16/32/64
-> quartet counterparts share the same per-segment K
-> live env steps to max(K) in the current batch
-> each row accumulates reward/done only through its own K
-> storage builds a per-row done-masked K-step return
-> sampler evidence records the executed K
```

Changed semantic object: `horizon_k`, from one scalar training constant to a
per-row rollout and return contract.

Forbidden old assumptions:

- `frontres_segment_k` is both the initial horizon and the maximum horizon.
- a single scalar horizon can represent a mixed Segment Replay batch.
- short trial metadata may be padded with `K=8` for counterfactual branches.

## Scope

- Add the formal maximum-horizon config with default `64`.
- Preserve sampler state logic for `8/16/32/64`.
- Propagate per-row K through trial metadata, quartet expansion, rollout,
  reward/done accumulation, storage returns, evidence, and diagnostics.

## Non-Scope

- PPO objective, advantage scaling, reward formula, sampler priority formula.
- Active task dimensions or specialist-mode semantics.
- Offline sequence-eval episode length.

## Core Parameter Path

```text
RslRlFrontRESAlgorithmCfg.frontres_segment_max_horizon_k
-> FrontRESUnified.frontres_segment_max_horizon_k
-> _resolve_live_max_horizon_k
-> FrontRESSegmentSampler.sample_rollout_rows(...).horizon_k
-> batch.frontres_segment_budget_horizon_k
-> _current_trial_metadata
-> _run_live_rollout_capture
-> FrontRESSegmentLiveRolloutCapture.horizon_k
-> FrontRESSegmentRolloutStorage.compute_returns_and_advantages
-> build_live_sampler_evidence(...).horizon_k
-> sampler state update and production diagnostics
```

## Test Contract

- S1 `T-dist`: sampler states produce `8/16/32/64` under ceiling `64`.
- S1 `T-value/T-mask`: mixed per-row K produces hand-checkable done-masked
  returns.
- S2 `T-connect/T-role`: official algorithm config reaches the live sampler;
  quartet rows inherit matching K; capture and storage preserve mixed K.
- S0: changed Python files compile.

## Stop Condition

Offline tests prove mixed K reaches the formal training connector and affects
per-row returns. Live simulator behavior remains S4-unconfirmed until a short
Stage 3 sentinel prints more than one horizon value.

## Offline Evidence

- S0: changed owner/config/runner files compile with `frontres/bin/python -m py_compile`.
- S1 `T-dist`: sampler contract proves `[8,16,32,64,64,8]` for all states under ceiling 64.
- S1 `T-value/T-mask`: live-probe contract proves horizons `[1,4]` produce returns `[1,4]`.
- S2 `T-connect/T-role`: fake live probe carries `[1,3]` through reset and performs three env steps; quartet metadata expands `[8,32]` to all four paired branches.
- S1/S2 diagnostics: temporal motion metrics ignore frames beyond each row's K.
- Aggregate Segment Replay suite: `contract_count=43 failed_count=0`.

Wiring state: `integrated-offline`. S4 real IsaacLab horizon diversity remains unconfirmed.
