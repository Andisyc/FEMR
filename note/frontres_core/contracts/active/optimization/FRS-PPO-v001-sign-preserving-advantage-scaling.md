contract_id: FRS-PPO-v001
status: active
effective_date: 2026-07-09
updated_date: 2026-07-13
supersedes: none
scope: Stage 3 Segment PPO advantage scaling semantics

# FrontRES Segment PPO Advantage Scaling Contract

## Design Delta

Old design:

```text
Segment PPO could use standard PPO advantage normalization:
A_norm = (A - mean(A)) / std(A)
```

New design:

```text
Segment PPO uses sign-preserving scale-only advantage scaling by default:
A_scaled = A / rms(A)
```

Changed semantic object:

```text
segment advantage
-> PPO surrogate weight
-> actor mean update direction
-> post-update KL / trust-region behavior
```

Forbidden old assumption:

```text
positive gain rows may be turned negative merely because they are below the
current batch mean
```

## Method Meaning

For direct Delta SE HRL, a positive Segment Replay gain means:

```text
Repaired rollout is better than Noisy rollout under the executable score.
```

This is a no-regret statement, not only a rank inside the current mini-batch.
If all valid rows have positive gain, all valid sampled actions should remain
encouraged.  Their strength may differ, but their sign must not flip.

Standard PPO mean-centering is valid for ordinary policy-gradient ranking, but
it changes this FEMR meaning:

```text
[0.01, 0.03, 0.06]
-> standard mean-centering
-> some positive gains become negative training weights
```

Therefore the default Segment Replay HRL route must not use standard
mean-centering as a silent stabilizer.

## Default Rule

Default live Segment PPO:

```text
advantage_normalization = scale_only
```

Allowed modes:

```text
none
  Use rollout advantage exactly as stored.  Cleanest semantics, weakest scale
  control.

scale_only
  Divide by RMS magnitude.  Preserves sign and relative strength while reducing
  batch-scale sensitivity.

standard
  Subtract mean and divide by standard deviation.  Allowed only as an explicit
  PPO ablation or debugging comparison, not the default FrontRES Segment HRL
  path.
```

## Invariants

- Positive advantages remain positive after default scaling.
- Negative advantages remain negative after default scaling.
- Zero advantages remain zero.
- Scaling never changes row order by absolute magnitude.
- Old policy tensors, returns, and advantages remain detached from gradient.
- KL and rollback still use old/new policy distribution statistics, not the
  advantage scaling rule.

## Code Ownership

```text
source/rsl_rl/rsl_rl/algorithms/frontres_segment_ppo.py
  owns advantage scaling inside compute_frontres_segment_ppo_loss.

source/rsl_rl/rsl_rl/runners/frontres_segment_live_probe.py
  owns live single-update config selection and must pass scale_only by default.

source/rsl_rl/rsl_rl/frontres/frontres_segment_storage.py
  owns stored raw rollout advantages before algorithm-side scaling.
```

## Required Tests

S1 / T-scale:

```text
frontres_segment_algorithm_contract.py
  all-positive gains remain positive under scale_only;
  the same all-positive gains include a negative row under standard mode.
```

S2 / T-connect:

```text
frontres_segment_live_single_update_contract.py
  live single-update default uses scale_only independent of base PPO
  normalize_advantage_per_mini_batch.
```

S4 / T-live:

```text
short live Stage 3 sentinel should print advantage_scale and sign_flip_count
before formal training if advantage instability remains suspicious.
```

## Decision Boundary

If training still has KL explosion after scale-only advantage scaling, do not
reclassify it as an advantage-normalization bug by default.  Continue the
runtime-probing chain:

```text
advantage distribution
-> policy std/sigma
-> one-step mean delta
-> post-update KL rollback
-> observation normalizer / Stage2 checkpoint compatibility
```
