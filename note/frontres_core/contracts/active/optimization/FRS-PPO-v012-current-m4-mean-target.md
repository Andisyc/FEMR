---
contract_id: FRS-PPO-v012
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-PPO-v011
scope: Grouped B8/M4 symlog PPO with current-transaction Scenario mean Critic targets
---

# Grouped PPO With Current M4 Mean Targets

## Scalar signal

```text
utility_sm     = sign(G_total_sm) * log1p(abs(G_total_sm))
value_target_s = mean_m=1..4(utility_sm)
advantage_sm   = utility_sm - V_old(s)
```

The target carrier contains exactly eight detached finite values keyed to
Scenario source identities 0..7. PPO expands each target to the four matching
rows only after proving Scenario/state/old-value identity. No previous visit's
utility may enter this carrier. Actor ratios and advantages use only the 32
current-policy attempts.

## Optimizer authority

One Adam retains disjoint Actor and Critic groups. TRAIN-v024 installs Actor LR
`3e-7 -> 1e-6`; Critic LR remains `1e-5`; Actor loss weight remains one.
Actor and Critic gradients are clipped separately at 0.5 before exactly one
optimizer step. Replay priority never changes loss mass.

## Required evidence

- eight row-aligned current-M4 means reach Critic loss and normalizer;
- all 32 Actor advantages remain current `utility - V_old`;
- changing persisted Replay history cannot change a current target;
- row permutation preserving Scenario identity preserves targets and loss;
- missing, reordered or non-finite targets fail closed;
- failed transaction advances neither optimizer nor Replay;
- checkpoint-v19 restores Replay-v5, normalizer and LR identity and rejects v18.

