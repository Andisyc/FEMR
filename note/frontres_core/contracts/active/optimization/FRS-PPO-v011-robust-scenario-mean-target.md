---
contract_id: FRS-PPO-v011
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-PPO-v010
scope: Grouped B8/M4 symlog PPO with replay-owned robust Scenario mean Critic targets
---
# Grouped PPO With Robust Scenario Mean Targets

## Design delta

PPO-v010 used the current transaction's M4 mean as each Scenario Critic target.
PPO-v011 accepts one detached target per Scenario from the FRS-METHOD-v024
Replay preview. The Critic still predicts expected symlog utility. Actor credit,
grouped mass and optimizer authority are unchanged.

## Scalar signal and B8 reduction

```text
utility_sm       = sign(G_total_sm) * log1p(abs(G_total_sm))
value_target_s   = robust_mean(compatible committed utilities plus current M4)
advantage_sm     = utility_sm - V_old(s)
```

The target carrier contains exactly eight finite values keyed to the eight
Scenario source identities. PPO expands each Scenario target to its four rows
only after proving Scenario/state/old-value identity. It rejects a missing,
duplicate, reordered or non-finite target. It does not recompute Replay
statistics or inspect policy-window state.

The Critic value normalizer previews its one update from the same eight robust
targets. Actor advantages and all PPO ratios remain current-policy attempt
evidence; historical utilities never enter Actor loss.

## Optimizer authority

One Adam retains disjoint Actor and Critic groups. TRAIN-v023 installs Actor LR
`3e-7 -> 1e-6`, Critic LR stays `1e-5`, Actor loss weight stays one, and Actor
and Critic gradients are separately clipped at 0.5 before exactly one optimizer
step. Replay priority never changes optimizer mass.

## Required evidence

- eight keyed robust targets reach Critic loss and normalizer in the same order;
- current per-attempt utilities still determine all 32 Actor advantages;
- ordinary first-visit M4, accumulated compatible visits and incompatible reset
  have independent exact answers;
- row permutation preserving Scenario keys preserves loss and targets;
- failed transaction advances neither optimizer nor Replay window;
- checkpoint-v18 restores target/window, normalizer and LR identity.
