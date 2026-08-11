---
contract_id: FRS-PPO-v010
status: active
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FRS-PPO-v009
scope: Grouped B8/M4 symlog PPO with actual Actor optimizer-group LR curriculum
---

# Grouped PPO With Actual Actor-LR Curriculum

## Design delta

FRS-PPO-v009 multiplied the Actor loss by a small warmup weight. Under Adam,
global gradient scaling is largely normalized away, and later norm clipping can
erase it further. FRS-PPO-v010 fixes Actor loss weight at one and schedules the
actual Actor parameter-group LR. Critic target, Actor advantage, grouped mass,
separate clipping and exact-one Adam remain unchanged.

## Scalar signal and B8 reduction

For each Scenario `s=1..8`, attempts `m=1..4` use:

```text
utility_sm      = sign(G_total_sm) * log1p(abs(G_total_sm))
value_target_s  = mean_m utility_sm
advantage_sm    = utility_sm - V_old(s)
```

The Critic therefore receives eight distinct state targets per transaction;
M4 reduces action-outcome noise within each state. Actor mass remains equal by
motion, Scenario and attempt. Replay score and source never change loss mass.

## Optimizer authority

One Adam retains two named, disjoint groups. Before backward/step, the sealed
TRAIN-v022 phase installs Actor LR `3e-7 -> 1e-6` on the Actor group and verifies
Critic LR `1e-5`. Actor loss weight and entropy weight multiplier remain one.
Actor and Critic gradients are clipped separately at 0.5, followed by exactly
one optimizer call. Failure before commit restores the pre-transaction group LR
and changes no optimizer, replay or curriculum state.

## Required evidence

- eight unique Critic states, eight exact-M targets and 32 Actor rows;
- phase LR reaches the actual named Actor group before Adam step;
- first low-DR transaction changes Actor/std slowly but nonzero and updates Critic;
- grouped permutation invariance and exact-one update;
- failed/invalid transaction has zero parameter, optimizer, replay and
  curriculum mutation;
- checkpoint-v17 restores both group LRs and their next scheduled transition.

