---
contract_id: FRS-METHOD-v025
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-METHOD-v024
scope: Outer sealed-Scenario selection with fresh current-policy M4 Critic targets
---

# Current-Visit Scenario Replay

## Design correction

Outer Replay selects which sealed Scenario should be executed again. A selected
Scenario is always materialized and rerun by the current frozen `pi_old` for
exact M4 Repair attempts before any update. Replay is not Experience Replay of
old numerical outcomes: utilities from earlier optimizer versions never enter
the current Critic target, Actor advantage, loss, or value normalizer.

The former compatible utility window, policy anchor, Gaussian KL gate, reset
counter and cross-visit robust target are retired. TRAIN-v023 evidence showed
that almost every revisit reset its window, so those fields added lifecycle
state without supplying a stable target.

## Current-visit target and priority

For each of the eight current Scenarios:

```text
u_sm        = sign(G_total_sm) * log1p(abs(G_total_sm))
target_s    = mean_m=1..4(u_sm)
var_s       = sample_variance_m=1..4(u_sm)
SE_s        = sqrt(var_s / 4)
h95_s       = 1.96 * SE_s
E_V_s       = max(abs(V_old(s) - target_s) - h95_s, 0)
E_A_s       = mean_m abs(u_sm - target_s)
```

`target_s` is the ordinary exact-M4 mean from this transaction only. Current
M4 variance and confidence width are diagnostics and uncertainty corrections
for Replay priority; they never alter the Critic target. `E_V` is the latest
current-policy calibration priority and `E_A` is the latest current-policy
Repair-spread priority. Neither priority is averaged across policy versions.

## Outer Replay state and lifecycle

Replay persists only stable Scenario identity, DR class, latest K-local
priorities, lifetime committed visit count, staleness, active/archive
membership, capacity, RNG and last committed receipt. The per-K active pool
remains `64 -> 128 -> 256`. Breadth may expand only in joint optimization after
every active Scenario has at least four committed fresh M4 visits at that K.
There is no policy-compatibility reset; every visit count denotes a real rerun.

Selection, current evidence, eight targets, priority refresh, membership,
capacity, staleness and RNG are staged without mutation. They commit only after
the matching exact-one Adam receipt. Failed, duplicate, partial or mismatched
transactions change neither Replay nor optimizer state.

## Preserved boundaries

- Actor remains the deployable 158D full-6D direct Delta SE(3) policy.
- Critic remains the 449D action-pre `V(s)` for expected symlog utility.
- Raw FRS-GAIN-v008, per-attempt symlog, B8/M4, K8/K16/K32 and DR are unchanged.
- Actor advantage remains current `u_sm - V_old(s)` for all 32 attempts.
- Actor/Critic joint update, LR groups, separate clipping and exact-one Adam are unchanged.
- No old action, log probability, advantage or utility is used as training data.
- No M16 collection, distributional Critic or variance head is introduced.

## Persistence

Checkpoint-v19 stores Replay schema v5 without utility windows or policy
anchors. Checkpoint-v18/replay-v4 reject before mutable restore; removed
historical outcomes are not migrated, zero-filled or reconstructed.

## Falsifiers

- A replayed Scenario reaches PPO without a fresh current-policy M4 rollout.
- Any earlier visit's utility changes the current Critic target or normalizer.
- Policy mean/sigma, KL, anchor or reset state remains required by Replay.
- Target differs from the row-aligned current M4 arithmetic mean.
- A Scenario with fewer than four lifetime committed K-local visits opens breadth expansion.
- Failed or duplicate commit changes Replay, optimizer or normalizer state.
