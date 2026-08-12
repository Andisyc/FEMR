---
contract_id: FRS-TRAIN-v024
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-TRAIN-v023
scope: Fresh HSL-v2 Stage-3 B8/M4 campaign with current-visit Critic targets
---

# Current-Visit-Target Low-DR Coupled Cold Start

## Campaign identity

```text
method_contract_id       = FRS-METHOD-v025
training_contract_id     = FRS-TRAIN-v024
gain_contract_id         = FRS-GAIN-v008
optimization_contract_id = FRS-PPO-v012
checkpoint_schema        = frontres-v024-checkpoint-v19
replay_schema            = frontres-outer-scenario-replay-v5
```

This is a fresh Stage-3 campaign from HSL-v2 Actor/std and 158D prefix
normalizer. Critic, optimizer, value normalizer, Replay and curriculum start
fresh. Checkpoint-v18 and earlier cannot resume. Evaluation follows
FRS-EVAL-v006.

## Preserved campaign

```text
K8/M4/B8  : joint_init=200, coupled_ramp=500, joint=1300
K16/M4/B8 : joint_init=300, coupled_ramp=300, joint=900
K32/M4/B8 : joint_init=400, coupled_ramp=300, joint=625
low_dr_joint_init Actor LR = 3e-7
coupled_ramp Actor LR       = linear 3e-7 -> 1e-6
joint Actor LR              = 1e-6
Critic LR                   = 1e-5 throughout
```

Actor and Critic update together from the first commit. K transitions preserve
parameters and optimizer state while resetting Actor LR and DR to the lower
informative point. DR classes, 2.381 ceiling, pool capacities and B8 slot
schedules remain unchanged.

## Formal transaction

One transaction selects eight distinct ScenarioKeys, freezes `pi_old`, reruns
every selected Scenario with exact current M4 Repair attempts, computes 32
GAIN-v008 symlog utilities, then forms eight current-M4 means. PPO-v012 uses
those means only for Critic supervision and retains all 32 current-attempt
Actor advantages. Exactly one grouped Adam step commits optimizer, value
normalizer, latest Replay priorities, visits, DR/K counters and checkpoint.

## Diagnostics and stop conditions

Each commit reports current M4 target, M4 outcome variance/SE/h95, current
excess calibration error, current Repair spread, lifetime visit count,
staleness, slots, group LRs, gradient/parameter deltas, DR state and exact-one
receipt. It does not report policy-window compatibility, KL or resets.

Stop before training on stale utility consumption, missing fresh rollout,
target/current-M4 mismatch, non-atomic commit, checkpoint-v19 mismatch, old
checkpoint migration or official-route divergence.

