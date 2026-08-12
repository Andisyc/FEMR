---
contract_id: FRS-TRAIN-v023
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-TRAIN-v022
scope: Fresh HSL-v2 Stage-3 B8/M4 campaign with policy-compatible robust Critic targets
---
# Robust-Target Low-DR Coupled Cold Start

## Campaign identity

This is a fresh Stage-3 campaign. HSL-v2 initializes only Actor/std and the
158D prefix normalizer. Critic, optimizer, value normalizer, Replay evidence
windows and curriculum state start fresh.

```text
method_contract_id       = FRS-METHOD-v024
training_contract_id     = FRS-TRAIN-v023
gain_contract_id         = FRS-GAIN-v008
optimization_contract_id = FRS-PPO-v011
checkpoint_schema        = frontres-v023-checkpoint-v18
replay_schema            = frontres-outer-scenario-replay-v4
```

No checkpoint-v17 or earlier Stage-3 artifact may resume. Evaluation follows
FRS-EVAL-v005.

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

Actor and Critic update together from the first commit. Every K resets Actor LR
and DR to the existing lower informative point while preserving Actor, Critic
and optimizer state. The DR caps, 20/30/40/10 classes, capacity ladder and B8
slot schedules remain unchanged.

## Formal transaction

One transaction freezes `pi_old`, materializes eight distinct sealed Scenarios
and collects exact M4 current-policy Repair attempts per Scenario. GAIN-v008
and per-attempt symlog produce 32 utilities. Replay previews policy
compatibility, current-window statistics, `E_V_learn` and eight robust Scenario
targets. PPO-v011 uses those targets only for Critic supervision and retains
the 32 current-attempt Actor advantages. Exactly one grouped Adam step commits
optimizer, value normalizer, Replay/window state, DR/K counters and checkpoint
identity together.

## Diagnostics and stop conditions

Each commit reports current-M4 means, robust targets, compatible sample/visit
counts, robust location, outcome variance, SE/h95, excess calibration error,
policy symmetric KL/reset, `E_A`, replay slots, group LRs, gradients/parameters,
DR state and exact-one receipt.

Stop before training on target/Actor row mixing, incompatible-window pooling,
non-atomic Replay target state, a missing diagnostic, checkpoint-v18 mismatch,
any checkpoint-v17 migration, or official route divergence.
