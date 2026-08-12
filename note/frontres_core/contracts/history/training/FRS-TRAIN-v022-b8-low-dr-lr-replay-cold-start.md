---
contract_id: FRS-TRAIN-v022
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FRS-TRAIN-v021
scope: Fresh HSL-v2 Stage-3 B8/M4 campaign with actual Actor-LR and bounded Replay curricula
superseded_by: FRS-TRAIN-v023
---

# B8 Low-DR Coupled Actor/Critic Cold Start

## Campaign identity

This is a fresh Stage-3 campaign. HSL-v2 initializes Actor/std and the 158D
prefix normalizer only. Critic, optimizer, value normalizer, replay and
curriculum state start fresh.

```text
method_contract_id       = FRS-METHOD-v023
training_contract_id     = FRS-TRAIN-v022
gain_contract_id         = FRS-GAIN-v008
optimization_contract_id = FRS-PPO-v010
checkpoint_schema        = frontres-v022-checkpoint-v17
```

No TRAIN-v021/checkpoint-v16 policy may resume.

## K/M/B and coupled schedule

```text
K8/M4/B8  : joint_init=200, coupled_ramp=500, joint=1300
K16/M4/B8 : joint_init=300, coupled_ramp=300, joint=900
K32/M4/B8 : joint_init=400, coupled_ramp=300, joint=625
```

`B8` means eight distinct sealed Scenario states and eight Critic targets per
transaction. Each Scenario contributes four Actor attempts, so one committed
transaction contains 32 policy rows and 64 Repair/Noisy role rows.

Actor and Critic update from the first commit. `actor_loss_weight` is fixed at
one. The actual Actor optimizer-group LR is the coupled authority:

```text
low_dr_joint_init: Actor LR = 3e-7
coupled_ramp:      Actor LR increases linearly from 3e-7 to 1e-6
joint:             Actor LR = 1e-6
Critic LR:         1e-5 throughout
```

Every K resets Actor LR to `3e-7` and DR to its existing lower cap, then ramps
both. The same Actor, Critic and optimizer state continue across K.

## DR and Replay curricula

The K-specific DR caps and 20/30/40/10 four-class distribution are unchanged.
Replay follows FRS-METHOD-v023: capacity 64 during DR adaptation, expansion to
128 then 256 only after full DR and four visits per active Scenario. The B8 slot
schedule is 1-new/6-EV/1-stale in the two adaptation phases and
1-new/4-EA/2-EV/1-stale in joint.

## Formal transaction

The route freezes one `pi_old`, materializes eight distinct sealed Scenarios,
collects exact M4 per Scenario, applies FRS-GAIN-v008 and per-attempt symlog,
then performs one grouped PPO update. Before backward the optimization owner
sets the Actor parameter group's LR to the sealed phase LR and verifies the
Critic group remains `1e-5`. One matching receipt atomically commits optimizer,
value normalizer, Replay Curriculum, DR/K counters and checkpoint identity.

## Diagnostics and stop conditions

Every commit reports B/M/K, phase, actual Actor/Critic group LR, parameter and
gradient deltas, DR state, per-slot replay purpose, active/archive/capacity and
minimum visits, `E_V/E_A`, replacement, exact-one receipt and checkpoint state.

Stop before long training on any non-B8 transaction, loss-weight ramp, wrong
group LR, fewer than eight Critic targets, Replay/DR simultaneous expansion,
missing joint `E_V`, non-atomic failure, checkpoint-v17 mismatch, or official
route divergence.
