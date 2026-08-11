---
contract_id: FRS-TRAIN-v021
status: superseded
effective_date: 2026-08-11
updated_date: 2026-08-11
supersedes: FRS-TRAIN-v020
superseded_by: FRS-TRAIN-v022
scope: Historical HSL-v2 Stage-3 campaign with low-DR coupled Actor/Critic adaptation, phase-aware DR-compatible replay and checkpoint-v16
---

# Low-DR Coupled Actor/Critic Cold Start

## Campaign Identity

This is a fresh Stage-3 campaign. HSL-v2 initializes only Actor/std and the
158D prefix normalizer. Critic, optimizer, value normalizer, replay state,
curriculum counters and transaction state start fresh. No TRAIN-v020 or
checkpoint-v15 policy may resume.

```text
method_contract_id       = FRS-METHOD-v022
training_contract_id     = FRS-TRAIN-v021
gain_contract_id         = FRS-GAIN-v008
optimization_contract_id = FRS-PPO-v009
checkpoint_schema        = frontres-v021-checkpoint-v16
```

## Fixed K/M And Coupled Schedule

The five stage counts retain their established numeric budget but change their
first-phase meaning:

```text
K8/M4  : joint_init=200, coupled_ramp=500, joint=1300
K16/M4 : joint_init=300, coupled_ramp=300, joint=900
K32/M4 : joint_init=400, coupled_ramp=300, joint=625
```

For stage-local committed iteration `l`, let
`N_w = N_joint_init + N_coupled_ramp`:

```text
0 <= l < N_joint_init:
  phase = low_dr_joint_init
  actor_loss_weight = (l + 1) / N_w

N_joint_init <= l < N_w:
  phase = coupled_ramp
  actor_loss_weight = (l + 1) / N_w

l >= N_w:
  phase = joint
  actor_loss_weight = 1
```

Actor/std and Critic update from the first committed transaction. There is no
zero Actor weight and no Critic-only phase. The same Critic continues through
all K stages.

## Coupled DR Curriculum

Each K starts from its existing informative lower cap:

```text
K8:  0.50
K16: 0.60
K32: 0.70
reference ceiling: 2.381
advance rule: linear-coupled-v1
advance updates: 700 / 600 / 700
```

For `0 <= l < N_w`, DR progress is `l/N_w`; at `l=N_w` it is one. Thus the
first transaction uses the configured lower cap, Actor influence and `d_cap`
grow together, and `joint` begins at the full current-K distribution. Every
transaction from `l=0` draws Easy/Medium/Hard/Broken-tail with relative weights
20/30/40/10 inside the current cap. A K transition keeps Actor/Critic/optimizer
state, returns to the new K's lower cap and resets Actor weight to its small
nonzero first value.

## Formal Transaction

At transaction start FRS-METHOD-v022 draws each slot's current DR class, then
selects global/replay/review within the current K and absolute DR interval.
The route freezes `pi_old`, materializes two distinct sealed Scenarios, samples
fresh exact-M actions and computes raw Gain plus per-attempt symlog utility.

Before mutation it validates phase, actor weight, `d_cap`, class interval,
stable keys, exact M4 rows, both replay scores and one staged replay delta.
After one grouped optimizer step, one matching receipt atomically commits value
normalizer, replay state, curriculum iteration and checkpoint identity.

## Fixed Optimization

```text
Actor LR  = 3e-6
Critic LR = 1e-5
max_grad_norm per group = 0.5
utility G0 = 1
global/replay/review = 0.40/0.50/0.10
```

Gain, Critic target, Actor advantage, grouped mass and exact-one update remain
FRS-GAIN-v008/FRS-PPO-v009 semantics.

## Diagnostics

Every committed transaction reports phase, stage iteration, Actor weight,
Actor/Critic deltas, DR progress/cap/class/interval, source, Scenario digest,
`E_V`, `E_A`, selected score kind, EMA values, staleness, pool sizes and exact-
one optimizer/replay/checkpoint deltas.

## Persistence And Stop Conditions

Checkpoint-v16 restores exact model, optimizer groups/LRs/step, normalizers,
K/M/DR/phase counters, replay v2 state, RNG and receipt. Strict load rejects
checkpoint-v15 and incompatible schedules before mutable state changes.

Do not start long training if Actor/std fail to change on the first valid
transaction, Critic fails to update, DR and Actor weight are not coupled,
phase-selected replay score is wrong, replay crosses the current DR interval,
exact-one atomicity fails, checkpoint-v16 roundtrip diverges, or the official
Stage-3 entrypoint does not reach these owners.
