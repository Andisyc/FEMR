---
contract_id: FRS-TRAIN-v020
status: superseded
effective_date: 2026-08-10
updated_date: 2026-08-11
supersedes: FRS-TRAIN-v019
scope: Fresh HSL-v2 Stage-3 campaign with M4, fixed split LR, outer prioritized sealed-Scenario replay and checkpoint-v15
---

# Outer Scenario Replay Cold Start

## Campaign Identity

This is a fresh Stage-3 campaign. It may initialize Actor/std and the 158D
prefix normalizer from the accepted HSL-v2 proposal checkpoint. Critic,
optimizer, value normalizer, outer Scenario replay state, curriculum counters
and transaction state start fresh. No TRAIN-v019 checkpoint may resume.

```text
method_contract_id = FRS-METHOD-v021
training_contract_id = FRS-TRAIN-v020
gain_contract_id = FRS-GAIN-v008
optimization_contract_id = FRS-PPO-v008
checkpoint_schema = frontres-v020-checkpoint-v15
```

## Fixed Training Schedule

```text
K8/M4  -> K16/M4 -> K32/M4
Actor LR  = 3e-6
Critic LR = 1e-5
Critic remains the 449D support-conditioned state value
global/replay/review = 0.40/0.50/0.10
```

Critic-only, actor-ramp and joint phase counts, the lower-to-higher per-K DR
schedule, `dr_scale=2.381`, fixed symmetric-log utility `G0=1`, separate
gradient clipping and exact-one grouped Adam update remain TRAIN-v019 values.

## Formal Transaction

At transaction start the outer Scenario replay owner selects two distinct
current-K Scenario sources. Global selections receive a fresh isolated
perturbation seed. Replay/review selections reuse the stored seed and complete
ScenarioKey. The route then freezes the current `pi_old`, materializes both
Scenarios, samples fresh exact-M Repair actions and collects all evidence.

Before optimizer mutation, the route validates:

- two distinct stable ScenarioKeys and exact M4 rows each;
- identical Scenario identity across each M group;
- finite `U(G_m)`, `V_old(s)` and `mean abs(U(G_m)-V_old(s))`;
- no stale action/log-probability/return/advantage carrier;
- one staged replay delta bound to transaction and policy-snapshot identity.

After exactly one optimizer step and its committed receipt, the owner applies
the staged replay delta exactly once. Duplicate receipt application fails
closed. Any pre-commit rejection leaves replay state and replay RNG unchanged.

## Diagnostics

Every committed transaction reports:

- stable ScenarioKey digest, source class and visit transaction;
- perturbation seed, noisy hash, active K and fresh policy snapshot;
- per-Scenario utility mean, old value, mean absolute utility error and EMA;
- replay rank, staleness, replay/review pool sizes and committed visit count;
- exact-one optimizer delta and exact-one replay-state delta.

These diagnostics establish route and lifecycle facts only. Policy quality
still requires a later held-out audit.

## Persistence

Checkpoint-v15 stores the complete v019 model/optimizer/normalizer/curriculum/
RNG/transaction identity plus the outer replay state. Strict load validates the
Contract IDs, schema, ScenarioKey fields, per-K score keys and owner RNG before
restoring any mutable state. Evaluation may load model state read-only but may
not mutate or reconstruct training replay state.

## Stop Conditions

Do not start long training if any of these remains open:

- Scenario materialization is not reproducible from its stored seed;
- replay/review reaches a different Scenario identity;
- rejected or duplicate commits change replay state;
- K-specific priority leaks across K;
- checkpoint-v15 fails strict save/load/first-consumer equivalence;
- official training composition does not reach the new owner.
