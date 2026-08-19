---
contract_id: FRS-METHOD-v026
status: active-pre-training
effective_date: 2026-08-17
updated_date: 2026-08-19
supersedes: FRS-METHOD-v025-for-fresh-stage3-training
scope: Hierarchical relational Repair evidence and Actor-only Segment Replay
---
# Relational Repair Method

The active fresh Stage-3 route keeps the existing full-6D FrontRES Actor,
frozen GMT, one-action-K execution, B8 x M4 sealed transactions, K/DR
curriculum, and HSL-v2 initialization. It replaces the scalar Repair objective
with a hierarchical partial order.

For each Repair trajectory, the evidence owner produces one structured Outcome:

\[
O=(S,H,L,I,C),
\]

where \(S\) is survival evidence, \(H\) is the no-load / unplanned-switch /
illegal-contact vector, \(L\) is recovery quality, \(I\) is the Intent error
vector, and \(C\) is full-6D repair cost. The comparator publishes only
`BETTER`, `WORSE`, `SAME`, `INCOMPARABLE`, or `INVALID`. `INVALID` fails the
transaction closed; `SAME` and `INCOMPARABLE` publish no training edge.

Within each sealed Scenario, directed edges \((i,j)\) mean Repair \(i\) is
better than Repair \(j\). FRS-PPO-v014 consumes these edges with an Actor-only
optimizer. It constructs no scalar Gain, return, advantage, Critic target, or
value loss. A transaction with no comparable edge is zero-write and is
recollected without advancing Replay, optimizer, curriculum, or checkpoint.

Outer Replay stores Scenario identity, per-K edge density, visits, staleness,
capacity state, and RNG only. It never stores scalar utility or Critic
calibration. Checkpoint-v20 stores the Actor-only optimizer, frozen
compatibility Critic state, relational Replay, K/DR schedule, RNG, observation
normalizers, GMT identity, and the last completed transaction boundary.

The official selector is:

```text
frontres_training_objective=segment_replay_relational_preference_v014
frontres_relational_actor_only=true
```

The launcher exposes this selector as `MODE=relational_preference_train`. A fresh run must
use the HSL-v2 initializer. A resumed relational run must use checkpoint-v20;
scalar checkpoint-v19 is rejected before mutation.

FRS-METHOD-v025 remains a characterized scalar compatibility route. It is not
the default for the next fresh campaign. Active-pre-training means the
composition root, CPU contracts, and persistence round-trip are
closed. It does not claim simulator execution, threshold calibration,
convergence, policy quality, deployment quality, or Formal PASS.
