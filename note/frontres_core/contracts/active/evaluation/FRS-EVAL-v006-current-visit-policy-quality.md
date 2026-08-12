---
contract_id: FRS-EVAL-v006
status: active
effective_date: 2026-08-12
updated_date: 2026-08-12
supersedes: FRS-EVAL-v005
scope: Clean-anchored local and composition evaluation for checkpoint-v19 current-visit-target policies
---

# Current-Visit-Target Policy Evaluation

Local one-action-K and deployment-composition evaluation preserve EVAL-v005
Clean/Noisy/Repair identity, GAIN-v008 fields, inference-only isolation and
zero training-state mutation. The policy remains a 158D Actor and 449D state
value Critic.

Held-out evaluation reports raw Gain, per-attempt symlog utility, held-out M4
mean and `V(s)`. `V(s)` is interpreted as expected current-policy utility. The
evaluator neither constructs Replay training targets nor loads mutable Replay
state. Replay-v5 may be inspected for checkpoint identity only.

New evidence must prove strict checkpoint-v19 identity, zero Replay mutation,
correct value labeling, and restoration after success and exception.
Checkpoint-v18 and earlier cannot resume TRAIN-v024. Evaluation does not
authorize training and does not infer policy quality from checkpoint loading.

