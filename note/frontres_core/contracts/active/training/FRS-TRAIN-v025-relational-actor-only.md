---
contract_id: FRS-TRAIN-v025
status: active-pre-training
effective_date: 2026-08-17
updated_date: 2026-08-19
supersedes: FRS-TRAIN-v024-for-relational-route
scope: Fresh Segment Replay campaign using hierarchical relational Gain and Actor-only preference optimization
---
# Relational Segment Replay

This route is a fresh-training migration. It does not resume checkpoint-v19,
because checkpoint-v19 stores scalar Gain, value normalizer, Critic optimizer
state, and scalar Replay priorities.

One transaction selects sealed Scenario identities, executes the same-K Repair
attempts, classifies every valid Repair, builds same-Scenario preference edges,
and commits at most one Actor-only preference step. `SAME`, `INCOMPARABLE`, or invalid
evidence never becomes a zero-valued scalar row.

The outer Replay may retain Scenario identity and edge-density diagnostics, but
it must not reuse scalar Critic calibration or historical utility targets.
Failures roll back Actor, optimizer, Replay, curriculum, and checkpoint state.

FRS-PPO-v014 owns the optimization and the global Actor LR schedule. The first
100 committed updates use `3e-7`, the next 50 ramp to `1e-6`, and subsequent
updates remain at `1e-6`. K transitions do not reset Actor LR or Adam state.

This Contract is admitted only after the migration manifest, module alignment,
code review, and formal pre-training audit are closed. The current engineering
plan stops at that gate; training is explicitly a human-run action.
