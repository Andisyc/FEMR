# FRS-TRAIN-v025 Bounded-1 Relational Telemetry Repair Plan

## Accepted Boundary

- Project a committed FRS-TRAIN-v025 relational update without reading or
  inventing scalar Gain, return, advantage, value loss, Critic target, or
  Critic-normalizer evidence.
- Preserve the complete FRS-TRAIN-v024 scalar telemetry route unchanged.
- Keep telemetry and formal audit read-only; checkpoint-v20 save remains after
  a validated exact-one committed receipt.
- Do not run IsaacLab, training, server operations, Git actions, or destructive
  operations.

## Root Cause And Pinch Point

The third bounded run completed one real relational Actor update, then
`build_frontres_formal_update_summary` read `ppo.value_loss`. The downstream
serializer and public telemetry view also retain scalar-only requirements. The
training-telemetry projection is the Pinch Point between a committed result and
iteration advance/checkpoint save.

## Repair And Proof

1. Characterize the exact committed relational result through
   `require_frontres_committed_result`; it must currently fail at `value_loss`.
2. Add an explicit relational projection from the existing immutable relation
   report and transaction diagnostics. Never zero-fill absent scalar evidence.
3. Extend the public telemetry view and formal final audit with two closed
   identities: relational Actor-only or legacy scalar Actor/Critic.
4. Make the formal train log consume relation edges/counts instead of grouped
   scalar metrics on the relational route.
5. Run exact transaction/telemetry tests, negative mixed-identity tests,
   checkpoint-v20, entrypoint/launcher, aggregate suite, official offline
   preflight, compile, JSON, diff, and final review. Stop before live execution.

## Stop Condition

Stop if the repair requires new Gain/PPO semantics, fabricated scalar fields,
checkpoint schema changes, MOSAIC host behavior, or live execution to establish
an offline fact.
